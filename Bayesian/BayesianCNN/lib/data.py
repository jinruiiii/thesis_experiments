import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets

from lib.seed import SEED

DATASET_ROOT = "../../Datasets"
_CIFAR10_GREYSCALE_WEIGHTS = torch.tensor([0.299, 0.587, 0.114])
_CIFAR10_RGB_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_RGB_STD = (0.2023, 0.1994, 0.2010)

DATASET_CONFIGS = {
    "cifar10": {
        "cls": datasets.CIFAR10,
        "mean": (0.4807,),
        "std": (0.2512,),
        "in_channels": 1,
        "num_classes": 10,
    },
    "fashion_mnist": {
        "cls": datasets.FashionMNIST,
        "mean": (0.2860,),
        "std": (0.3530,),
        "in_channels": 1,
        "num_classes": 10,
    },
}


def get_dataset_input_spec(dataset_name, cifar10_grayscale=True):
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"dataset must be one of {sorted(DATASET_CONFIGS)}, got {dataset_name!r}"
        )
    config = DATASET_CONFIGS[dataset_name]
    if dataset_name == "cifar10" and not cifar10_grayscale:
        return {
            "mean": _CIFAR10_RGB_MEAN,
            "std": _CIFAR10_RGB_STD,
            "in_channels": 3,
        }
    return {
        "mean": config["mean"],
        "std": config["std"],
        "in_channels": config["in_channels"],
    }


def _normalize_tensors(x, mean, std):
    mean_t = torch.tensor(mean).view(1, -1, 1, 1)
    std_t = torch.tensor(std).view(1, -1, 1, 1)
    return x.sub_(mean_t).div_(std_t)


def _fashion_mnist_to_tensors(raw_dataset, mean, std):
    """Vectorized ToTensor + Normalize over the whole dataset (done once)."""
    x = raw_dataset.data.unsqueeze(1).float().div_(255.0)
    x = _normalize_tensors(x, mean, std)
    y = torch.as_tensor(raw_dataset.targets, dtype=torch.long)
    return x, y


def _cifar10_greyscale_to_tensors(raw_dataset, mean, std):
    """Convert CIFAR-10 RGB to greyscale NCHW and normalize."""
    x = torch.from_numpy(raw_dataset.data).float().div_(255.0)
    x = x.matmul(_CIFAR10_GREYSCALE_WEIGHTS)
    x = x.unsqueeze(1)
    x = _normalize_tensors(x, mean, std)
    y = torch.as_tensor(raw_dataset.targets, dtype=torch.long)
    return x, y


def _cifar10_rgb_to_tensors(raw_dataset, mean, std):
    """Convert CIFAR-10 RGB to NCHW and normalize."""
    x = torch.from_numpy(raw_dataset.data).float().div_(255.0)
    x = x.permute(0, 3, 1, 2)
    x = _normalize_tensors(x, mean, std)
    y = torch.as_tensor(raw_dataset.targets, dtype=torch.long)
    return x, y


def _dataset_to_tensors(raw_dataset, dataset_name, mean, std, cifar10_grayscale=True):
    if dataset_name == "fashion_mnist":
        return _fashion_mnist_to_tensors(raw_dataset, mean, std)
    if dataset_name == "cifar10":
        if cifar10_grayscale:
            return _cifar10_greyscale_to_tensors(raw_dataset, mean, std)
        return _cifar10_rgb_to_tensors(raw_dataset, mean, std)
    raise ValueError(
        f"dataset must be one of {sorted(DATASET_CONFIGS)}, got {dataset_name!r}"
    )


def build_dataloaders(
    dataset_name, batch_size, train_frac=0.8, seed=SEED, cifar10_grayscale=True
):
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"dataset must be one of {sorted(DATASET_CONFIGS)}, got {dataset_name!r}"
        )
    config = DATASET_CONFIGS[dataset_name]
    dataset_cls = config["cls"]
    input_spec = get_dataset_input_spec(dataset_name, cifar10_grayscale=cifar10_grayscale)
    expected_in_channels = input_spec["in_channels"]

    training_data_raw = dataset_cls(root=DATASET_ROOT, train=True, download=True)
    test_data_raw = dataset_cls(root=DATASET_ROOT, train=False, download=True)

    x_all, y_all = _dataset_to_tensors(
        training_data_raw,
        dataset_name,
        input_spec["mean"],
        input_spec["std"],
        cifar10_grayscale=cifar10_grayscale,
    )
    x_test, y_test = _dataset_to_tensors(
        test_data_raw,
        dataset_name,
        input_spec["mean"],
        input_spec["std"],
        cifar10_grayscale=cifar10_grayscale,
    )
    assert x_all.shape[1] == expected_in_channels
    assert x_test.shape[1] == expected_in_channels
    if dataset_name == "cifar10":
        assert x_all.shape[2:] == (32, 32)
        assert x_test.shape[2:] == (32, 32)

    train_size = int(train_frac * len(training_data_raw))
    val_size = len(training_data_raw) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(len(training_data_raw)),
        [train_size, val_size],
        generator=generator,
    )
    train_idx = torch.tensor(train_indices.indices, dtype=torch.long)
    val_idx = torch.tensor(val_indices.indices, dtype=torch.long)
    train_dataset = TensorDataset(x_all[train_idx], y_all[train_idx])
    val_dataset = TensorDataset(x_all[val_idx], y_all[val_idx])
    test_dataset = TensorDataset(x_test, y_test)

    pin_memory = torch.cuda.is_available()
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=pin_memory,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        generator=g,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        generator=g,
    )
    return train_loader, val_loader, test_loader
