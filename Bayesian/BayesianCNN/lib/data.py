import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets

from lib.seed import SEED

DATASET_ROOT = "../../Datasets"
DATASET_CONFIGS = {
    "cifar10": {
        "cls": datasets.CIFAR10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "in_channels": 3,
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


def _dataset_to_tensors(raw_dataset, mean, std):
    """Vectorized ToTensor + Normalize over the whole dataset (done once)."""
    data = raw_dataset.data
    if isinstance(data, torch.Tensor):
        # FashionMNIST: uint8 tensor [N, H, W]
        x = data.unsqueeze(1).float().div_(255.0)
    else:
        # CIFAR-10: uint8 numpy array [N, H, W, C]
        x = torch.from_numpy(data).permute(0, 3, 1, 2).float().div_(255.0)
    mean_t = torch.tensor(mean).view(1, -1, 1, 1)
    std_t = torch.tensor(std).view(1, -1, 1, 1)
    x = x.sub_(mean_t).div_(std_t)
    y = torch.as_tensor(raw_dataset.targets, dtype=torch.long)
    return x, y


def build_dataloaders(dataset_name, batch_size, train_frac=0.8, seed=SEED):
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"dataset must be one of {sorted(DATASET_CONFIGS)}, got {dataset_name!r}"
        )
    config = DATASET_CONFIGS[dataset_name]
    dataset_cls = config["cls"]

    training_data_raw = dataset_cls(root=DATASET_ROOT, train=True, download=True)
    test_data_raw = dataset_cls(root=DATASET_ROOT, train=False, download=True)

    x_all, y_all = _dataset_to_tensors(training_data_raw, config["mean"], config["std"])
    x_test, y_test = _dataset_to_tensors(test_data_raw, config["mean"], config["std"])

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
