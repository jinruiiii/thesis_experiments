import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets

from lib.seed import SEED

def seed_worker(worker_id):
    """Function to ensure DataLoader workers use different seeds derived from the base seed"""
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


DATASET_ROOT = "../../Datasets"
SUPPORTED_DATASETS = {
    "fashion_mnist": datasets.FashionMNIST,
    "kmnist": datasets.KMNIST,
    "cifar10": datasets.CIFAR10,
}

DATASET_METADATA = {
    "fashion_mnist": {"input_dim": 784, "num_classes": 10},
    "kmnist": {"input_dim": 784, "num_classes": 10},
    "cifar10": {"input_dim": 1024, "num_classes": 10},
}

_CIFAR10_GREYSCALE_WEIGHTS = torch.tensor([0.299, 0.587, 0.114])


def get_dataset_metadata(dataset_name):
    if dataset_name not in DATASET_METADATA:
        raise ValueError(
            f"dataset must be one of {sorted(DATASET_METADATA)}, got {dataset_name!r}"
        )
    return dict(DATASET_METADATA[dataset_name])


def _mnist_to_flat_tensors(raw_dataset):
    """Vectorized ToTensor + flatten over the whole dataset (done once)."""
    x = raw_dataset.data.float().div_(255.0).flatten(1)
    y = torch.as_tensor(raw_dataset.targets, dtype=torch.long)
    return x, y


def _cifar10_greyscale_to_flat_tensors(raw_dataset):
    """Convert CIFAR-10 RGB images to greyscale and flatten."""
    x = torch.from_numpy(raw_dataset.data).float().div_(255.0)
    x = x.matmul(_CIFAR10_GREYSCALE_WEIGHTS)
    x = x.flatten(1)
    y = torch.as_tensor(raw_dataset.targets, dtype=torch.long)
    return x, y


def _dataset_to_flat_tensors(raw_dataset, dataset_name):
    if dataset_name in ("fashion_mnist", "kmnist"):
        return _mnist_to_flat_tensors(raw_dataset)
    if dataset_name == "cifar10":
        return _cifar10_greyscale_to_flat_tensors(raw_dataset)
    raise ValueError(
        f"dataset must be one of {sorted(SUPPORTED_DATASETS)}, got {dataset_name!r}"
    )


def build_dataloaders(dataset_name, batch_size, train_frac=0.8, seed=SEED):
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"dataset must be one of {sorted(SUPPORTED_DATASETS)}, got {dataset_name!r}"
        )

    dataset_cls = SUPPORTED_DATASETS[dataset_name]
    expected_input_dim = DATASET_METADATA[dataset_name]["input_dim"]

    training_data_raw = dataset_cls(root=DATASET_ROOT, train=True, download=True)
    test_data_raw = dataset_cls(root=DATASET_ROOT, train=False, download=True)

    x_all, y_all = _dataset_to_flat_tensors(training_data_raw, dataset_name)
    x_test, y_test = _dataset_to_flat_tensors(test_data_raw, dataset_name)
    assert x_all.shape[1] == expected_input_dim
    assert x_test.shape[1] == expected_input_dim

    train_size = int(train_frac * len(training_data_raw))
    val_size = len(training_data_raw) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(len(training_data_raw)), [train_size, val_size], generator=generator
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
