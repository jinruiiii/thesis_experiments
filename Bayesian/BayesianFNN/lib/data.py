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
}


def _dataset_to_flat_tensors(raw_dataset):
    """Vectorized ToTensor + flatten over the whole dataset (done once)."""
    x = raw_dataset.data.float().div_(255.0).flatten(1)
    y = torch.as_tensor(raw_dataset.targets, dtype=torch.long)
    return x, y


def build_dataloaders(dataset_name, batch_size, train_frac=0.8, seed=SEED):
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"dataset must be one of {sorted(SUPPORTED_DATASETS)}, got {dataset_name!r}"
        )

    dataset_cls = SUPPORTED_DATASETS[dataset_name]

    training_data_raw = dataset_cls(root=DATASET_ROOT, train=True, download=True)
    test_data_raw = dataset_cls(root=DATASET_ROOT, train=False, download=True)

    x_all, y_all = _dataset_to_flat_tensors(training_data_raw)
    x_test, y_test = _dataset_to_flat_tensors(test_data_raw)

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
