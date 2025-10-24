"""Utility helpers for loading MedMNIST datasets and dataloaders.

This module centralises the repeated boilerplate used across the notebooks
when preparing MedMNIST datasets.  It exposes a small convenience API that
returns the train/test datasets alongside ready-to-use ``DataLoader``
instances, ensuring that notebooks can focus on the experiments themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


__all__ = [
    "MedMNISTDataBundle",
    "default_medmnist_transform",
    "load_medmnist_data",
]


@dataclass
class MedMNISTDataBundle:
    """Container with the datasets, loaders and metadata for a MedMNIST split."""

    data_flag: str
    info: dict
    train_dataset: Dataset
    test_dataset: Dataset
    train_loader: DataLoader
    test_loader: DataLoader

    @property
    def num_classes(self) -> int:
        """Return the number of classes defined by the MedMNIST metadata."""

        labels = self.info.get("label", {})
        return len(labels)

    @property
    def label_names(self) -> dict:
        """Expose the mapping ``label_id -> label_name`` for convenience."""

        return self.info.get("label", {})


def default_medmnist_transform() -> transforms.Compose:
    """Return the default tensor/normalisation pipeline used in the notebooks."""

    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )


def _build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def load_medmnist_data(
    *,
    data_flag: str = "breastmnist",
    batch_size: int = 128,
    download: bool = True,
    transform: Optional[transforms.Compose] = None,
    target_transform: Optional[transforms.Compose] = None,
    train_split: str = "train",
    test_split: str = "test",
    shuffle_train: bool = True,
    shuffle_test: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> MedMNISTDataBundle:
    """Load MedMNIST datasets and dataloaders with sensible defaults."""

    if data_flag not in INFO:
        raise KeyError(f"Unknown MedMNIST dataset flag: {data_flag!r}")

    info = INFO[data_flag]
    dataset_class = getattr(medmnist, info["python_class"])

    transform = transform or default_medmnist_transform()

    train_dataset = dataset_class(
        split=train_split,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )
    test_dataset = dataset_class(
        split=test_split,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )

    train_loader = _build_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    test_loader = _build_loader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return MedMNISTDataBundle(
        data_flag=data_flag,
        info=info,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        test_loader=test_loader,
    )
