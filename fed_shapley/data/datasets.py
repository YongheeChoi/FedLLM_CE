"""
data/datasets.py — Dataset loading for CIFAR-10, CIFAR-100, and Tiny ImageNet.

Standard data augmentation transforms are applied:
- CIFAR (32x32): RandomCrop(32, pad=4), RandomHorizontalFlip, Normalize
- TinyImageNet (64x64): RandomCrop(64, pad=8), RandomHorizontalFlip, Normalize

Tiny ImageNet is downloaded on demand if not already present.
"""

import os
import zipfile
import shutil
import urllib.request
from typing import Tuple

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms


# ------------------------------------------------------------------------------
# Normalization statistics (mean, std) per dataset
# ------------------------------------------------------------------------------
_STATS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
    "tinyimagenet": {
        "mean": (0.4802, 0.4481, 0.3975),
        "std": (0.2770, 0.2691, 0.2821),
    },
}

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def _download_tinyimagenet(data_dir: str) -> str:
    """Download and extract Tiny ImageNet if not already present.

    Args:
        data_dir: Root directory where data will be stored.

    Returns:
        Path to the extracted tiny-imagenet-200 directory.
    """
    root = os.path.join(data_dir, "tiny-imagenet-200")
    if os.path.isdir(root):
        print(f"[Dataset] Tiny ImageNet already at {root}")
        return root

    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")
    os.makedirs(data_dir, exist_ok=True)

    if not os.path.isfile(zip_path):
        print(f"[Dataset] Downloading Tiny ImageNet from {TINY_IMAGENET_URL} ...")
        urllib.request.urlretrieve(TINY_IMAGENET_URL, zip_path)
        print("[Dataset] Download complete.")

    print("[Dataset] Extracting Tiny ImageNet ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    # Restructure validation set for ImageFolder compatibility
    _restructure_tinyimagenet_val(root)
    print(f"[Dataset] Tiny ImageNet ready at {root}")
    return root


def _restructure_tinyimagenet_val(root: str) -> None:
    """Reorganize Tiny ImageNet validation set into class subdirectories.

    The default validation structure stores all images in val/images/ with
    a single annotation file. ImageFolder requires val/<class>/<img> layout.

    Args:
        root: Path to extracted tiny-imagenet-200 directory.
    """
    val_dir = os.path.join(root, "val")
    ann_file = os.path.join(val_dir, "val_annotations.txt")

    if not os.path.isfile(ann_file):
        return  # Already restructured or missing

    # Parse annotation file: filename → class_id
    img_to_class = {}
    with open(ann_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                img_to_class[parts[0]] = parts[1]

    images_dir = os.path.join(val_dir, "images")
    for img_name, class_id in img_to_class.items():
        src = os.path.join(images_dir, img_name)
        dst_dir = os.path.join(val_dir, class_id)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, img_name)
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.move(src, dst)

    # Remove empty images dir and annotation file
    if os.path.isdir(images_dir) and not os.listdir(images_dir):
        os.rmdir(images_dir)
    if os.path.isfile(ann_file):
        os.remove(ann_file)


def load_dataset(
    dataset_name: str, data_dir: str
) -> Tuple[Dataset, Dataset]:
    """Load a dataset with standard training and test transforms.

    Args:
        dataset_name: One of 'cifar10', 'cifar100', 'tinyimagenet'.
        data_dir: Root directory for dataset storage/download.

    Returns:
        Tuple of (train_dataset, test_dataset) with transforms applied.

    Raises:
        ValueError: If dataset_name is not recognized.
    """
    os.makedirs(data_dir, exist_ok=True)
    stats = _STATS[dataset_name]
    mean, std = stats["mean"], stats["std"]

    if dataset_name in ("cifar10", "cifar100"):
        # CIFAR-10 / CIFAR-100: 32x32 images
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        cls = torchvision.datasets.CIFAR10 if dataset_name == "cifar10" \
            else torchvision.datasets.CIFAR100
        train_ds = cls(root=data_dir, train=True, download=True, transform=train_transform)
        test_ds = cls(root=data_dir, train=False, download=True, transform=test_transform)

    elif dataset_name == "tinyimagenet":
        # Tiny ImageNet: 64x64 images, 200 classes
        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        root = _download_tinyimagenet(data_dir)
        train_ds = torchvision.datasets.ImageFolder(
            root=os.path.join(root, "train"),
            transform=train_transform,
        )
        test_ds = torchvision.datasets.ImageFolder(
            root=os.path.join(root, "val"),
            transform=test_transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Choose from: cifar10, cifar100, tinyimagenet.")

    return train_ds, test_ds


def get_num_classes(dataset_name: str) -> int:
    """Return the number of classes for a given dataset.

    Args:
        dataset_name: One of 'cifar10', 'cifar100', 'tinyimagenet'.

    Returns:
        Integer number of classes.

    Raises:
        ValueError: If dataset_name is not recognized.
    """
    mapping = {
        "cifar10": 10,
        "cifar100": 100,
        "tinyimagenet": 200,
    }
    if dataset_name not in mapping:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return mapping[dataset_name]
