"""
Chargement des données.

Signature imposée :
get_dataloaders(config: dict) -> (train_loader, val_loader, test_loader, meta: dict)

Le dictionnaire meta doit contenir au minimum :
- "num_classes": int
- "input_shape": tuple (ex: (3, 32, 32) pour des images)
"""

from typing import Tuple, Dict
import numpy as np

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import SVHN

from src.preprocessing import get_preprocess_transforms


def get_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Crée et retourne les DataLoaders d'entraînement/validation/test et des métadonnées.
    """

    # Prétraitements (identiques train / val / test pour M2)
    preprocess = get_preprocess_transforms(config)

    # Dataset SVHN
    full_train = SVHN(
        root=config["dataset"]["root"],
        split="train",
        download=config["dataset"]["download"],
        transform=preprocess,
    )

    test_set = SVHN(
        root=config["dataset"]["root"],
        split="test",
        download=config["dataset"]["download"],
        transform=preprocess,
    )

    # Split validation (10 % du train)
    seed = config["train"]["seed"]
    rng = np.random.default_rng(seed)

    n_total = len(full_train)
    n_val = int(0.1 * n_total)

    indices = rng.permutation(n_total)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_set = Subset(full_train, train_indices)
    val_set = Subset(full_train, val_indices)

    # DataLoaders
    batch_size = config["train"]["batch_size"]
    num_workers = config["dataset"]["num_workers"]

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Métadonnées
    meta = {
        "num_classes": config["model"]["num_classes"],
        "input_shape": tuple(config["model"]["input_shape"]),
    }

    return train_loader, val_loader, test_loader, meta
