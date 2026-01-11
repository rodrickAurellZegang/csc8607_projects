"""
Utils génériques.

Fonctions utilitaires pour :
- reproductibilité
- sélection du device
- comptage des paramètres
- sauvegarde de la configuration
"""

import os
import random
import yaml
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Initialise les seeds pour assurer la reproductibilité.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Rendre les opérations déterministes (au prix de performances)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer: str | None = "auto") -> str:
    """
    Retourne 'cpu' ou 'cuda' selon la disponibilité et la préférence.
    """
    if prefer == "cpu":
        return "cpu"
    if prefer == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"

    # mode auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def count_parameters(model) -> int:
    """
    Retourne le nombre de paramètres entraînables du modèle.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_config_snapshot(config: dict, out_dir: str) -> None:
    """
    Sauvegarde une copie de la configuration YAML dans out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "config_snapshot.yaml")

    with open(out_path, "w") as f:
        yaml.safe_dump(config, f)
