"""
Pré-traitements pour le dataset SVHN.

Les transformations retournées sont :
- appliquées de manière identique à train / val / test
- invariantes (aucune opération aléatoire)
"""

from torchvision import transforms


def get_preprocess_transforms(config: dict):
    """
    Retourne les transformations de pré-traitement.
    """

    preprocess_cfg = config["preprocess"]

    transform_list = []

    # Resize si spécifié (pas le cas pour SVHN)
    if preprocess_cfg.get("resize") is not None:
        transform_list.append(transforms.Resize(preprocess_cfg["resize"]))

    # Conversion en tenseur
    transform_list.append(transforms.ToTensor())

    # Normalisation
    if preprocess_cfg.get("normalize") is not None:
        mean = preprocess_cfg["normalize"]["mean"]
        std = preprocess_cfg["normalize"]["std"]
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)
