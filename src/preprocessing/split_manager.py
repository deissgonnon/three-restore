"""Gestion des splits train/val/test figés

Les splits, une fois générés, sont sauvegardés dans data/splits/ et ne
doivent plus être modifiés d'une expérience à l'autre.
"""


def load_split(split_name: str, splits_dir: str):
    """Charge la liste d'images pour un split donné depuis data/splits/."""
    raise NotImplementedError


def create_splits(image_list, ratios: dict, seed: int, splits_dir: str):
    """Génère et sauvegarde les splits train/val/test (une seule fois)."""
    raise NotImplementedError
