"""Chargement des images et annotations VisDrone.

TODO: implémenter le Dataset PyTorch qui lit data/raw/visdrone/ selon
les chemins définis dans configs/paths.yaml.
"""

from torch.utils.data import Dataset


class VisDroneDataset(Dataset):
    """Dataset PyTorch pour les images/annotations VisDrone."""

    def __init__(self, root_dir: str, split: str = "train"):
        self.root_dir = root_dir
        self.split = split

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError
