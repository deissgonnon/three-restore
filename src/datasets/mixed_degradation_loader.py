"""Loader pour les paires d'images (dégradée -> claire) avec mélange des dégradations."""

import os
import random
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class MixedDegradationDataset(Dataset):
    """
    Dataset qui charge une image claire et sélectionne aléatoirement une dégradation 
    synthétique correspondante (brouillard, pluie ou faible luminosité).
    Ne modifie pas la géométrie (pas de crop/flip) pour préserver les boîtes englobantes.
    """

    def __init__(self, splits_dir: str, synthetic_dirs: dict, split: str = "train", transform=None):
        """
        Args:
            splits_dir: Chemin vers les images claires (ex: data/splits).
            synthetic_dirs: Dictionnaire des chemins de dégradation.
                            ex: {"fog": "data/synthetic/fog", ...}
            split: "train", "val", ou "test".
            transform: Transformations optionnelles.
        """
        self.splits_dir = Path(splits_dir)
        self.synthetic_dirs = {k: Path(v) for k, v in synthetic_dirs.items()}
        self.split = split
        
        # Trouver toutes les images dans le dossier du split (on suppose data/splits/train/images)
        self.split_path = self.splits_dir / split / "images"
        if not self.split_path.exists():
            # Essayer d'autres structures courantes si celle-ci n'existe pas
            self.split_path = self.splits_dir / "images" / split
            if not self.split_path.exists():
                self.split_path = self.splits_dir # Fallback
                
        self.clean_images = sorted(list(self.split_path.rglob("*.jpg")) + list(self.split_path.rglob("*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.clean_images)

    def _pad_image(self, img, multiple=16):
        """Pad image to make its dimensions divisible by `multiple` (required for MoCE-IR)."""
        w, h = img.size
        pad_w = (multiple - w % multiple) % multiple
        pad_h = (multiple - h % multiple) % multiple
        if pad_w > 0 or pad_h > 0:
            # Padding is added to the right and bottom, so top-left coords (0,0) remain unchanged.
            # This preserves bounding box coordinates without needing to adjust them.
            img = TF.pad(img, (0, 0, pad_w, pad_h), padding_mode='reflect')
        return img

    def __getitem__(self, idx):
        clean_path = self.clean_images[idx]
        
        # Choisir aléatoirement une dégradation
        deg_type = random.choice(list(self.synthetic_dirs.keys()))
        deg_base_dir = self.synthetic_dirs[deg_type]
        
        # Construire le chemin vers l'image dégradée en remplaçant la racine
        # ex: data/splits/train/images/001.jpg -> data/synthetic/fog/train/images/001.jpg
        rel_path = clean_path.relative_to(self.splits_dir)
        deg_path = deg_base_dir / rel_path

        # Si l'image n'est pas encore générée, on utilise l'image claire (fallback pour éviter de crasher)
        if not deg_path.exists():
            deg_path = clean_path
            
        clean_img = Image.open(clean_path).convert("RGB")
        deg_img = Image.open(deg_path).convert("RGB")

        # Padding pour être divisible par 16 (prérequis pour MoCE-IR)
        clean_img = self._pad_image(clean_img, multiple=16)
        deg_img = self._pad_image(deg_img, multiple=16)

        clean_tensor = TF.to_tensor(clean_img)
        deg_tensor = TF.to_tensor(deg_img)

        # Les boîtes englobantes ne sont pas modifiées ici.
        return deg_tensor, clean_tensor
