"""Dataloader dédié à la baseline MoCE-IR, calqué sur l'officiel (AIOTrainDataset).

"""

import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


def _data_augmentation(image, mode):
    """8 transformations géométriques, copiées de l'officiel (image HWC numpy)."""
    if mode == 0:
        out = image
    elif mode == 1:
        out = np.flipud(image)
    elif mode == 2:
        out = np.rot90(image)
    elif mode == 3:
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        out = np.rot90(image, k=2)
    elif mode == 5:
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        out = np.rot90(image, k=3)
    elif mode == 7:
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise ValueError(f"Mode d'augmentation invalide: {mode}")
    return out


def _random_augmentation(*args):
    # Comme dans l'officiel : randint(1, 7) -> le mode 0 (identité) n'est jamais tiré.
    flag_aug = random.randint(1, 7)
    out = []
    for data in args:
        out.append(_data_augmentation(data, flag_aug).copy())
    return out


class MoCEIRDataset(Dataset):
    """Paires (dégradée -> claire) fusionnées sur toutes les dégradations, pour MoCE-IR.
    """

    def __init__(self, splits_dir: str, synthetic_dirs: dict, split: str = "train",
                 patch_size: int = 128):
        """
        Args:
            splits_dir: Racine des images claires (ex: data/splits).
            synthetic_dirs: {"fog": "data/synthetic/fog", ...}, une entrée par tâche.
            split: "train", "val" ou "test".
            patch_size: Taille du crop (128 par défaut, comme l'officiel).
        """
        self.splits_dir = Path(splits_dir)
        self.synthetic_dirs = {k: Path(v) for k, v in synthetic_dirs.items()}
        self.split = split
        self.patch_size = patch_size

        # Résolution du dossier d'images claires (mêmes fallbacks que MixedDegradationDataset)
        split_path = self.splits_dir / split / "images"
        if not split_path.exists():
            split_path = self.splits_dir / "images" / split
            if not split_path.exists():
                split_path = self.splits_dir
        self.clean_images = sorted(list(split_path.rglob("*.jpg")) + list(split_path.rglob("*.png")))

        self.pairs = self._init_pairs()
        if len(self.pairs) == 0:
            raise ValueError(
                f"[MoCEIRDataset] Aucune paire (dégradée, claire) trouvée pour split={split} "
                f"dans {splits_dir} + {list(synthetic_dirs.keys())}")

    def _init_pairs(self):
        """Fusionne les tâches : une entrée par paire (dégradée, claire) existante."""
        pairs = []
        # sorted -> ids de tâche déterministes (ex: fog=0, lowlight=1, rain=2)
        for de_id, (task, base_dir) in enumerate(sorted(self.synthetic_dirs.items())):
            n_pairs = 0
            for clean_path in self.clean_images:
                rel_path = clean_path.relative_to(self.splits_dir)
                deg_path = base_dir / rel_path
                if not deg_path.exists():
                    continue
                pairs.append({"img": str(deg_path), "clean": str(clean_path), "de_type": de_id})
                n_pairs += 1
            print(f"[MoCEIRDataset] {self.split}/{task}: {n_pairs} paires")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        lr = np.array(Image.open(pair["img"]).convert("RGB"))
        hr = np.array(Image.open(pair["clean"]).convert("RGB"))

        if self.split == "train":
            lr, hr = _random_augmentation(*self._crop_patch(lr, hr))
        else:
            lr, hr = self._center_patch(lr, hr)

        lr = to_tensor(lr)
        hr = to_tensor(hr)
        return [pair["img"], pair["de_type"]], lr, hr

    def _ensure_min_size(self, img_1, img_2):
        """Redimensionne si l'image est plus petite que patch_size (l'officiel planterait)."""
        h, w = img_1.shape[0], img_1.shape[1]
        if h < self.patch_size or w < self.patch_size:
            scale = self.patch_size / min(h, w)
            new_size = (round(w * scale), round(h * scale))
            img_1 = np.array(Image.fromarray(img_1).resize(new_size, Image.BILINEAR))
            img_2 = np.array(Image.fromarray(img_2).resize(new_size, Image.BILINEAR))
        return img_1, img_2

    def _crop_patch(self, img_1, img_2):
        """Crop aléatoire au même endroit pour lr et hr (comme l'officiel)."""
        img_1, img_2 = self._ensure_min_size(img_1, img_2)
        h, w = img_1.shape[0], img_1.shape[1]
        ind_h = random.randint(0, h - self.patch_size)
        ind_w = random.randint(0, w - self.patch_size)
        return (img_1[ind_h:ind_h + self.patch_size, ind_w:ind_w + self.patch_size],
                img_2[ind_h:ind_h + self.patch_size, ind_w:ind_w + self.patch_size])

    def _center_patch(self, img_1, img_2):
        """Crop centré déterministe (même zone pour lr et hr) pour la validation."""
        img_1, img_2 = self._ensure_min_size(img_1, img_2)
        h, w = img_1.shape[0], img_1.shape[1]
        top = (h - self.patch_size) // 2
        left = (w - self.patch_size) // 2
        return (img_1[top:top + self.patch_size, left:left + self.patch_size],
                img_2[top:top + self.patch_size, left:left + self.patch_size])
