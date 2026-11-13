"""Application d'un modèle de restauration aux images de test.

Contient la logique réutilisable (chargement du modèle, prétraitement,
inférence, sauvegarde) pour n'importe quel modèle : MoCE-IR, DEP-RNet, etc.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def load_image(path: Path) -> np.ndarray:
    """Charge une image en RGB (HWC, uint8)."""
    return np.array(Image.open(path).convert("RGB"))


def to_model_input(img: np.ndarray, device: str, divisible: int = 16) -> torch.Tensor:
    """Convertit une image HWC en tensor [1,3,H,W] dans [0,1], H,W divisibles par `divisible`."""
    h, w = img.shape[:2]
    h = h - (h % divisible)
    w = w - (w % divisible)
    img = img[:h, :w]
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    return t


def to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convertit un tensor [1,3,H,W] en image HWC uint8."""
    arr = tensor[0].permute(1, 2, 0).cpu().numpy()
    arr = np.clip(arr, 0, 1)
    return (arr * 255.0).astype(np.uint8)


def restore_folder(model: torch.nn.Module, input_dir: Path, output_dir: Path,
                   device: str, extensions=("*.jpg", "*.png")) -> int:
    """Applique le modèle à toutes les images de input_dir et les écrit dans output_dir.

    Retourne le nombre d'images traitées.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for ext in extensions for p in input_dir.glob(ext))

    model.eval()
    with torch.no_grad():
        for p in tqdm(paths, desc=f"Restauration {input_dir.parent.name}"):
            img = load_image(p)
            inp = to_model_input(img, device)
            out = model(inp)
            Image.fromarray(to_image(out)).save(output_dir / p.name)
    return len(paths)


def load_restoration_model(name: str, weights: str, device: str) -> torch.nn.Module:
    """Charge un modèle de restauration par nom et retourne le module PyTorch.

    Supporte "moce_ir" ; étendre ici pour ajouter d'autres modèles.
    """
    if name == "moce_ir":
        import yaml

        from src.baselines.moce_ir import MoceIrBaseline

        root = Path(__file__).resolve().parent.parent.parent
        with open(root / "configs/baselines.yaml") as f:
            cfg = yaml.safe_load(f)["baselines"]["all_in_one"]["moce_ir"]
        kwargs = {k: v for k, v in cfg.items() if k not in ["weights_path"]}
        wrapper = MoceIrBaseline(weights_path=weights, **kwargs)
        return wrapper.model.to(device)
    raise ValueError(f"Modèle de restauration non supporté: {name}")