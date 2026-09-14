"""Suivi visuel de l'entraînement : dégrade un petit échantillon fixe d'images
du val set et sauvegarde les restaurations du modèle à chaque époque.

Permet de voir la progression de l'entraînement dans outputs/figures/epoch_XX/.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image, pad as tf_pad

from ..degradation.degradation_pipeline import DegradationPipeline


def _pad_to_multiple(img, multiple: int = 16):
    """Pad une image PIL pour que ses dimensions soient divisibles par `multiple`.

    Prérequis pour MoCE-IR (downsampling successifs par 2). Le padding est ajouté
    à droite et en bas pour ne pas décaler les coordonnées.
    """
    w, h = img.size
    pad_w = (multiple - w % multiple) % multiple
    pad_h = (multiple - h % multiple) % multiple
    if pad_w > 0 or pad_h > 0:
        img = tf_pad(img, (0, 0, pad_w, pad_h), padding_mode='reflect')
    return img



def select_validation_samples(val_dataset, n_per_degradation: int = 2,
                              seed: int = 42) -> list:
    """Sélectionne un échantillon fixe d'images claires du val set.

    Retourne une liste de dicts : {"name", "clean", "degraded", "de_type"}.
    - ``clean`` : image claire (tensor 3xHxW, [0,1]).
    - ``degraded`` : image dégradée (tensor 3xHxW, [0,1]).
    - ``de_type`` : nom de la dégradation ("fog", "rain", "lowlight").
    """
    # Récupère les paires (dégradée, claire) du dataset, groupées par type.
    pairs_by_type = {}
    for pair in val_dataset.pairs:
        de_type = pair["de_type"]
        pairs_by_type.setdefault(de_type, []).append(pair)

    # Nom des tâches dans l'ordre trié (même logique que MoCEIRDataset).
    task_names = sorted(val_dataset.synthetic_dirs.keys())

    rng = np.random.default_rng(seed)
    samples = []
    for de_id, task in enumerate(task_names):
        pairs = pairs_by_type.get(de_id, [])
        if not pairs:
            continue
        # Sélection déterministe : n_per_degradation paires distinctes.
        chosen = rng.choice(len(pairs), size=min(n_per_degradation, len(pairs)),
                            replace=False)
        for i in chosen:
            pair = pairs[int(i)]
            clean = Image.open(pair["clean"]).convert("RGB")
            degraded = Image.open(pair["img"]).convert("RGB")
            # Padding pour être divisible par 16 (prérequis pour MoCE-IR),
            # cohérent avec le loader d'entraînement.
            clean = _pad_to_multiple(clean, multiple=16)
            degraded = _pad_to_multiple(degraded, multiple=16)
            name = Path(pair["clean"]).stem
            samples.append({
                "name": f"{task}_{name}_{int(i)}",
                "clean": to_tensor(clean),
                "degraded": to_tensor(degraded),
                "de_type": task,
            })
    return samples


def save_epoch_samples(model, samples: list, epoch: int, output_dir: str,
                       device: str = "cuda") -> None:
    """Restaure les échantillons avec le modèle et sauvegarde les images.

    Pour chaque échantillon, sauvegarde 3 fichiers dans output_dir/epoch_XX/ :
    - ``{name}_clean.png`` : image claire (référence).
    - ``{name}_degraded.png`` : image dégradée (entrée).
    - ``{name}_restored.png`` : sortie du modèle.
    """
    out_root = Path(output_dir) / f"epoch_{epoch:04d}"
    out_root.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for sample in samples:
            degraded = sample["degraded"].unsqueeze(0).to(device)
            restored = model(degraded).squeeze(0).cpu()

            to_pil_image(sample["clean"]).save(out_root / f"{sample['name']}_clean.png")
            to_pil_image(sample["degraded"]).save(out_root / f"{sample['name']}_degraded.png")
            to_pil_image(restored.clamp(0, 1)).save(out_root / f"{sample['name']}_restored.png")

    print(f"Échantillons de l'époque {epoch} sauvegardés dans {out_root}")