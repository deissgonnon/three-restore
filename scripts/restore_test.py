#!/usr/bin/env python3
"""Applique un modèle de restauration aux images de test et les enregistre.

Usage:
    python scripts/restore_test.py --model moce_ir
    python scripts/restore_test.py --model moce_ir --weights outputs/best_moce_ir.pth
    python scripts/restore_test.py --model moce_ir --degradations fog rain
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.evaluation.restore import load_restoration_model, restore_folder  # noqa: E402
from src.utils.path_utils import load_paths_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Restaure les images de test avec un modèle")
    parser.add_argument("--model", default="moce_ir", help="Nom du modèle (moce_ir, ...)")
    parser.add_argument("--weights", default=None, help="Chemin des poids (défaut: outputs/best_moce_ir.pth)")
    parser.add_argument("--degradations", nargs="+", default=["fog", "rain", "lowlight"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    paths = load_paths_config(str(ROOT / "configs/paths.yaml"))
    weights = args.weights or str(ROOT / "outputs/best_moce_ir.pth")

    model = load_restoration_model(args.model, weights, args.device)
    print(f"Modèle: {args.model} | device: {args.device} | poids: {weights}")

    for deg in args.degradations:
        src = ROOT / paths["data"][f"synthetic_{deg}"] / "test" / "images"
        dst = ROOT / "data" / "restored" / args.model / deg
        if not src.exists():
            print(f"  [skip] {deg}: dossier absent {src}")
            continue
        n = restore_folder(model, src, dst, args.device)
        print(f"  {deg}: {n} images -> {dst}")


if __name__ == "__main__":
    main()