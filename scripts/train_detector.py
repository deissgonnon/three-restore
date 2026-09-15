#!/usr/bin/env python3
"""Entraîne le détecteur fixe sur le split train VisDrone.

Le détecteur entraîné sert de référence FIXE pour l'évaluation du module
de restauration (image nette / dégradée / restaurée) — voir doc de suivi §6.4.

Usage:
    python scripts/train_detector.py
    python scripts/train_detector.py --epochs 50 --batch 32 --model yolov8s.pt
    python scripts/train_detector.py --no_convert   # réutilise les labels déjà convertis

Le modèle est configurable via configs/detection.yaml (sections `detector`
et `training`). Changer de modèle = changer `detector.name` ("yolo" | "rtdetr").
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.datasets.visdrone_yolo import build_dataset_yaml, convert_split  # noqa: E402
from src.detection import build_detector  # noqa: E402
from src.utils.path_utils import load_paths_config  # noqa: E402


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_data(splits_dir: Path, labels_root: Path, class_names: dict,
                 no_convert: bool) -> Path:
    """Convertit VisDrone -> YOLO et écrit le YAML dataset. Retourne le YAML."""
    labels_root.mkdir(parents=True, exist_ok=True)

    if not no_convert:
        print("Conversion des annotations VisDrone -> YOLO ...")
        for split in ["train", "val"]:
            out_dir = labels_root / split / "labels"
            n = convert_split(splits_dir / split, out_dir)
            print(f"  {split}: {n} images converties -> {out_dir}")
        for split in ["train", "val"]:
            src_imgs = splits_dir / split / "images"
            dst_imgs = labels_root / split / "images"
            dst_imgs.mkdir(parents=True, exist_ok=True)
            for img in src_imgs.glob("*.jpg"):
                if not (dst_imgs / img.name).exists():
                    shutil.copy2(img, dst_imgs / img.name)
        print("Images copiées dans data/yolo_labels/")
    else:
        print("Conversion ignorée (--no_convert) — réutilisation des labels existants.")

    dataset_yaml = labels_root / "visdrone.yaml"
    build_dataset_yaml(splits_dir, labels_root, class_names, dataset_yaml)
    print(f"Dataset YAML écrit : {dataset_yaml}")
    return dataset_yaml


def main():
    parser = argparse.ArgumentParser(description="Entraîne le détecteur fixe sur VisDrone")
    parser.add_argument("--config", default=str(ROOT / "configs/detection.yaml"),
                        help="Fichier de config détection")
    parser.add_argument("--model", default=None, help="Modèle de départ (ex: yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None, help="GPU id ou 'cpu'")
    parser.add_argument("--no_convert", action="store_true",
                        help="Ne pas reconvertir les annotations (réutilise les labels existants)")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    det_cfg = cfg.get("detector", {})
    tr_cfg = cfg.get("training", {})

    paths_cfg = load_paths_config(str(ROOT / "configs/paths.yaml"))
    splits_dir = ROOT / paths_cfg["data"]["splits_dir"]
    labels_root = ROOT / "data" / "yolo_labels"

    class_names = tr_cfg.get("class_names", det_cfg.get("class_names", {}))
    dataset_yaml = prepare_data(splits_dir, labels_root, class_names, args.no_convert)

    if args.model:
        det_cfg["weights"] = args.model
    if args.epochs:
        tr_cfg["epochs"] = args.epochs
    if args.batch:
        tr_cfg["batch"] = args.batch
    if args.imgsz:
        tr_cfg["imgsz"] = args.imgsz
    if args.device:
        tr_cfg["device"] = args.device

    detector = build_detector(det_cfg)
    print(f"\n=== Entraînement {detector.name} ({det_cfg.get('weights')}) sur VisDrone ===")

    data_cfg = {"dataset_yaml": dataset_yaml}
    result = detector.train(data_cfg, tr_cfg)

    print("\n=== Entraînement terminé ===")
    print(f"Meilleur modèle : {result.best_weights}")
    print(f"Résultats : {result.save_dir}")


if __name__ == "__main__":
    main()