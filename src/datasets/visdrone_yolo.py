"""Conversion des annotations VisDrone vers le format YOLO."""

from pathlib import Path
from typing import Dict, List, Tuple


def visdrone_to_yolo(ann_path: Path, img_size: Tuple[int, int],
                     class_offset: int = 1) -> List[str]:
    """Convertit une annotation VisDrone en lignes YOLO."""
    w_img, h_img = img_size
    lines = []
    with open(ann_path, "r") as f:
        for row in f.read().strip().splitlines():
            parts = row.split(",")
            if len(parts) < 6:
                continue
            x, y, w, h = (int(float(v)) for v in parts[:4])
            score = int(float(parts[4]))
            cls = int(float(parts[5])) - class_offset
            if score == 0 or cls < 0:
                continue
            cx = min(max((x + w / 2) / w_img, 0.0), 1.0)
            cy = min(max((y + h / 2) / h_img, 0.0), 1.0)
            nw = min(max(w / w_img, 0.0), 1.0)
            nh = min(max(h / h_img, 0.0), 1.0)
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def convert_split(split_dir: Path, out_labels_dir: Path,
                  class_offset: int = 1) -> int:
    """Convertit toutes les annotations d'un split vers le format YOLO."""
    from PIL import Image

    images_dir = split_dir / "images"
    anns_dir = split_dir / "annotations"
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for ann in sorted(anns_dir.glob("*.txt")):
        img_path = images_dir / (ann.stem + ".jpg")
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            img_size = im.size
        lines = visdrone_to_yolo(ann, img_size, class_offset)
        if not lines:
            continue
        (out_labels_dir / (ann.stem + ".txt")).write_text("\n".join(lines) + "\n")
        n += 1
    return n


def build_dataset_yaml(splits_dir: Path, labels_root: Path,
                       class_names: Dict[int, str], out_yaml: Path) -> dict:
    """Écrit le YAML dataset attendu par ultralytics."""
    import yaml

    data = {
        "path": str(labels_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {int(k): v for k, v in class_names.items()},
    }
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return data