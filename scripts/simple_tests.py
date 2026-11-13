from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

ROOT = Path("/workspace/202427000031/DEP-RNet")
DATA = ROOT / "data"
WEIGHTS = ROOT / "outputs/checkpoints/detector/yolo26l_visdrone/weights/best.pt"
OUT_DIR = ROOT / "outputs/baseline_table"
EVAL_DIR = DATA / "eval_sets"

CLEAN_DIR = DATA / "splits/test/images"
YOLO_LABELS = DATA / "yolo_labels/test/labels"  # created by the baseline table script

DEGRADATIONS = {"fog": "fog", "rain": "rain", "low-light": "lowlight"}  # table name -> folder name
CLASS_NAMES = {
    0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
    5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor",
}
PEDESTRIANS = {0, 1}
VEHICLES = set(range(2, 10))


def build_eval_set(key, names):
    """Ultralytics needs images/ and labels/ side by side: symlink clean images and labels."""
    root = EVAL_DIR / f"clean_{key}"
    (root / "images").mkdir(parents=True, exist_ok=True)
    for n in names:
        link = root / "images" / n
        if not link.exists():
            link.symlink_to((CLEAN_DIR / n).resolve())
    if not (root / "labels").exists():
        (root / "labels").symlink_to(YOLO_LABELS.resolve())
    yaml_path = root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "path": str(root.resolve()), "train": "images", "val": "images", "names": CLASS_NAMES,
    }))
    return yaml_path


# Image sets: all clean test images, plus the same subsets used for each degradation in the table
clean = sorted(f.name for f in CLEAN_DIR.glob("*.jpg"))
subsets = {"all": clean}
for deg, folder in DEGRADATIONS.items():
    restored = {f.name for f in (DATA / "restored/moce_ir" / folder).glob("*.jpg")}
    subsets[deg] = [n for n in clean if n in restored]

model = YOLO(WEIGHTS)
rows = []
for key, names in subsets.items():
    data = build_eval_set(key, names)
    m = model.val(data=str(data), plots=False, verbose=False,
                  project=str(OUT_DIR / "val_runs"), name=f"clean_{key}", exist_ok=True)
    ap50 = {int(k): v for k, v in zip(m.box.ap_class_index, m.box.ap50)}
    rows.append({
        "set": key, "images": len(names),
        "mAP50": m.box.map50, "mAP50:95": m.box.map,
        "AP-vehicle": np.mean([ap50[i] for i in VEHICLES if i in ap50]),
        "AP-pedestrian": np.mean([ap50[i] for i in PEDESTRIANS if i in ap50]),
    })

table = pd.DataFrame(rows).round(4)
print(table.to_string(index=False))
table.to_csv(OUT_DIR / "clean_detection.csv", index=False)