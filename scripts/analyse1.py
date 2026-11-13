from pathlib import Path

import lpips
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from ultralytics import YOLO

ROOT = Path("/workspace/202427000031/DEP-RNet")
DATA = ROOT / "data"
WEIGHTS = ROOT / "outputs/checkpoints/detector/yolo26l_visdrone/weights/best.pt"
OUT_DIR = ROOT / "outputs/baseline_table"
EVAL_DIR = DATA / "eval_sets"

CLEAN_DIR = DATA / "raw/visdrone/VisDrone2019-DET-test-dev/VisDrone2019-DET-test-dev/images"
ANN_DIR = DATA / "synthetic/fog/test/annotations"   # raw VisDrone annotations
YOLO_LABELS = DATA / "yolo_labels/test/labels"

DEGRADATIONS = {"fog": "fog", "rain": "rain", "low-light": "lowlight"}  # table name -> folder name
METHODS = [  # (table name, slug, images folder pattern relative to DATA)
    ("Degraded input", "input", "synthetic/{}/test/images"),
    ("Baseline 1 (MoCE-IR)", "moce_ir", "restored/moce_ir/{}"),
]
CLASS_NAMES = {
    0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
    5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor",
}
PEDESTRIANS = {0, 1}
VEHICLES = set(range(2, 10))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def method_dir(pattern, folder):
    return DATA / pattern.format(folder)


def cached(path, fn):
    """Load a CSV if it exists, otherwise compute it and save it."""
    if path.exists():
        return pd.read_csv(path)
    df = fn()
    df.to_csv(path, index=False)
    return df


# --- 1. Labels: raw VisDrone -> YOLO ---
def convert_labels():
    """class = category - 1, normalized cx cy w h; ignored regions (0) and others (11) are dropped."""
    YOLO_LABELS.mkdir(parents=True, exist_ok=True)
    for ann in ANN_DIR.glob("*.txt"):
        w, h = Image.open(CLEAN_DIR / f"{ann.stem}.jpg").size
        lines = []
        for line in ann.read_text().splitlines():
            p = line.split(",")
            if len(p) < 6 or not 1 <= int(p[5]) <= 10:
                continue
            x, y, bw, bh = map(float, p[:4])
            lines.append(f"{int(p[5]) - 1} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
        (YOLO_LABELS / ann.name).write_text("\n".join(lines))


def common_names(folder):
    """Images available for the clean set and every method (same images for a fair comparison)."""
    dirs = [CLEAN_DIR] + [method_dir(p, folder) for _, _, p in METHODS]
    return sorted(set.intersection(*[{f.name for f in d.glob("*.jpg")} for d in dirs]))


# --- 2. PSNR / SSIM / LPIPS against the clean images ---
def quality_metrics(names):
    lp = lpips.LPIPS(net="alex").to(DEVICE)  # use the same backbone as your previous LPIPS values

    def tensor(a):
        return torch.from_numpy(a).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(DEVICE)

    rows = []
    for deg, folder in DEGRADATIONS.items():
        for method, _, pattern in METHODS:
            img_dir = method_dir(pattern, folder)
            n_cropped = 0
            for n in names[deg]:
                gt = np.array(Image.open(CLEAN_DIR / n).convert("RGB"))
                img = np.array(Image.open(img_dir / n).convert("RGB"))
                if gt.shape != img.shape:  # e.g. output cropped to a multiple of 16
                    h, w = min(gt.shape[0], img.shape[0]), min(gt.shape[1], img.shape[1])
                    gt, img = gt[:h, :w], img[:h, :w]
                    n_cropped += 1
                with torch.no_grad():
                    lp_val = lp(tensor(gt), tensor(img)).item()
                rows.append({
                    "method": method, "degradation": deg, "image": n,
                    "PSNR": peak_signal_noise_ratio(gt, img, data_range=255),
                    "SSIM": structural_similarity(gt, img, channel_axis=2, data_range=255),
                    "LPIPS": lp_val,
                })
            print(f"{method} | {deg}: {n_cropped} images with a size mismatch (cropped to the common top-left area)")
    return pd.DataFrame(rows)


# --- 3. Detection metrics with Ultralytics ---
def build_eval_set(slug, deg, img_dir, names):
    """Ultralytics needs images/ and labels/ side by side.
    Images are symlinked; labels are rewritten for each image size, because restored images
    are cropped to a multiple of 16 (bottom/right), which shifts the normalized coordinates.
    """
    root = EVAL_DIR / f"{slug}_{deg}"
    (root / "images").mkdir(parents=True, exist_ok=True)
    labels = root / "labels"
    if labels.is_symlink():  # leftover from a previous version
        labels.unlink()
    labels.mkdir(exist_ok=True)

    for n in names:
        link = root / "images" / n
        if not link.exists():
            link.symlink_to((img_dir / n).resolve())

        w0, h0 = Image.open(CLEAN_DIR / n).size  # original size (labels are normalized by it)
        w1, h1 = Image.open(img_dir / n).size    # size of the evaluated image
        lines = []
        for line in (YOLO_LABELS / f"{Path(n).stem}.txt").read_text().splitlines():
            if not line.strip():
                continue
            c, cx, cy, bw, bh = line.split()[:5]
            cx, cy, bw, bh = float(cx) * w0, float(cy) * h0, float(bw) * w0, float(bh) * h0  # pixels
            x1, x2 = max(cx - bw / 2, 0), min(cx + bw / 2, w1)
            y1, y2 = max(cy - bh / 2, 0), min(cy + bh / 2, h1)
            if x2 > x1 and y2 > y1:
                lines.append(f"{c} {(x1 + x2) / 2 / w1:.6f} {(y1 + y2) / 2 / h1:.6f} "
                             f"{(x2 - x1) / w1:.6f} {(y2 - y1) / h1:.6f}")
        (labels / f"{Path(n).stem}.txt").write_text("\n".join(lines))

    yaml_path = root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "path": str(root.resolve()), "train": "images", "val": "images", "names": CLASS_NAMES,
    }))
    return yaml_path


def detection_metrics(names):
    model = YOLO(WEIGHTS)
    rows = []
    for deg, folder in DEGRADATIONS.items():
        for method, slug, pattern in METHODS:
            data = build_eval_set(slug, deg, method_dir(pattern, folder), names[deg])
            m = model.val(data=str(data), plots=False, verbose=False,
                          project=str(OUT_DIR / "val_runs"), name=f"{slug}_{deg}", exist_ok=True)
            ap50 = {int(k): v for k, v in zip(m.box.ap_class_index, m.box.ap50)}
            rows.append({
                "method": method, "degradation": deg,
                "mAP50": m.box.map50, "mAP50:95": m.box.map,
                "AP-vehicle": np.mean([ap50[i] for i in VEHICLES if i in ap50]),
                "AP-pedestrian": np.mean([ap50[i] for i in PEDESTRIANS if i in ap50]),
            })
    return pd.DataFrame(rows)


# --- Run ---
OUT_DIR.mkdir(parents=True, exist_ok=True)
if not any(YOLO_LABELS.glob("*.txt")):
    convert_labels()

names = {deg: common_names(folder) for deg, folder in DEGRADATIONS.items()}
print({deg: len(n) for deg, n in names.items()}, "images per degradation")

quality = cached(OUT_DIR / "quality_per_image.csv", lambda: quality_metrics(names))
detection = cached(OUT_DIR / "detection.csv", lambda: detection_metrics(names))

keys = ["method", "degradation"]
table = (quality.groupby(keys)[["PSNR", "SSIM", "LPIPS"]].mean()
         .join(detection.set_index(keys))
         .loc[[(m[0], d) for d in DEGRADATIONS for m in METHODS]]
         .reset_index().round(4))

print(table.to_string(index=False))
table.to_csv(OUT_DIR / "baseline_table.csv", index=False)