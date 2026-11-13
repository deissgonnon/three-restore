import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DATA = Path("data")
LABELS = DATA / "yolo_labels"
PEDESTRIANS = {0, 1}  # pedestrian, people (all other classes are drawn as vehicles)

SPLIT = "val"
N = 5
SEED = 42
DEGRADED = {"fog": "fog", "rain": "rain", "low-light": "lowlight"}
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def index(folder):
    """Map file name -> path for all images in a folder (recursive)."""
    return {p.name: p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS}


def draw_boxes(ax, label_path, w, h):
    """Draw YOLO boxes: pedestrians in red, vehicles in lime."""
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        cls, cx, cy, bw, bh = line.split()[:5]
        cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
        color = "red" if int(cls) in PEDESTRIANS else "lime"
        ax.add_patch(Rectangle(((cx - bw / 2) * w, (cy - bh / 2) * h), bw * w, bh * h,
                               fill=False, edgecolor=color, linewidth=1))


clean = index(LABELS / SPLIT / "images")
degraded = {deg: index(DATA / "synthetic" / folder / SPLIT) for deg, folder in DEGRADED.items()}

# Same 5 images (same file names) for the three degradations
common = sorted(set(clean).intersection(*degraded.values()))
names = random.Random(SEED).sample(common, N)

for deg, idx in degraded.items():
    fig, axes = plt.subplots(N, 3, figsize=(15, 3.8 * N))
    for row, name in zip(axes, names):
        img_clean, img_deg = plt.imread(clean[name]), plt.imread(idx[name])
        h, w = img_deg.shape[:2]

        row[0].imshow(img_clean)
        row[0].set_title(f"clean | ID: {Path(name).stem}\n{clean[name]}", fontsize=7)
        row[1].imshow(img_deg)
        row[1].set_title(f"{deg}\n{idx[name]}", fontsize=7)
        row[2].imshow(img_deg)
        draw_boxes(row[2], LABELS / SPLIT / "labels" / f"{Path(name).stem}.txt", w, h)
        row[2].set_title("annotations (vehicles: green, pedestrians: red)", fontsize=7)

        for ax in row:
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(Path("outputs") / f"examples_{deg}.png", dpi=100)
    plt.close(fig)