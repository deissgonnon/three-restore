from collections import Counter
from pathlib import Path

DATA = Path("data")
RAW = DATA / "raw" / "visdrone"
LABELS = DATA / "yolo_labels"

CLASS_NAMES = {
    0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
    5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor",
}
PEDESTRIANS = {0, 1}
VEHICLES = {2, 3, 4, 5, 6, 7, 8, 9}


def n_images(folder):
    return sum(p.suffix.lower() in {".jpg", ".jpeg", ".png"} for p in folder.rglob("*"))


def count_yolo(labels_dir):
    c = Counter()
    for txt in labels_dir.glob("*.txt"):
        c.update(int(l.split()[0]) for l in txt.read_text().splitlines() if l.strip())
    return c


def count_raw(ann_dir):
    """Raw VisDrone annotations: category is the 6th field (0 and 11 are ignored)."""
    c = Counter()
    for txt in ann_dir.glob("*.txt"):
        for l in txt.read_text().splitlines():
            p = l.split(",")
            if len(p) >= 6 and 1 <= int(p[5]) <= 10:
                c[int(p[5]) - 1] += 1
    return c


def print_table(headers, rows):
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(str(x).rjust(w) if isinstance(x, int) else str(x).ljust(w)
                         for x, w in zip(r, widths)))


# --- Instances per class ---
counts = {s: count_yolo(LABELS / s / "labels") for s in ["train", "val"]}
counts["global"] = counts["train"] + counts["val"]
cols = ["train", "val", "global"]

print(f"{'id':<3} {'class':<17}" + "".join(f"{c:>10}" for c in cols))
for cid, name in CLASS_NAMES.items():
    print(f"{cid:<3} {name:<17}" + "".join(f"{counts[c][cid]:>10}" for c in cols))
print(f"{'':<3} {'TOTAL':<17}" + "".join(f"{sum(counts[c].values()):>10}" for c in cols))

# --- Split table ---
counts["test"] = count_raw(RAW / "VisDrone2019-DET-test-dev" / "annotations")
paths = {
    "train": RAW / "VisDrone2019-DET-train",
    "val": RAW / "VisDrone2019-DET-val",
    "test": RAW / "VisDrone2019-DET-test-dev",
    "real low-light/night test": DATA / "real_lowlight_test",
}

rows = []
for split, path in paths.items():
    c = counts.get(split)
    stats = (sum(c.values()), sum(c[i] for i in VEHICLES), sum(c[i] for i in PEDESTRIANS)) if c else ("N/A",) * 3
    rows.append([split, n_images(path), *stats, str(path)])

print()
print_table(["split", "images", "boxes", "vehicles", "pedestrians", "path"], rows)

# --- Degradations ---
rows = []
for name, folder in {"fog": "fog", "rain": "rain", "low-light": "lowlight"}.items():
    path = DATA / "synthetic" / folder
    rows.append([name, str(path), "Yes" if n_images(path) else "No"])

print()
print_table(["degradation", "output path", "done"], rows)