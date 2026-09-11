import os

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

SPLITS = ["train", "val", "test"]
JOUR_BASE = "data/splits/{split}/images"
NUIT_BASE = "data/real_lowlight_test/{split}/images"


def count_images(folder):
    if not os.path.isdir(folder):
        return 0
    return sum(1 for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS))


stats = []
for split in SPLITS:
    n_jour = count_images(JOUR_BASE.format(split=split))
    n_nuit = count_images(NUIT_BASE.format(split=split))
    stats.append((split, n_jour, n_nuit))

total_jour = sum(s[1] for s in stats)
total_nuit = sum(s[2] for s in stats)

print(f"{'split':<8}{'jour':>8}{'nuit':>8}{'total':>8}{'% nuit':>10}")
print("-" * 42)
for split, n_jour, n_nuit in stats:
    total = n_jour + n_nuit
    pct = (n_nuit / total * 100) if total else 0
    print(f"{split:<8}{n_jour:>8}{n_nuit:>8}{total:>8}{pct:>9.1f}%")

print("-" * 42)
total = total_jour + total_nuit
pct = (total_nuit / total * 100) if total else 0
print(f"{'total':<8}{total_jour:>8}{total_nuit:>8}{total:>8}{pct:>9.1f}%")