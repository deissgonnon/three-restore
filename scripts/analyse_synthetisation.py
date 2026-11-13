#!/usr/bin/env python3
"""Analyse de la synthèse des dégradations (brouillard, pluie, faible lumière).

À partir des images claires de `data/splits/{split}/images` et des images
dégradées de `data/synthetic/{fog,rain,lowlight}/{split}/images` (appariées
par nom de fichier), ce script génère :

  1. une planche d'exemples : clair + brouillard + pluie + faible lumière
     pour quelques images échantillonnées ;
  2. une figure d'analyse globale :
       (a) distribution de la luminance moyenne (BT.601) par condition ;
       (b) distribution du SSIM dégradé vs clair ;
       (c) décalage moyen des canaux (ΔR, ΔV, ΔB) vs clair ;
       (d) couverture : nombre d'images appariées par split ;
  3. un tableau récapitulatif dans la console et un CSV des statistiques
     par image.

Usage :
  python scripts/analyse_synthetisation.py
  python scripts/analyse_synthetisation.py --split val --n-samples 6
  python scripts/analyse_synthetisation.py --synthetic-root "data/synthetic v2"
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import gaussian_kde
from skimage.metrics import structural_similarity
from tqdm import tqdm

DEGRADATIONS = ("fog", "rain", "lowlight")
CONDITIONS = ("clear",) + DEGRADATIONS
LABELS = {
    "clear": "Clair (GT)",
    "fog": "Brouillard",
    "rain": "Pluie",
    "lowlight": "Faible lumière",
}
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Palette catégorielle validée (mode clair) : ordre fixe, jamais cyclé.
COLORS = {
    "clear": "#2a78d6",     # bleu   — slot 1
    "fog": "#eb6834",       # orange — slot 2
    "rain": "#1baf7a",      # aqua   — slot 3
    "lowlight": "#eda100",  # jaune  — slot 4
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse des images synthétiques (fog/rain/lowlight).")
    parser.add_argument("--synthetic-root", default="data/synthetic",
                        help="Racine des images synthétiques (défaut : data/synthetic).")
    parser.add_argument("--clear-root", default="data/splits",
                        help="Racine des images claires (défaut : data/splits).")
    parser.add_argument("--split", default="train", choices=SPLITS,
                        help="Split utilisé pour les exemples et les statistiques (défaut : train).")
    parser.add_argument("--n-samples", type=int, default=4,
                        help="Nombre de lignes d'exemples (défaut : 4).")
    parser.add_argument("--n-stats", type=int, default=300,
                        help="Nombre d'images pour les statistiques globales (défaut : 300).")
    parser.add_argument("--max-size", type=int, default=1024,
                        help="Côté maximal des images pour les statistiques (défaut : 1024 px).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Graine pour l'échantillonnage (défaut : 42).")
    parser.add_argument("--figures", choices=("samples", "stats", "both"), default="both",
                        help="Figures à générer (défaut : both).")
    parser.add_argument("--output-dir", default="outputs/synthetisation_analysis",
                        help="Répertoire de sortie (défaut : outputs/synthetisation_analysis).")
    return parser.parse_args()


def list_image_names(directory: Path) -> set[str]:
    """Noms relatifs des images d'un dossier (sans les artefacts de notebook)."""
    if not directory.is_dir():
        return set()
    return {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
        and ".ipynb_checkpoints" not in path.parts
    }


def matched_names(clear_root: Path, synthetic_root: Path, split: str) -> list[str]:
    """Noms des images claires présentes pour toutes les dégradations du split."""
    clear_names = list_image_names(clear_root / split / "images")
    sets = [clear_names]
    for degradation in DEGRADATIONS:
        folder = synthetic_root / degradation / split / "images"
        if folder.is_dir():
            sets.append(list_image_names(folder))
    return sorted(set.intersection(*sets))


def coverage_per_split(clear_root: Path, synthetic_root: Path) -> dict[str, int]:
    """Nombre d'images appariées clair + 3 dégradations, par split."""
    return {split: len(matched_names(clear_root, synthetic_root, split)) for split in SPLITS}


def load_rgb(path: Path, max_size: int | None = None) -> np.ndarray:
    """Charge une image RGB en float32 ; redimensionne le côté le plus long si demandé."""
    image = Image.open(path).convert("RGB")
    if max_size is not None:
        image.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def luminance(image: np.ndarray) -> float:
    """Luminance moyenne BT.601 (0-255)."""
    return float(np.mean(0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]))


def pair_stats(clear: np.ndarray, degraded: np.ndarray) -> dict[str, float]:
    """Statistiques de similarité entre une image dégradée et sa référence claire."""
    delta = degraded - clear
    mse = float(np.mean(delta ** 2))
    ssim = float(structural_similarity(clear, degraded, data_range=255.0, channel_axis=-1))
    return {
        "lum": luminance(degraded),
        "ssim": ssim,
        "psnr": float("inf") if mse == 0 else float(10.0 * np.log10(255.0 ** 2 / mse)),
        "mae": float(np.mean(np.abs(delta))),
        "dr": float(np.mean(delta[..., 0])),
        "dg": float(np.mean(delta[..., 1])),
        "db": float(np.mean(delta[..., 2])),
    }


def setup_style() -> None:
    """Style commun des figures (surface claire, traits fins, grille discrète)."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "savefig.dpi": 150,
    })


def _spines(ax) -> None:
    """Garde uniquement les axes gauche et bas."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_tag(ax, tag: str) -> None:
    ax.text(0.01, 1.03, tag, transform=ax.transAxes, fontsize=12, fontweight="bold",
            color=INK_MUTED, va="bottom", ha="left")


def _kde(values: np.ndarray, n_points: int = 300):
    """Estimation de densité sur l'étendue des valeurs (ligne lissée)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or float(np.std(values)) == 0.0:
        return None, None
    xs = np.linspace(float(np.min(values)), float(np.max(values)), n_points)
    return xs, gaussian_kde(values)(xs)


def _kde_line(ax, values: np.ndarray, color: str, label: str, ink: str) -> None:
    """Trace une KDE avec libellé direct au sommet de la densité."""
    xs, density = _kde(values)
    if xs is None:
        return
    ax.plot(xs, density, color=color, lw=2, label=label)
    peak = int(np.argmax(density))
    ax.text(xs[peak], density[peak], f"  {label}", color=ink, fontsize=9,
            va="center", ha="left", fontweight="bold")


def build_samples_figure(names: list[str], clear_root: Path, synthetic_root: Path,
                         split: str) -> plt.Figure:
    """Planche d'exemples : une ligne par image, une colonne par condition."""
    fig, axes = plt.subplots(len(names), len(CONDITIONS),
                             figsize=(12.5, 3.15 * len(names)))
    axes = np.atleast_2d(axes)

    for row, name in enumerate(names):
        for col, condition in enumerate(CONDITIONS):
            ax = axes[row, col]
            folder = clear_root / split / "images" if condition == "clear" \
                else synthetic_root / condition / split / "images"
            ax.imshow(np.asarray(Image.open(folder / name).convert("RGB")))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col == 0:
                ax.set_ylabel(Path(name).stem, rotation=0, ha="right", va="center",
                              fontsize=9, color=INK_SECONDARY, labelpad=8)
            if row == 0:
                ax.text(0.02, 1.06, "■", color=COLORS[condition], fontsize=13,
                        transform=ax.transAxes, va="bottom")
                ax.text(0.07, 1.06, LABELS[condition], color=INK, fontsize=12,
                        fontweight="bold", transform=ax.transAxes, va="bottom")

    fig.subplots_adjust(left=0.16, wspace=0.02, hspace=0.06)
    fig.suptitle(f"Synthèse des dégradations — split {split} ({Path(synthetic_root).name})",
                 color=INK, fontsize=13, fontweight="bold", y=0.995)
    return fig


def build_global_figure(stats: dict, coverage: dict[str, int], synthetic_root: Path,
                        split: str, n_stats: int) -> plt.Figure:
    """Figure d'analyse globale : luminance, SSIM, canaux, couverture."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    (ax_lum, ax_ssim), (ax_chan, ax_cov) = axes

    # (a) Distribution de la luminance moyenne par condition.
    for condition in CONDITIONS:
        _kde_line(ax_lum, stats["lum"][condition], COLORS[condition],
                  LABELS[condition], INK_SECONDARY)
    _panel_tag(ax_lum, "(a)")
    ax_lum.set_title("Luminance moyenne (BT.601)")
    ax_lum.set_xlabel("Luminance (0-255)")
    ax_lum.set_ylabel("Densité")
    ax_lum.set_xlim(0, 255)
    ax_lum.set_ylim(bottom=0)
    ax_lum.legend(fontsize=9, loc="upper right", handlelength=1.4, handletextpad=0.5)
    _spines(ax_lum)

    # (b) Distribution du SSIM dégradé vs clair.
    for degradation in DEGRADATIONS:
        _kde_line(ax_ssim, stats["ssim"][degradation], COLORS[degradation],
                  LABELS[degradation], INK_SECONDARY)
    _panel_tag(ax_ssim, "(b)")
    ax_ssim.set_title("SSIM dégradé vs clair")
    ax_ssim.set_xlabel("SSIM")
    ax_ssim.set_ylabel("Densité")
    ax_ssim.set_xlim(0.0, 1.0)
    ax_ssim.set_ylim(bottom=0)
    ax_ssim.legend(fontsize=9, loc="upper left", handlelength=1.4, handletextpad=0.5)
    _spines(ax_ssim)

    # (c) Décalage moyen des canaux vs clair, avec écart-type.
    channels = [("ΔR", "dr"), ("ΔV", "dg"), ("ΔB", "db")]
    x = np.arange(len(channels))
    width = 0.26
    for i, degradation in enumerate(DEGRADATIONS):
        means = [float(np.mean(stats[key][degradation])) for _, key in channels]
        stds = [float(np.std(stats[key][degradation])) for _, key in channels]
        ax_chan.bar(x + (i - 1) * width, means, width, yerr=stds,
                    color=COLORS[degradation], label=LABELS[degradation],
                    edgecolor=SURFACE, linewidth=1.5, capsize=3,
                    error_kw={"ecolor": INK_MUTED, "elinewidth": 0.8, "capthick": 0.8})
    _panel_tag(ax_chan, "(c)")
    ax_chan.set_title("Décalage moyen des canaux vs clair")
    ax_chan.set_xticks(x, [name for name, _ in channels])
    ax_chan.set_ylabel("Δ du canal (0-255)")
    ax_chan.axhline(0.0, color=BASELINE, lw=0.8)
    ax_chan.legend(fontsize=9, loc="upper right", handlelength=1.4, handletextpad=0.5)
    ax_chan.grid(axis="x", visible=False)
    _spines(ax_chan)

    # (d) Couverture par split : images appariées pour les 3 dégradations.
    bars = ax_cov.bar(range(len(SPLITS)), [coverage[s] for s in SPLITS],
                      color=COLORS["clear"], edgecolor=SURFACE, linewidth=1.5)
    for bar, value in zip(bars, [coverage[s] for s in SPLITS]):
        ax_cov.text(bar.get_x() + bar.get_width() / 2, value + 12, f"{value:,}",
                    ha="center", va="bottom", fontsize=10, color=INK_SECONDARY,
                    fontweight="bold")
    _panel_tag(ax_cov, "(d)")
    ax_cov.set_title(f"Couverture : images claires présentes pour les 3 dégradations"
                     f"\n({Path(synthetic_root).name})")
    ax_cov.set_xticks(range(len(SPLITS)), SPLITS)
    ax_cov.set_ylabel("Nombre d'images")
    ax_cov.set_ylim(top=max(coverage.values()) * 1.15)
    ax_cov.grid(axis="x", visible=False)
    _spines(ax_cov)

    fig.suptitle(f"Analyse globale de la synthèse — split {split}, {n_stats} images "
                 f"({Path(synthetic_root).name})",
                 color=INK, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def print_summary(stats: dict, n_stats: int) -> None:
    """Tableau récapitulatif console (luminance, SSIM, PSNR, MAE)."""
    header = (f"{'condition':<16}{'luminance':>10}{'Δlum':>9}{'SSIM':>8}"
              f"{'PSNR (dB)':>10}{'MAE':>9}")
    print(header)
    print("-" * len(header))
    lum_clear = float(np.mean(stats["lum"]["clear"]))
    for condition in CONDITIONS:
        lum = float(np.mean(stats["lum"][condition]))
        row = f"{LABELS[condition]:<16}{lum:>10.1f}{lum - lum_clear:>+9.1f}"
        if condition != "clear":
            row += (f"{np.mean(stats['ssim'][condition]):>8.3f}"
                    f"{np.mean(stats['psnr'][condition]):>10.2f}"
                    f"{np.mean(stats['mae'][condition]):>9.2f}")
        print(row)
    print(f"\nn = {n_stats} images appariées.")


def save_csv(stats: dict, names: list[str], split: str, path: Path) -> None:
    """CSV des statistiques par image (une ligne par image)."""
    columns = ["split", "fichier"]
    columns += [f"lum_{c}" for c in CONDITIONS]
    for degradation in DEGRADATIONS:
        columns += [f"ssim_{degradation}", f"psnr_{degradation}", f"mae_{degradation}",
                    f"dr_{degradation}", f"dg_{degradation}", f"db_{degradation}"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for i, name in enumerate(names):
            row = {"split": split, "fichier": name}
            for condition in CONDITIONS:
                row[f"lum_{condition}"] = f"{stats['lum'][condition][i]:.2f}"
            for degradation in DEGRADATIONS:
                for key in ("ssim", "psnr", "mae", "dr", "dg", "db"):
                    value = stats[key][degradation][i]
                    row[f"{key}_{degradation}"] = "inf" if value == float("inf") else f"{value:.4f}"
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    clear_root = Path(args.clear_root)
    synthetic_root = Path(args.synthetic_root)
    if not synthetic_root.is_dir():
        raise FileNotFoundError(f"Racine synthétique absente : {synthetic_root}")
    if not (clear_root / args.split / "images").is_dir():
        raise FileNotFoundError(
            f"Split de jour absent : {clear_root / args.split / 'images'}. "
            "Lancer d'abord python -m src.preprocessing.day_night_classifier."
        )

    names = matched_names(clear_root, synthetic_root, args.split)
    print(f"Split {args.split} : {len(names)} images appariées clair + 3 dégradations "
          f"dans {synthetic_root}.")
    coverage = coverage_per_split(clear_root, synthetic_root)

    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = synthetic_root.name.replace(" ", "-")

    if args.figures in ("samples", "both"):
        sample_names = list(rng.choice(names, size=min(args.n_samples, len(names)),
                                       replace=False))
        fig = build_samples_figure(sample_names, clear_root, synthetic_root, args.split)
        out = output_dir / f"exemples_{tag}_{args.split}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"Planche d'exemples : {out}")

    if args.figures in ("stats", "both"):
        n_stats = min(args.n_stats, len(names))
        stats_names = list(rng.choice(names, size=n_stats, replace=False))
        stats = {key: {condition: [] for condition in CONDITIONS}
                 for key in ("lum",)}
        for key in ("ssim", "psnr", "mae", "dr", "dg", "db"):
            stats[key] = {degradation: [] for degradation in DEGRADATIONS}

        clear_folder = clear_root / args.split / "images"
        for name in tqdm(stats_names, desc=f"Statistiques {args.split}"):
            clear = load_rgb(clear_folder / name, max_size=args.max_size)
            stats["lum"]["clear"].append(luminance(clear))
            for degradation in DEGRADATIONS:
                degraded = load_rgb(synthetic_root / degradation / args.split / "images" / name,
                                    max_size=args.max_size)
                for key, value in pair_stats(clear, degraded).items():
                    stats[key][degradation].append(value)

        csv_path = output_dir / f"stats_{tag}_{args.split}.csv"
        save_csv(stats, stats_names, args.split, csv_path)
        print(f"Statistiques par image : {csv_path}")

        fig = build_global_figure(stats, coverage, synthetic_root, args.split, n_stats)
        out = output_dir / f"analyse_globale_{tag}_{args.split}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"Analyse globale : {out}\n")

        print_summary(stats, n_stats)


if __name__ == "__main__":
    main()
