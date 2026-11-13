#!/usr/bin/env python3
"""Évalue une baseline de restauration sur les données de test.

Métriques requises : PSNR, SSIM, LPIPS.
Métriques optionnelles : NIQE, BRISQUE, temps d'inférence, paramètres, FLOPs.

Usage:
    python scripts/evaluate_baseline.py --baseline moce_ir
    python scripts/evaluate_baseline.py --baseline moce_ir --weights outputs/best_moce_ir.pth
    python scripts/evaluate_baseline.py --baseline moce_ir --degradations fog rain
    python scripts/evaluate_baseline.py --baseline moce_ir --limit 50       # échantillon
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.baselines.moce_ir import MoceIrBaseline  # noqa: E402
from src.evaluation import restoration_metrics as M  # noqa: E402
from src.utils.path_utils import load_paths_config  # noqa: E402


def load_baseline(name: str, weights: str, device: str):
    """Charge la baseline demandée et retourne (wrapper, model)."""
    if name == "moce_ir":
        import yaml

        with open(ROOT / "configs/baselines.yaml") as f:
            cfg = yaml.safe_load(f)["baselines"]["all_in_one"]["moce_ir"]
        kwargs = {k: v for k, v in cfg.items() if k not in ["weights_path"]}
        wrapper = MoceIrBaseline(weights_path=weights, **kwargs)
        return wrapper, wrapper.model
    raise ValueError(f"Baseline non supportée: {name}")


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _to_model_input(img: np.ndarray, device: str) -> torch.Tensor:
    """Convertit une image HWC en tensor [1,3,H,W] avec H,W divisibles par 16."""
    h, w = img.shape[:2]
    h = h - (h % 16)
    w = w - (w % 16)
    img = img[:h, :w]
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    return t


def evaluate_split(model, clean_dir: Path, deg_dir: Path, device: str,
                   with_reference: bool, limit: int):
    """Évalue le modèle sur toutes les paires (dégradée, claire) d'un split."""
    clean_paths = sorted(clean_dir.glob("*.jpg")) + sorted(clean_dir.glob("*.png"))
    if limit:
        clean_paths = clean_paths[:limit]

    psnr, ssim, lpips = [], [], []
    times = []

    model.eval()
    with torch.no_grad():
        for clean_path in tqdm(clean_paths, desc=f"Évaluation {deg_dir.parent.name}"):
            deg_path = deg_dir / clean_path.name
            if not deg_path.exists():
                continue

            clean = load_image(clean_path)
            deg = load_image(deg_path)

            deg_t = _to_model_input(deg, device)
            h, w = deg_t.shape[-2:]

            t0 = time.perf_counter()
            res_t = model(deg_t)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

            res = res_t[0].permute(1, 2, 0).cpu().numpy()
            res = np.clip(res, 0, 1)

            if with_reference:
                clean_crop = clean[:h, :w]
                psnr.append(M.compute_psnr(res, clean_crop))
                ssim.append(M.compute_ssim(res, clean_crop))
                lpips.append(M.compute_lpips(res, clean_crop))

    stats = {}
    if with_reference:
        stats.update({
            "PSNR": float(np.mean(psnr)),
            "SSIM": float(np.mean(ssim)),
            "LPIPS": float(np.mean(lpips)),
        })
    stats["inference_time_s"] = float(np.mean(times))
    return stats


def main():
    parser = argparse.ArgumentParser(description="Évalue une baseline de restauration sur test")
    parser.add_argument("--baseline", default="moce_ir")
    parser.add_argument("--weights", default=None, help="Chemin des poids (défaut: outputs/best_moce_ir.pth)")
    parser.add_argument("--degradations", nargs="+", default=["fog", "rain", "lowlight"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="Limiter à N images par dégradation")
    args = parser.parse_args()

    paths = load_paths_config(str(ROOT / "configs/paths.yaml"))
    splits_dir = ROOT / paths["data"]["splits_dir"]
    weights = args.weights or str(ROOT / "outputs/best_moce_ir.pth")

    wrapper, model = load_baseline(args.baseline, weights, args.device)
    model = model.to(args.device)
    print(f"Baseline: {args.baseline} | device: {args.device} | poids: {weights}")

    # Métriques modèle (params, FLOPs, temps) — une seule fois
    n_params = M.count_parameters(model)
    flops = M.count_flops(model, input_size=(256, 256), device=args.device)
    inf_time = M.measure_inference_time(model, input_size=(256, 256), device=args.device)
    print(f"\n=== Métriques modèle ===")
    print(f"Paramètres : {n_params:,}")
    print(f"FLOPs (256x256) : {flops/1e9:.2f} G")
    print(f"Temps inférence (256x256) : {inf_time*1000:.1f} ms")

    clean_test = splits_dir / "test" / "images"
    out_file = ROOT / "outputs" / "eval_moce_ir_test.txt"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"Baseline: {args.baseline} | device: {args.device} | poids: {weights}")
    lines.append(f"\n=== Métriques modèle ===")
    lines.append(f"Paramètres : {n_params:,}")
    lines.append(f"FLOPs (256x256) : {flops/1e9:.2f} G")
    lines.append(f"Temps inférence (256x256) : {inf_time*1000:.1f} ms")
    lines.append(f"\n=== Évaluation sur test ({clean_test}) ===")

    print(f"\n=== Évaluation sur test ({clean_test}) ===")
    for deg in args.degradations:
        deg_dir = ROOT / paths["data"][f"synthetic_{deg}"] / "test" / "images"
        if not deg_dir.exists():
            print(f"  [skip] {deg}: dossier absent {deg_dir}")
            continue
        print(f"\n--- {deg} ---")
        lines.append(f"\n--- {deg} ---")
        stats = evaluate_split(model, clean_test, deg_dir, args.device,
                               with_reference=True, limit=args.limit)
        for k, v in stats.items():
            txt = f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}"
            print(txt)
            lines.append(txt)

    out_file.write_text("\n".join(lines) + "\n")
    print(f"\nRésultats écrits dans {out_file}")


if __name__ == "__main__":
    main()
