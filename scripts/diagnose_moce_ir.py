#!/usr/bin/env python3
"""Diagnostic complet du pipeline MoCE-IR, de l'entrée à la sortie.

Objectif : comprendre pourquoi MoCE-IR produit des restaurations "bizarres"
(teinte crépusculaire, artefacts, etc.) en traçant chaque étape du modèle.

Ce script instrumente le modèle avec des hooks PyTorch pour capturer :
  1. Les activations intermédiaires (patch_embed, encodeur, latent, décodeur,
     refinement, sortie) — stats min/max/moyenne/std + cartes de features.
  2. Le routage des experts (AdapterLayer) — quels experts sont sélectionnés
     et avec quels poids de gate, à chaque étage du décodeur.
  3. Les statistiques entrée/sortie (moyenne, std, min, max, histogramme).
  4. L'analyse spectrale (FFT) entrée/sortie pour détecter des artefacts
     haute-fréquence ou un décalage de contraste.
  5. La comparaison avec l'image claire (référence) si disponible.

Usage:
    python scripts/diagnose_moce_ir.py --image chemin/vers/degradee.jpg \
        --weights outputs/best_moce_ir.pth --output_dir outputs/diagnostics/moce_ir
    python scripts/diagnose_moce_ir.py --image ... --clean chemin/vers/claire.jpg
    python scripts/diagnose_moce_ir.py --image ... --patch_size 128  # régime entraînement
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.baselines.moce_ir import MoceIrBaseline  # noqa: E402
from src.visualization.training_samples import _pad_to_multiple  # noqa: E402


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def tensor_stats(t: torch.Tensor) -> dict:
    """Statistiques d'un tensor : min, max, mean, std, abs-mean."""
    return {
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.mean()),
        "std": float(t.std()),
        "abs_mean": float(t.abs().mean()),
    }


def fmt_stats(d: dict) -> str:
    return (f"min={d['min']:.4f} max={d['max']:.4f} "
            f"mean={d['mean']:.4f} std={d['std']:.4f} |abs|={d['abs_mean']:.4f}")


def save_feature_grid(t: torch.Tensor, path: Path, n: int = 16) -> None:
    """Sauvegarde une grille de n cartes de features (normalisées par canal)."""
    t = t[0]  # [C, H, W]
    c = t.shape[0]
    n = min(n, c)
    # Sélectionne n canaux répartis uniformément.
    idx = np.linspace(0, c - 1, n).astype(int)
    feats = t[idx]
    # Normalisation par canal pour la visualisation.
    feats = feats - feats.amin(dim=(1, 2), keepdim=True)
    feats = feats / (feats.amax(dim=(1, 2), keepdim=True) + 1e-6)
    # Grille carrée.
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    tiles = []
    for i in range(n):
        tiles.append(to_pil_image(feats[i].clamp(0, 1)))
    tw, th = tiles[0].size
    grid = Image.new("RGB", (cols * tw, rows * th), (0, 0, 0))
    for i, tile in enumerate(tiles):
        grid.paste(tile.convert("RGB"), ((i % cols) * tw, (i // cols) * th))
    grid.save(path)


def spectral_stats(t: torch.Tensor) -> dict:
    """Analyse spectrale (FFT) d'un tensor image [1,3,H,W] dans [0,1].

    Retourne la fraction d'énergie dans les basses fréquences et le rapport
    de contraste global (RMS). Utile pour détecter un flou ou un artefact.
    """
    x = t[0].float()  # [3, H, W]
    # Passe en niveaux de gris pour une analyse simple.
    g = x.mean(dim=0)
    g = g - g.mean()
    f = torch.fft.rfft2(g)
    mag = f.abs() ** 2
    total = mag.sum()
    h, w = g.shape
    # Énergie dans un disque central (basses fréquences) — rayon 10% de la taille.
    dev = g.device
    yy, xx = torch.meshgrid(torch.arange(h, device=dev),
                            torch.arange(w // 2 + 1, device=dev), indexing="ij")
    cy, cx = h / 2, 0.0
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    low = (mag * (r < 0.1 * h)).sum()
    return {
        "low_freq_frac": float(low / (total + 1e-8)),
        "rms": float(g.std()),
    }


# ---------------------------------------------------------------------------
# Instrumentation du modèle
# ---------------------------------------------------------------------------
class MoCEIRDiagnostics:
    """Attache des hooks au modèle MoCEIR et collecte les activations + routage."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.activations = {}
        self.routing = {}
        self._hooks = []
        self._register()

    def _register(self):
        m = self.model

        # Patch embed (entrée).
        self._hook(m.patch_embed, "patch_embed")

        # Encodeur : chaque EncoderResidualGroup + Downsample.
        for i, (block, downsample) in enumerate(m.enc):
            self._hook(block, f"enc_{i}_block")
            self._hook(downsample, f"enc_{i}_downsample")

        # Latent.
        self._hook(m.latent, "latent")

        # Décodeur : Upsample, fusion, DecoderResidualGroup (contient les experts).
        for i, (upsample, fusion, block) in enumerate(m.dec):
            self._hook(upsample, f"dec_{i}_upsample")
            self._hook(fusion, f"dec_{i}_fusion")
            self._hook(block, f"dec_{i}_block")
            # Hooks sur les AdapterLayer (routage des experts) dans ce bloc.
            for j, layer in enumerate(block.layers):
                self._hook_routing(layer.adapter, f"dec_{i}_block_{j}_adapter")

        # Refinement + sortie.
        self._hook(m.refinement, "refinement")
        self._hook(m.output, "output")

    def _hook(self, module, name):
        def make_fn(n):
            def fn(_mod, _inp, out):
                self.activations[n] = out.detach()
            return fn
        self._hooks.append(module.register_forward_hook(make_fn(name)))

    def _hook_routing(self, adapter, name):
        """Capture les gates et indices des experts sélectionnés."""
        def make_fn(n):
            def fn(mod, _inp, out):
                # Recalcule le routage pour capturer gates/indices (déterministe en eval).
                x, freq_emb, shared = _inp
                gates, top_k_indices, top_k_values, _ = mod.routing(x, freq_emb)
                self.routing[n] = {
                    "gates": gates.detach(),
                    "top_k_indices": top_k_indices.detach(),
                    "top_k_values": top_k_values.detach(),
                }
            return fn
        self._hooks.append(adapter.register_forward_hook(make_fn(name)))

    def clear(self):
        self.activations.clear()
        self.routing.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------
def build_report(diag: MoCEIRDiagnostics, inp: torch.Tensor, out: torch.Tensor,
                 clean: torch.Tensor | None) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("DIAGNOSTIC MoCE-IR — de l'entrée à la sortie")
    lines.append("=" * 70)

    lines.append("\n[1] STATISTIQUES ENTRÉE / SORTIE")
    lines.append(f"  Entrée  : {fmt_stats(tensor_stats(inp))}")
    lines.append(f"  Sortie  : {fmt_stats(tensor_stats(out))}")
    if clean is not None:
        lines.append(f"  Claire  : {fmt_stats(tensor_stats(clean))}")

    lines.append("\n[2] ANALYSE SPECTRALE (FFT)")
    for name, t in [("Entrée", inp), ("Sortie", out)]:
        s = spectral_stats(t)
        lines.append(f"  {name:6s}: low_freq_frac={s['low_freq_frac']:.4f} rms={s['rms']:.4f}")
    if clean is not None:
        s = spectral_stats(clean)
        lines.append(f"  Claire : low_freq_frac={s['low_freq_frac']:.4f} rms={s['rms']:.4f}")

    lines.append("\n[3] ACTIVATIONS INTERMÉDIAIRES")
    for name, t in diag.activations.items():
        lines.append(f"  {name:24s} {str(tuple(t.shape)):16s} {fmt_stats(tensor_stats(t))}")

    lines.append("\n[4] ROUTAGE DES EXPERTS (gates par étage du décodeur)")
    for name, r in diag.routing.items():
        gates = r["gates"]  # [B, num_experts]
        idx = r["top_k_indices"]  # [B, k]
        vals = r["top_k_values"]  # [B, k]
        g = gates[0].cpu().numpy()
        i = idx[0].cpu().numpy()
        v = vals[0].cpu().numpy()
        sel = ", ".join(f"E{int(ii)}({float(vv):.3f})" for ii, vv in zip(i, v))
        lines.append(f"  {name:28s} gates={np.round(g, 3)} -> {sel}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Diagnostic MoCE-IR entrée->sortie")
    parser.add_argument("--image", type=str, required=True, help="Image dégradée")
    parser.add_argument("--clean", type=str, default=None, help="Image claire (référence)")
    parser.add_argument("--weights", type=str, default="outputs/best_moce_ir.pth")
    parser.add_argument("--patch_size", type=int, default=None,
                        help="Si défini, crop central de cette taille (régime entraînement)")
    parser.add_argument("--output_dir", type=str, default="outputs/diagnostics/moce_ir")
    parser.add_argument("--save_features", action="store_true",
                        help="Sauvegarde les grilles de cartes de features")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Chargement du modèle.
    cfg = load_yaml(str(ROOT / "configs/baselines.yaml"))["baselines"]["all_in_one"]["moce_ir"]
    kwargs = {k: v for k, v in cfg.items() if k not in ["weights_path"]}
    print(f"Chargement du modèle depuis {args.weights}...")
    wrapper = MoceIrBaseline(weights_path=args.weights, **kwargs)
    wrapper.model.to(device)
    wrapper.model.eval()

    # Chargement de l'image.
    img = Image.open(args.image).convert("RGB")
    print(f"Image chargée : {img.size[0]}x{img.size[1]} px")

    if args.patch_size is not None:
        w, h = img.size
        left, top = (w - args.patch_size) // 2, (h - args.patch_size) // 2
        img = img.crop((left, top, left + args.patch_size, top + args.patch_size))
        print(f"Crop central {args.patch_size}x{args.patch_size} appliqué")

    # Padding à un multiple de 16 (prérequis MoCE-IR).
    img_padded = _pad_to_multiple(img, multiple=16)
    inp = to_tensor(img_padded).unsqueeze(0).to(device)

    clean = None
    if args.clean:
        c = Image.open(args.clean).convert("RGB")
        c = _pad_to_multiple(c, multiple=16)
        clean = to_tensor(c).unsqueeze(0).to(device)

    # Instrumentation.
    diag = MoCEIRDiagnostics(wrapper.model)

    # Inférence.
    with torch.no_grad():
        out = wrapper.model(inp)

    # Rapport texte.
    report = build_report(diag, inp, out, clean)
    print(report)

    # Sauvegarde.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.txt").write_text(report + "\n")

    # Images.
    to_pil_image(inp[0].cpu().clamp(0, 1)).save(out_dir / "input.png")
    to_pil_image(out[0].cpu().clamp(0, 1)).save(out_dir / "output.png")
    if clean is not None:
        to_pil_image(clean[0].cpu().clamp(0, 1)).save(out_dir / "clean.png")

    # Cartes de features.
    if args.save_features:
        feat_dir = out_dir / "features"
        feat_dir.mkdir(parents=True, exist_ok=True)
        for name, t in diag.activations.items():
            save_feature_grid(t, feat_dir / f"{name}.png")

    diag.remove()
    print(f"\nRésultats sauvegardés dans {out_dir}/")


if __name__ == "__main__":
    main()