"""Métriques de restauration : PSNR, SSIM, LPIPS (requis) ; NIQE, BRISQUE,
temps d'inférence, paramètres, FLOPs (facultatif).

Doc de suivi section 6.3.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def _to_tensor(image) -> torch.Tensor:
    """Convertit une image (PIL, numpy HWC uint8/float, ou tensor) en tensor
    float32 [1, 3, H, W] dans [0, 1]."""
    if isinstance(image, torch.Tensor):
        t = image.float()
        if t.dim() == 3:
            t = t.unsqueeze(0)
        if t.max() > 1.0:
            t = t / 255.0
        return t
    if isinstance(image, np.ndarray):
        arr = image
    else:  # PIL
        arr = np.array(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return t


def compute_psnr(restored, target, data_range: float = 1.0) -> float:
    """PSNR entre deux images (mêmes dimensions)."""
    r = _to_tensor(restored)
    t = _to_tensor(target)
    mse = torch.mean((r - t) ** 2).item()
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def compute_ssim(restored, target, data_range: float = 1.0) -> float:
    """SSIM (skimage) entre deux images."""
    from skimage.metrics import structural_similarity

    r = _to_tensor(restored)[0].permute(1, 2, 0).numpy()
    t = _to_tensor(target)[0].permute(1, 2, 0).numpy()
    return float(structural_similarity(r, t, data_range=data_range, channel_axis=-1))


_lpips_model = None


def _get_lpips():
    global _lpips_model
    if _lpips_model is None:
        import lpips

        _lpips_model = lpips.LPIPS(net="alex").eval()
    return _lpips_model


def compute_lpips(restored, target) -> float:
    """LPIPS (AlexNet) entre deux images. Plus bas = mieux."""
    model = _get_lpips()
    r = _to_tensor(restored)
    t = _to_tensor(target)
    with torch.no_grad():
        return float(model(r, t).item())


def compute_niqe(image) -> float:
    """NIQE (sans référence) — nécessite piq."""
    import piq

    t = _to_tensor(image)
    if t.shape[-2] < 96 or t.shape[-1] < 96:
        raise ValueError("NIQE nécessite une image >= 96x96")
    with torch.no_grad():
        return float(piq.niqe(t, data_range=1.0).item())


def compute_brisque(image) -> float:
    """BRISQUE (sans référence) — nécessite piq."""
    import piq

    t = _to_tensor(image)
    with torch.no_grad():
        return float(piq.brisque(t, data_range=1.0).item())


def count_parameters(model: nn.Module) -> int:
    """Nombre de paramètres du modèle."""
    return sum(p.numel() for p in model.parameters())


def count_flops(model: nn.Module, input_size: Tuple[int, int] = (256, 256),
                device: str = "cpu") -> float:
    """FLOPs (MACs) du modèle sur une entrée input_size (H, W)."""
    from thop import profile

    model = model.to(device).eval()
    dummy = torch.randn(1, 3, *input_size).to(device)
    with torch.no_grad():
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
    return float(macs)


def measure_inference_time(model: nn.Module, input_size: Tuple[int, int] = (256, 256),
                           device: str = "cpu", num_runs: int = 20) -> float:
    """Temps d'inférence moyen (s) sur input_size, en excluant le warmup."""
    model = model.to(device).eval()
    dummy = torch.randn(1, 3, *input_size).to(device)
    with torch.no_grad():
        for _ in range(3):  # warmup
            model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        import time

        times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return float(np.mean(times))
