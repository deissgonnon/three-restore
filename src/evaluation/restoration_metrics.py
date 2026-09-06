"""Métriques de restauration : PSNR, SSIM, LPIPS (requis) ; NIQE, BRISQUE,
temps d'inférence, paramètres, FLOPs (facultatif).

Doc de suivi section 6.3.
"""


def compute_psnr(restored, target):
    raise NotImplementedError


def compute_ssim(restored, target):
    raise NotImplementedError


def compute_lpips(restored, target):
    raise NotImplementedError
