"""Tests pour src/evaluation/restoration_metrics.py"""

import pytest
from src.evaluation.restoration_metrics import compute_psnr, compute_ssim, compute_lpips


def test_psnr_not_implemented():
    with pytest.raises(NotImplementedError):
        compute_psnr(None, None)


def test_ssim_not_implemented():
    with pytest.raises(NotImplementedError):
        compute_ssim(None, None)


def test_lpips_not_implemented():
    with pytest.raises(NotImplementedError):
        compute_lpips(None, None)
