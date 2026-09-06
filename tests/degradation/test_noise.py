"""Tests pour src/degradation/noise.py"""

import pytest
from src.degradation.noise import fractal_noise


def test_fractal_noise_not_implemented():
    with pytest.raises(NotImplementedError):
        fractal_noise((64, 64), seed=42)
