"""Tests pour src/degradation/lowlight.py"""

import pytest
from src.degradation.lowlight import synthesize_lowlight


def test_synthesize_lowlight_not_implemented():
    with pytest.raises(NotImplementedError):
        synthesize_lowlight(image=None, gamma=2.0, noise_std=0.02,
                             color_shift_strength=0.1, seed=42)
