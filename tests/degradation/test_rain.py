"""Tests pour src/degradation/rain.py"""

import pytest
from src.degradation.rain import synthesize_rain


def test_synthesize_rain_not_implemented():
    with pytest.raises(NotImplementedError):
        synthesize_rain(image=None, streak_density=0.3, streak_angle=10,
                         streak_length=20, seed=42)
