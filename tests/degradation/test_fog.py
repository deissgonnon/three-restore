"""Tests pour src/degradation/fog.py"""

import pytest
from src.degradation.fog import synthesize_fog


def test_synthesize_fog_not_implemented():
    with pytest.raises(NotImplementedError):
        synthesize_fog(image=None, beta=1.0, atmospheric_light=0.9, seed=42)
