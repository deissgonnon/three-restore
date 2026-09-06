"""Tests pour src/baselines/airnet.py"""

import pytest
from src.baselines.airnet import AirnetBaseline


def test_restore_not_implemented():
    baseline = AirnetBaseline()
    with pytest.raises(NotImplementedError):
        baseline.restore(image=None)
