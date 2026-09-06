"""Tests pour src/baselines/restormer.py"""

import pytest
from src.baselines.restormer import RestormerBaseline


def test_restore_not_implemented():
    baseline = RestormerBaseline()
    with pytest.raises(NotImplementedError):
        baseline.restore(image=None)
