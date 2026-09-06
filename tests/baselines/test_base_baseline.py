"""Tests pour src/baselines/base_baseline.py"""

import pytest
from src.baselines.base_baseline import BaseBaseline


def test_restore_not_implemented():
    baseline = BaseBaseline()
    with pytest.raises(NotImplementedError):
        baseline.restore(None)
