"""Tests pour src/baselines/moce_ir.py"""

import pytest
from src.baselines.moce_ir import MoceIrBaseline


def test_restore_not_implemented():
    baseline = MoceIrBaseline()
    with pytest.raises(NotImplementedError):
        baseline.restore(image=None)
