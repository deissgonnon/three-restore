"""Tests pour src/baselines/promptir.py"""

import pytest
from src.baselines.promptir import PromptirBaseline


def test_restore_not_implemented():
    baseline = PromptirBaseline()
    with pytest.raises(NotImplementedError):
        baseline.restore(image=None)
