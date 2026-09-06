"""Tests pour src/evaluation/joint_evaluation.py"""

import pytest
from src.evaluation.joint_evaluation import evaluate_method


def test_evaluate_method_not_implemented():
    with pytest.raises(NotImplementedError):
        evaluate_method("dep_rnet", "fog", dataloader=None, model=None, detector=None)
