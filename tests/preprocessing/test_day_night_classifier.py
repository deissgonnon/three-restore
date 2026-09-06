"""Tests pour src/preprocessing/day_night_classifier.py"""

import pytest
from src.preprocessing.day_night_classifier import compute_presence_scores, classify_day_night


def test_compute_presence_scores_not_implemented():
    with pytest.raises(NotImplementedError):
        compute_presence_scores(["dummy.jpg"])


def test_classify_day_night_not_implemented():
    with pytest.raises(NotImplementedError):
        classify_day_night([0.5])
