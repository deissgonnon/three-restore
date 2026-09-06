"""Tests pour src/detection/metrics.py"""

import pytest
from src.detection.metrics import compute_detection_metrics


def test_compute_detection_metrics_not_implemented():
    with pytest.raises(NotImplementedError):
        compute_detection_metrics(predictions=[], ground_truth=[])
