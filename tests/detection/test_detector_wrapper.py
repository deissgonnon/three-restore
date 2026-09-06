"""Tests pour src/detection/detector_wrapper.py"""

import pytest
from src.detection.detector_wrapper import FixedDetector


def test_predict_not_implemented():
    detector = FixedDetector(weights_path="dummy.pt")
    with pytest.raises(NotImplementedError):
        detector.predict(image=None)
