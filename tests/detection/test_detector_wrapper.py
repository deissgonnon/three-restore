"""Tests pour src/detection/detector_wrapper.py"""

from src.detection.detector_wrapper import FixedDetector


def test_fixed_detector_initialization():
    detector = FixedDetector(weights_path="dummy.pt")
    assert detector.weights_path == "dummy.pt"
    assert detector.confidence_threshold == 0.25
    assert detector.iou_threshold == 0.45
