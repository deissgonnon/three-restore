"""Tests pour src/detection/factory.py et registry.py"""

import pytest

from src.detection import available_detectors, build_detector, get_detector


def test_available_detectors():
    assert "yolo" in available_detectors()
    assert "rtdetr" in available_detectors()


def test_get_detector_unknown():
    with pytest.raises(ValueError):
        get_detector("inexistant")


def test_build_detector_yolo():
    det = build_detector({"name": "yolo", "weights": "yolov8n.pt"})
    assert det.name == "yolo"
    assert det.weights == "yolov8n.pt"
    assert det.confidence_threshold == 0.25
    assert det.iou_threshold == 0.45


def test_build_detector_rtdetr():
    det = build_detector({"name": "rtdetr", "weights": "rtdetr-l.pt"})
    assert det.name == "rtdetr"
    assert det.weights == "rtdetr-l.pt"


def test_build_detector_unknown():
    with pytest.raises(ValueError):
        build_detector({"name": "inexistant"})