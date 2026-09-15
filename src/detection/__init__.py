"""Détection : interface commune, factory et plugins (YOLO, RT-DETR)."""

from src.detection.base import BaseDetector, Detection, TrainResult
from src.detection.registry import available_detectors, build_detector, get_detector, register

from src.detection import detectors  # noqa: F401  (enregistre les plugins)

__all__ = [
    "BaseDetector",
    "Detection",
    "TrainResult",
    "build_detector",
    "get_detector",
    "register",
    "available_detectors",
]
