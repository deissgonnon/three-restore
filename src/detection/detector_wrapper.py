"""Wrapper autour du détecteur fixe, utilisé pour tester images
nettes, dégradées et restaurées avec le même détecteur

Réutilise l'interface commune (src/detection/base.py) et la factory.
Config : configs/detection.yaml.
"""

from src.detection.base import BaseDetector, Detection
from src.detection.registry import build_detector


class FixedDetector:
    """Wrapper détecteur fixe pour l'évaluation en aval."""

    def __init__(self, weights_path: str, confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.45, name: str = "yolo"):
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self._detector: BaseDetector = build_detector({
            "name": name,
            "weights": weights_path,
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold,
        })

    def predict(self, image) -> list:
        """Retourne une liste de Detection."""
        return self._detector.predict(image)
