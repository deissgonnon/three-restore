"""Wrapper autour du détecteur fixe (YOLO), utilisé pour tester images
nettes, dégradées et restaurées avec le même détecteur (doc de suivi
section 6.4 : isole l'effet du module de restauration).

Config : configs/detection.yaml.
"""


class FixedDetector:
    """Wrapper détecteur fixe pour l'évaluation en aval."""

    def __init__(self, weights_path: str, confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.45):
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def predict(self, image):
        raise NotImplementedError
