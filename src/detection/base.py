"""Interface commune des détecteurs."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Detection:
    """Une détection : boîte (xyxy), classe, score."""
    box: List[float]
    label: int
    score: float


@dataclass
class TrainResult:
    """Résultat d'un entraînement."""
    save_dir: Path
    best_weights: Path
    metrics: Dict[str, Any] = field(default_factory=dict)


class BaseDetector(ABC):
    """API commune à tous les détecteurs (YOLO, RT-DETR, ...)."""

    def __init__(self, name: str, weights: Optional[str] = None,
                 confidence_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.name = name
        self.weights = weights
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    @abstractmethod
    def train(self, data_cfg: Dict[str, Any], train_cfg: Dict[str, Any]) -> TrainResult:
        """Entraîne le détecteur et retourne le résultat."""

    @abstractmethod
    def predict(self, image) -> List[Detection]:
        """Prédit les détections sur une image."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Sauvegarde le modèle."""

    @classmethod
    @abstractmethod
    def load(cls, path: str, **kwargs) -> "BaseDetector":
        """Charge un modèle sauvegardé."""