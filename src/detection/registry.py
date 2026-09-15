"""Registre et factory des détecteurs."""

from typing import Any, Dict, Type

from src.detection.base import BaseDetector

_REGISTRY: Dict[str, Type[BaseDetector]] = {}


def register(name: str):
    """Décorateur enregistrant une classe de détecteur par nom."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_detector(name: str) -> Type[BaseDetector]:
    """Retourne la classe de détecteur enregistrée sous `name`."""
    if name not in _REGISTRY:
        raise ValueError(f"Détecteur inconnu : {name!r}. "
                         f"Disponibles : {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_detectors() -> list:
    return sorted(_REGISTRY)


def build_detector(det_cfg: Dict[str, Any]) -> BaseDetector:
    """Construit un détecteur depuis la config `detector`."""
    name = det_cfg.get("name", "yolo")
    cls = get_detector(name)
    return cls(
        name=name,
        weights=det_cfg.get("weights"),
        confidence_threshold=det_cfg.get("confidence_threshold", 0.25),
        iou_threshold=det_cfg.get("iou_threshold", 0.45),
    )