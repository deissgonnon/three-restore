"""Détecteurs ultralytics (YOLO, RT-DETR)."""

from pathlib import Path
from typing import Any, Dict, List

from src.detection.base import BaseDetector, Detection, TrainResult
from src.detection.registry import register

ROOT = Path(__file__).resolve().parent.parent.parent


class UltralyticsDetector(BaseDetector):
    """Base commune aux modèles ultralytics."""

    _model_cls = None

    def __init__(self, name: str, weights: str = None,
                 confidence_threshold: float = 0.25, iou_threshold: float = 0.45):
        super().__init__(name, weights, confidence_threshold, iou_threshold)
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            if self._model_cls is None:
                raise NotImplementedError
            self._model = self._model_cls(self.weights)
        return self._model

    def train(self, data_cfg: Dict[str, Any], train_cfg: Dict[str, Any]) -> TrainResult:
        model = self._ensure_model()
        run_name = train_cfg.get("name") or self._default_run_name()
        kwargs = {
            "data": str(data_cfg["dataset_yaml"]),
            "epochs": train_cfg.get("epochs", 100),
            "imgsz": train_cfg.get("imgsz", 640),
            "batch": train_cfg.get("batch", 16),
            "lr0": train_cfg.get("lr0", 0.01),
            "optimizer": train_cfg.get("optimizer", "auto"),
            "patience": train_cfg.get("patience", 20),
            "seed": train_cfg.get("seed", 42),
            "device": train_cfg.get("device", 0),
            "workers": train_cfg.get("workers", 8),
            "project": str(ROOT / train_cfg.get("project", "outputs/checkpoints/detector")),
            "name": run_name,
            "exist_ok": True,
            "verbose": True,
        }
        if train_cfg.get("deterministic") is not None:
            kwargs["deterministic"] = train_cfg["deterministic"]
        results = model.train(**kwargs)
        save_dir = Path(results.save_dir)
        return TrainResult(
            save_dir=save_dir,
            best_weights=save_dir / "weights" / "best.pt",
            metrics=dict(results.results_dict or {}),
        )

    def _default_run_name(self) -> str:
        """Nom de run dérivé du modèle entraîné (ex: yolov8s_visdrone)."""
        stem = Path(self.weights).stem if self.weights else self.name
        return f"{stem}_visdrone"

    def predict(self, image) -> List[Detection]:
        res = self._ensure_model().predict(
            image, conf=self.confidence_threshold, iou=self.iou_threshold, verbose=False
        )[0]
        detections = []
        if res.boxes is not None and len(res.boxes) > 0:
            for box, label, score in zip(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.cls.cpu().numpy().astype(int),
                res.boxes.conf.cpu().numpy(),
            ):
                detections.append(Detection(
                    box=[float(v) for v in box], label=int(label), score=float(score)
                ))
        return detections

    def save(self, path: str) -> None:
        self._ensure_model().save(path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "UltralyticsDetector":
        inst = cls(
            name=kwargs.get("name", cls.__name__.lower()),
            weights=path,
            confidence_threshold=kwargs.get("confidence_threshold", 0.25),
            iou_threshold=kwargs.get("iou_threshold", 0.45),
        )
        inst._model = cls._model_cls(path)
        return inst


@register("yolo")
class YoloDetector(UltralyticsDetector):
    """YOLO (v8/11/12, backbones n/s/m/l/x)."""

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.weights)
        return self._model

    @classmethod
    def load(cls, path: str, **kwargs) -> "YoloDetector":
        inst = cls(
            name=kwargs.get("name", "yolo"),
            weights=path,
            confidence_threshold=kwargs.get("confidence_threshold", 0.25),
            iou_threshold=kwargs.get("iou_threshold", 0.45),
        )
        from ultralytics import YOLO
        inst._model = YOLO(path)
        return inst


@register("rtdetr")
class RTDetrDetector(UltralyticsDetector):
    """RT-DETR (rtdetr-l, rtdetr-x)."""

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import RTDETR
            self._model = RTDETR(self.weights)
        return self._model

    @classmethod
    def load(cls, path: str, **kwargs) -> "RTDetrDetector":
        inst = cls(
            name=kwargs.get("name", "rtdetr"),
            weights=path,
            confidence_threshold=kwargs.get("confidence_threshold", 0.25),
            iou_threshold=kwargs.get("iou_threshold", 0.45),
        )
        from ultralytics import RTDETR
        inst._model = RTDETR(path)
        return inst