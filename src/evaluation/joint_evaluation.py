"""Évaluation conjointe restauration + détection, pour produire le
tableau de résultats de référence (doc de suivi section 6.5).

Utilise le même détecteur fixe (src/detection/detector_wrapper.py) sur
images nettes, dégradées et restaurées.
"""

from .restoration_metrics import compute_psnr, compute_ssim, compute_lpips
from ..detection.metrics import compute_detection_metrics


def evaluate_method(method_name, degradation_type, dataloader, model, detector):
    """Calcule métriques de restauration + détection pour une méthode donnée."""
    raise NotImplementedError
