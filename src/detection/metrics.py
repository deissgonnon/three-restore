"""Calcul des métriques de détection : mAP@50, mAP@50:95, AP par
catégorie (véhicule/piéton), rappel, AP petits objets.

Config : configs/detection.yaml -> metrics.
"""


def compute_detection_metrics(predictions, ground_truth):
    """Calcule l'ensemble des métriques de détection requises."""
    raise NotImplementedError
