"""Filtrage des annotations par catégorie (véhicule/piéton).

Voir configs/dataset.yaml pour la liste des classes cibles et le
regroupement véhicule/piéton (doc de suivi section 5.2).
"""


def filter_annotations_by_category(annotations, target_classes):
    """Filtre une liste d'annotations selon les classes cibles."""
    raise NotImplementedError
