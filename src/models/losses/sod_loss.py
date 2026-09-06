"""Perte de détails des objets sensible à l'échelle (Scale-aware Object
Detail Loss, "perte SOD").

Renforce la supervision des petites zones correspondant aux véhicules et
piétons pendant l'entraînement à la restauration, afin de réduire la
perte des bords, contours et textures (doc de suivi section 1.2).
Config : configs/loss.yaml -> sod_loss.
"""

import torch.nn as nn


class ScaleAwareObjectDetailLoss(nn.Module):
    """Perte SOD : supervision renforcée sur les petits objets."""

    def __init__(self, small_object_scale_threshold: int = 32,
                 boost_factor: float = 2.0):
        super().__init__()
        self.small_object_scale_threshold = small_object_scale_threshold
        self.boost_factor = boost_factor

    def forward(self, restored, target, object_masks=None):
        raise NotImplementedError
