"""Pertes classiques utilisées en complément de la perte SOD (L1, perceptuelle).

Pondérations dans configs/loss.yaml -> weights.
"""

import torch.nn as nn


class PerceptualLoss(nn.Module):
    """Perte perceptuelle (features d'un réseau pré-entraîné)."""

    def __init__(self):
        super().__init__()

    def forward(self, restored, target):
        raise NotImplementedError
