"""Branche de détection allégée avec préservation des preuves (branche DEP).

Objectif : améliorer les contours véhicules/piétons, le contraste local
et les détails à haute fréquence (doc de suivi section 1.2).
Config : configs/model.yaml -> dep_branch.
"""

import torch.nn as nn


class DEPBranch(nn.Module):
    """Branche légère de préservation des preuves de détection."""

    def __init__(self, channels: int = 32, num_layers: int = 3):
        super().__init__()
        # TODO: définir les couches de la branche allégée

    def forward(self, x):
        raise NotImplementedError
