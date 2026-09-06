"""Réseau de restauration principal DEP-RNet.

Architecture définie dans configs/model.yaml. Intègre la branche DEP
(models/dep_branch/) comme module auxiliaire.
"""

import torch.nn as nn


class DEPRNet(nn.Module):
    """Réseau de restauration préservant les preuves de détection."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 base_channels: int = 64, num_blocks: int = 4):
        super().__init__()
        # TODO: définir l'architecture (backbone + intégration branche DEP)

    def forward(self, x):
        raise NotImplementedError
