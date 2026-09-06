"""Wrapper pour la méthode de référence moce_ir.

Config : configs/baselines.yaml -> moce_ir.
"""

from .base_baseline import BaseBaseline


class MoceIrBaseline(BaseBaseline):
    """Wrapper moce_ir, implémentant l'interface BaseBaseline."""

    def restore(self, image):
        raise NotImplementedError
