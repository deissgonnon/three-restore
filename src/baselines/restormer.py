"""Wrapper pour la méthode de référence restormer.

Config : configs/baselines.yaml -> restormer.
"""

from .base_baseline import BaseBaseline


class RestormerBaseline(BaseBaseline):
    """Wrapper restormer, implémentant l'interface BaseBaseline."""

    def restore(self, image):
        raise NotImplementedError
