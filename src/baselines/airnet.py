"""Wrapper pour la méthode de référence airnet.

Config : configs/baselines.yaml -> airnet.
"""

from .base_baseline import BaseBaseline


class AirnetBaseline(BaseBaseline):
    """Wrapper airnet, implémentant l'interface BaseBaseline."""

    def restore(self, image):
        raise NotImplementedError
