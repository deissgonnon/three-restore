"""Wrapper pour la méthode de référence promptir.

Config : configs/baselines.yaml -> promptir.
"""

from .base_baseline import BaseBaseline


class PromptirBaseline(BaseBaseline):
    """Wrapper promptir, implémentant l'interface BaseBaseline."""

    def restore(self, image):
        raise NotImplementedError
