"""Interface commune à toutes les méthodes de référence, pour garantir
une comparaison équitable (doc de suivi section 6.2).
"""


class BaseBaseline:
    """Interface commune : charge les poids et restaure une image."""

    def __init__(self, weights_path: str = None):
        self.weights_path = weights_path

    def restore(self, image):
        raise NotImplementedError
