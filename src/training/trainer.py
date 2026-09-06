"""Boucle d'entraînement du réseau de restauration.

Hyperparamètres dans configs/training.yaml.
"""


class Trainer:
    """Orchestre l'entraînement : forward, perte, backward, logging."""

    def __init__(self, model, optimizer, loss_fn, config: dict):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config

    def train_one_epoch(self, dataloader):
        raise NotImplementedError

    def validate(self, dataloader):
        raise NotImplementedError
