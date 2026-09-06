"""Sauvegarde et chargement des checkpoints d'entraînement.

Chemin de sortie : configs/paths.yaml -> outputs.checkpoints.
"""


def save_checkpoint(model, optimizer, epoch: int, checkpoint_dir: str):
    """Sauvegarde l'état du modèle et de l'optimiseur."""
    raise NotImplementedError


def load_checkpoint(model, optimizer, checkpoint_path: str):
    """Recharge un checkpoint pour reprendre l'entraînement."""
    raise NotImplementedError
