"""Chargement de configs/paths.yaml et résolution des chemins du projet."""

import yaml


def load_paths_config(config_path: str = "configs/paths.yaml") -> dict:
    """Charge configs/paths.yaml."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
