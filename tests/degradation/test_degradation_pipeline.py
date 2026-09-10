"""Test du pipeline de dégradation avec les configurations YAML."""

from pathlib import Path
import random

import numpy as np
import yaml
from PIL import Image

from src.degradation.degradation_pipeline import DegradationPipeline


def test_degradation_pipeline():
    # Chargement de la configuration de dégradation
    with Path("configs/degradation.yaml").open(encoding="utf-8") as config_file:
        degradation_config = yaml.safe_load(config_file)

    # Chargement d'une image test depuis le dataset
    with Path("configs/datasets.yaml").open(encoding="utf-8") as config_file:
        dataset_config = yaml.safe_load(config_file)["datasets"]["visdrone"]

    images_dir = Path(dataset_config["train_dir"]) / "images"
    image_paths = [
        path
        for path in images_dir.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    assert image_paths, f"Aucune image trouvée dans {images_dir}"

    
    image_path = random.choice(image_paths)
    #image_path = "data/degradation_test/pipeline_clear.jpg"  # Décommenter pour tester une image spécifique
    image = np.array(Image.open(image_path).convert("RGB"))

    # Initialisation du pipeline avec la configuration YAML
    pipeline = DegradationPipeline(degradation_config)

    # Application de toutes les dégradations
    results = pipeline.apply_all(image)

    # Vérifications des sorties
    assert "clear" in results
    assert "fog" in results
    assert "rain" in results
    assert "lowlight" in results

    for name, degraded_image in results.items():
        assert degraded_image.shape == image.shape, f"La forme de l'image {name} est incorrecte"
        assert degraded_image.dtype == np.uint8, f"Le type de données (dtype) de l'image {name} est incorrect"

    # Sauvegarde des images pour inspection visuelle
    output_dir = Path("data/degradation_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, degraded_image in results.items():
        Image.fromarray(degraded_image).save(output_dir / f"pipeline_{name}.jpg")

    print(f"Image testée : {image_path}")
    for name in results:
        print(f"Image {name} : {output_dir / f'pipeline_{name}.jpg'}")


def test_degradation_pipeline_with_seed_base_none():
    with Path("configs/degradation.yaml").open(encoding="utf-8") as config_file:
        degradation_config = yaml.safe_load(config_file)

    degradation_config["seed_base"] = None
    pipeline = DegradationPipeline(degradation_config)

    dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)
    results = pipeline.apply_all(dummy_image, seed=None)

    assert "clear" in results
    assert "fog" in results
    assert "rain" in results
    assert "lowlight" in results

    for name, degraded_image in results.items():
        assert degraded_image.shape == dummy_image.shape
        assert degraded_image.dtype == np.uint8

