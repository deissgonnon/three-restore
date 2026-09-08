"""Test réel de faible luminosité sur une image aléatoire du split train."""

from pathlib import Path
import random

import numpy as np
import yaml
from PIL import Image

from src.degradation.lowlight import LowlightGenerator


def test_apply_lowlight_on_random_train_image():
    with Path("configs/datasets.yaml").open(encoding="utf-8") as config_file:
        dataset = yaml.safe_load(config_file)["datasets"]["visdrone"]

    images_dir = Path(dataset["train_dir"]) / "images"
    image_paths = [
        path
        for path in images_dir.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    assert image_paths, f"Aucune image trouvée dans {images_dir}"

    random.seed(43)  # Pour la reproductibilité
    image_path = random.choice(image_paths)
    image = np.array(Image.open(image_path).convert("RGB"))
    lowlight_image = LowlightGenerator.apply_with_range(
        image=image,
        brightness_limit=(-0.2, -0.1),
        contrast_limit=(-0.4, 0.0),
        shot_noise_scale=0.0,
        saturation=0.8,
        seed=42,
    )

    assert lowlight_image.shape == image.shape
    assert lowlight_image.dtype == np.uint8

    output_dir = Path("data/degradation_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_dir / "original.jpg")
    Image.fromarray(lowlight_image).save(output_dir / "lowlight.jpg")
    print(f"Image testée : {image_path}")
    print(f"Image originale : {output_dir / 'original.jpg'}")
    print(f"Image faible luminosité : {output_dir / 'lowlight.jpg'}")
