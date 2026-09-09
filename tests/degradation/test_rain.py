"""Test réel de pluie sur une image aléatoire du split train."""

from pathlib import Path
import random

import numpy as np
import yaml
from PIL import Image

from src.degradation.rain import RainGenerator


def test_apply_rain_on_random_train_image():
    with Path("configs/datasets.yaml").open(encoding="utf-8") as config_file:
        dataset = yaml.safe_load(config_file)["datasets"]["visdrone"]

    images_dir = Path(dataset["train_dir"]) / "images"
    image_paths = [
        path
        for path in images_dir.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    assert image_paths, f"Aucune image trouvée dans {images_dir}"

    #random.seed(44)  # Pour la reproductibilité
    image_path = random.choice(image_paths)
    #image_path = "data/degradation_test/original.jpg"
    image = np.array(Image.open(image_path).convert("RGB"))
    rainy_image = RainGenerator.apply(
        image=image,
        rain_type="torrential",
        drop_length=20,
        drop_width=1,
        slant_range=(-20, 20),
        seed=42,
    )

    assert rainy_image.shape == image.shape
    assert rainy_image.dtype == np.uint8

    output_dir = Path("data/degradation_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_dir / "original.jpg")
    Image.fromarray(rainy_image).save(output_dir / "rainy.jpg")
    print(f"Image testée : {image_path}")
    print(f"Image originale : {output_dir / 'original.jpg'}")
    print(f"Image avec pluie : {output_dir / 'rainy.jpg'}")
