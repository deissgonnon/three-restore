"""Test réel du brouillard sur une image aléatoire du split train."""

from pathlib import Path
import random

import numpy as np
import yaml
from PIL import Image

from src.degradation.fog import FogGenerator


def test_apply_fog_on_random_train_image():
    with Path("configs/datasets.yaml").open(encoding="utf-8") as config_file:
        dataset = yaml.safe_load(config_file)["datasets"]["visdrone"]

    images_dir = Path(dataset["train_dir"]) / "images"
    image_paths = [
        path
        for path in images_dir.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    
    assert image_paths, f"Aucune image trouvée dans {images_dir}"

    #random.seed(45)  # Pour la reproductibilité
    image_path = random.choice(image_paths)
    #image_path = "data/degradation_test/original.jpg"
    image = np.array(Image.open(image_path).convert("RGB"))
    foggy_image = FogGenerator().apply(
        image=image,
        strength = 0.6,
        heterogeneity=0.35 ,
        seed=428,
    )

    assert foggy_image.shape == image.shape
    assert foggy_image.dtype == np.uint8

    output_dir = Path("data/degradation_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_dir / "original.jpg")
    Image.fromarray(foggy_image).save(output_dir / "foggy.jpg")
    print(f"Image testée : {image_path}")
    print(f"Image originale : {output_dir / 'original.jpg'}")
    print(f"Image avec brouillard : {output_dir / 'foggy.jpg'}")
