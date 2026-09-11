"""Test réel de classification d'une image du split train."""

from pathlib import Path
from random import choice

import yaml
from PIL import Image
from src.preprocessing.day_night_classifier import DayNightClassifier


def test_classify_train_image():
    with Path("configs/datasets.yaml").open(encoding="utf-8") as config_file:
        dataset = yaml.safe_load(config_file)["datasets"]["visdrone"]

    images_dir = Path(dataset["train_dir"]) / "images"
    image_paths = [
        path
        for path in images_dir.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    assert image_paths, f"Aucune image trouvée dans {images_dir}"
    image_path = choice(image_paths)

    classifier = DayNightClassifier()
    with Image.open(image_path) as image:
        is_night = classifier.is_night(image)
        #image.show()

    print(f"Image testée : {image_path}")
    print(f"Classification : {'nuit' if is_night else 'jour'}")
