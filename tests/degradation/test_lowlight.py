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

    #random.seed(43)  # Pour la reproductibilité
    image_path = random.choice(image_paths)
    #image_path = "data/degradation_test/original.jpg"
    image = np.array(Image.open(image_path).convert("RGB"))
    lowlight_image = LowlightGenerator.apply_with_range(
        image=image,
        gamma_limit=(150, 250),
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


def test_adaptive_gamma():
    # Créer deux images synthétiques : une très claire et une sombre
    bright_img = np.full((100, 100, 3), 180, dtype=np.uint8)
    dark_img = np.full((100, 100, 3), 60, dtype=np.uint8)

    base_gamma = 2.4
    ref_luma = 120.0

    adapted_bright = LowlightGenerator.adapt_gamma(base_gamma, bright_img, ref_luminance=ref_luma)
    adapted_dark = LowlightGenerator.adapt_gamma(base_gamma, dark_img, ref_luminance=ref_luma)

    print(f"\nLuminance image claire: {LowlightGenerator.compute_mean_luminance(bright_img):.1f} -> Gamma adapté: {adapted_bright:.2f}")
    print(f"Luminance image sombre: {LowlightGenerator.compute_mean_luminance(dark_img):.1f} -> Gamma adapté: {adapted_dark:.2f}")

    # L'image claire doit avoir un gamma plus fort pour assombrir davantage
    assert adapted_bright > base_gamma
    # L'image sombre doit avoir un gamma plus doux pour ne pas noircir totalement
    assert adapted_dark < base_gamma
    assert adapted_bright > adapted_dark

    # Tester l'application avec adaptive_gamma=True
    out_bright = LowlightGenerator.apply_with_range(bright_img, gamma_limit=base_gamma, adaptive_gamma=True, ref_luminance=ref_luma)
    out_dark = LowlightGenerator.apply_with_range(dark_img, gamma_limit=base_gamma, adaptive_gamma=True, ref_luminance=ref_luma)

    assert out_bright.shape == bright_img.shape
    assert out_dark.shape == dark_img.shape
    # L'image sombre ne doit pas être écrasée à 0
    assert np.mean(out_dark) > 0
