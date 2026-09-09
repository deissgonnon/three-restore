"""Pipeline generating degraded images (fog, rain, lowlight) from clear image."""

import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from .fog import FogGenerator
from .rain import RainGenerator
from .lowlight import LowlightGenerator


class DegradationPipeline:
    """Orchestrates fog, rain, and lowlight degradation with randomizable parameters."""

    def __init__(self, config: dict):
        self.config = config
        fog_config = config.get("fog", {})
        self.fog_gen = FogGenerator(
            depth_model=fog_config.get("depth_model"),
            blur_sigma=fog_config.get("blur_sigma", 1.5),
        )
        self.rain_gen = RainGenerator()
        self.lowlight_gen = LowlightGenerator()

    def _sample_param(self, param_range: list, seed: int) -> float:
        """Sample from a triangular range: [minimum, mode, maximum]."""
        if len(param_range) != 3:
            raise ValueError("A triangular range must contain [minimum, mode, maximum].")
        if param_range[0] == param_range[1] == param_range[2]:
            return float(param_range[0])

        rng = np.random.default_rng(seed)
        return float(rng.triangular(param_range[0], param_range[1], param_range[2]))

    def apply_fog(self, image: np.ndarray, seed: int = None) -> np.ndarray:
        """Apply fog with random parameters from config range."""
        if not self.config.get("fog", {}).get("enable", False):
            return image
        
        seed = seed or self.config.get("seed_base", 42)
        cfg = self.config["fog"]
        
        strength = self._sample_param(cfg["strength_range"], seed)
        heterogeneity = self._sample_param(cfg["heterogeneity_range"], seed + 1)
        
        return self.fog_gen.apply(image, strength=strength, heterogeneity=heterogeneity, seed=seed + 2)

    def apply_rain(self, image: np.ndarray, seed: int = None) -> np.ndarray:
        """Apply a randomly selected notebook rain profile."""
        if not self.config.get("rain", {}).get("enable", False):
            return image
        
        seed = seed or self.config.get("seed_base", 42)
        cfg = self.config["rain"]
        rain_types = cfg.get("rain_types", list(RainGenerator.RAIN_TYPES))
        rain_type = rain_types[np.random.default_rng(seed).integers(0, len(rain_types))]
        drop_length = cfg.get("drop_length")
        if isinstance(drop_length, list):
            drop_length = self._sample_param(drop_length, seed + 1)
            drop_length = int(round(drop_length))

        return self.rain_gen.apply(
            image,
            rain_type=rain_type,
            drop_length=drop_length,
            drop_width=cfg.get("drop_width", 1),
            slant_range=tuple(cfg.get("slant_range", (-10, 10))),
            seed=seed,
        )

    def apply_lowlight(self, image: np.ndarray, seed: int = None) -> np.ndarray:
        """Apply lowlight with random parameters from config range."""
        if not self.config.get("lowlight", {}).get("enable", False):
            return image
        
        seed = seed or self.config.get("seed_base", 42)
        cfg = self.config["lowlight"]
        
        brightness_limit = self._sample_param(cfg["brightness_limit_range"], seed)
        contrast_limit = self._sample_param(cfg["contrast_limit_range"], seed + 1)
        shot_noise = self._sample_param(cfg["shot_noise_range"], seed + 2)
        saturation = self._sample_param(cfg["saturation_range"], seed + 3)
        
        return self.lowlight_gen.apply_with_range(
            image,
            brightness_limit=(brightness_limit, brightness_limit),
            contrast_limit=(contrast_limit, contrast_limit),
            shot_noise_scale=shot_noise,
            saturation=saturation,
            seed=seed
        )

    def apply_all(self, image: np.ndarray, seed: int = None) -> dict:
        """Apply all enabled degradations, return dict with degradation_type -> image."""
        seed = seed or self.config.get("seed_base", 42)
        results = {"clear": image}
        
        if self.config.get("fog", {}).get("enable", False):
            results["fog"] = self.apply_fog(image, seed=seed + 100)
        if self.config.get("rain", {}).get("enable", False):
            results["rain"] = self.apply_rain(image, seed=seed + 200)
        if self.config.get("lowlight", {}).get("enable", False):
            results["lowlight"] = self.apply_lowlight(image, seed=seed + 300)
        
        return results


def main() -> None:
    with Path("configs/degradation.yaml").open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    pipeline = DegradationPipeline(config)
    output_root = Path("data/synthetic")
    source_root = Path("data/splits")
    for split_name in ("train", "val", "test"):
        split_dir = source_root / split_name
        images_dir = split_dir / "images"
        annotations_dir = split_dir / "annotations"
        if not images_dir.exists():
            raise FileNotFoundError(
                f"Split de jour absent: {images_dir}. "
                "Lancer d'abord python -m src.preprocessing.day_night_classifier."
            )
        image_paths = sorted(
            path
            for path in images_dir.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        for image_path in tqdm(image_paths, desc=f"Dégradations {split_name}"):
            image = np.array(Image.open(image_path).convert("RGB"))
            results = pipeline.apply_all(image, seed=config.get("seed_base", 42))
            relative_path = image_path.relative_to(images_dir)
            annotation = annotations_dir / relative_path.with_suffix(".txt")

            for degradation_name, degraded_image in results.items():
                if degradation_name == "clear":
                    continue
                destination = output_root / degradation_name / split_name
                output_image = destination / "images" / relative_path
                output_image.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(degraded_image).save(output_image)

                if annotation.exists():
                    output_annotation = destination / "annotations" / relative_path.with_suffix(".txt")
                    output_annotation.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(annotation, output_annotation)


if __name__ == "__main__":
    main()
