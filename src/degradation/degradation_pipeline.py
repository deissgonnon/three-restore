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
        self.seed_base = self._normalize_seed(config.get("seed_base"))
        self._rng = np.random.default_rng(self.seed_base) if self.seed_base is not None else None
        fog_config = config.get("fog", {})
        self.fog_gen = FogGenerator(
            depth_model=fog_config.get("depth_model"),
            blur_sigma=fog_config.get("blur_sigma", 1.5),
        )
        self.rain_gen = RainGenerator()
        self.lowlight_gen = LowlightGenerator()

    @classmethod
    def _normalize_seed(cls, seed: int | str | None) -> int | None:
        """Normalize seed to int or None (handles YAML 'None'/'null' strings)."""
        if seed is None:
            return None
        if isinstance(seed, str):
            clean = seed.strip().lower()
            if clean in ("none", "null", "~", ""):
                return None
            return int(seed)
        return int(seed)

    def _get_seed(self, seed: int | str | None = None) -> int | None:
        """Resolve seed from parameter or advance the reproducible sequence from seed_base."""
        if seed is not None:
            return self._normalize_seed(seed)
        if self._rng is not None:
            return int(self._rng.integers(0, 2**31 - 1))
        return None

    @staticmethod
    def _offset_seed(seed: int | None, offset: int) -> int | None:
        """Offset seed by an integer if seed is not None."""
        return None if seed is None else seed + offset

    def _sample_param(self, param_range: list, seed: int | None = None) -> float:
        """Sample from a triangular range: [minimum, mode, maximum]."""
        if len(param_range) != 3:
            raise ValueError("A triangular range must contain [minimum, mode, maximum].")
        if param_range[0] == param_range[1] == param_range[2]:
            return float(param_range[0])

        rng = np.random.default_rng(seed)
        return float(rng.triangular(param_range[0], param_range[1], param_range[2]))

    def apply_fog(self, image: np.ndarray, seed: int | None = None) -> np.ndarray:
        """Apply fog with random parameters from config range."""
        if not self.config.get("fog", {}).get("enable", False):
            return image
        
        seed = self._get_seed(seed)
        cfg = self.config["fog"]
        
        strength = self._sample_param(cfg["strength_range"], seed)
        heterogeneity = self._sample_param(cfg["heterogeneity_range"], self._offset_seed(seed, 1))
        
        return self.fog_gen.apply(
            image,
            strength=strength,
            heterogeneity=heterogeneity,
            seed=self._offset_seed(seed, 2),
        )

    def apply_rain(self, image: np.ndarray, seed: int | None = None) -> np.ndarray:
        """Apply a randomly selected notebook rain profile."""
        if not self.config.get("rain", {}).get("enable", False):
            return image
        
        seed = self._get_seed(seed)
        cfg = self.config["rain"]
        rain_types = cfg.get("rain_types", list(RainGenerator.RAIN_TYPES))
        rng = np.random.default_rng(seed)
        rain_type = rain_types[rng.integers(0, len(rain_types))]
        drop_length = cfg.get("drop_length")
        if isinstance(drop_length, list):
            drop_length = self._sample_param(drop_length, self._offset_seed(seed, 1))
            drop_length = int(round(drop_length))

        return self.rain_gen.apply(
            image,
            rain_type=rain_type,
            drop_length=drop_length,
            drop_width=cfg.get("drop_width", 1),
            slant_range=tuple(cfg.get("slant_range", (-10, 10))),
            seed=seed,
        )

    def apply_lowlight(self, image: np.ndarray, seed: int | None = None) -> np.ndarray:
        """Apply lowlight with random parameters from config range."""
        if not self.config.get("lowlight", {}).get("enable", False):
            return image
        
        seed = self._get_seed(seed)
        cfg = self.config["lowlight"]
        
        gamma_limit = None
        brightness_limit = None
        contrast_limit = None

        if "gamma_range" in cfg:
            gamma = self._sample_param(cfg["gamma_range"], seed)
            gamma_limit = (gamma, gamma)
        elif "gamma_limit_range" in cfg:
            gamma = self._sample_param(cfg["gamma_limit_range"], seed)
            gamma_limit = (gamma, gamma)
        elif "brightness_limit_range" in cfg and "contrast_limit_range" in cfg:
            brightness = self._sample_param(cfg["brightness_limit_range"], seed)
            contrast = self._sample_param(cfg["contrast_limit_range"], self._offset_seed(seed, 1))
            brightness_limit = (brightness, brightness)
            contrast_limit = (contrast, contrast)
        else:
            gamma_limit = (180, 250)

        shot_noise = self._sample_param(cfg.get("shot_noise_range", [0.0, 0.0, 0.0]), self._offset_seed(seed, 2))
        saturation = self._sample_param(cfg.get("saturation_range", [1.0, 1.0, 1.0]), self._offset_seed(seed, 3))
        
        adaptive_gamma = cfg.get("adaptive_gamma", False)
        ref_luminance = float(cfg.get("ref_luminance", 120.0))

        return self.lowlight_gen.apply_with_range(
            image,
            gamma_limit=gamma_limit,
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            shot_noise_scale=shot_noise,
            saturation=saturation,
            adaptive_gamma=adaptive_gamma,
            ref_luminance=ref_luminance,
            seed=seed,
        )

    def apply_all(self, image: np.ndarray, seed: int | None = None) -> dict:
        """Apply all enabled degradations, return dict with degradation_type -> image."""
        seed = self._get_seed(seed)
        results = {"clear": image}
        
        if self.config.get("fog", {}).get("enable", False):
            results["fog"] = self.apply_fog(image, seed=self._offset_seed(seed, 100))
        if self.config.get("rain", {}).get("enable", False):
            results["rain"] = self.apply_rain(image, seed=self._offset_seed(seed, 200))
        if self.config.get("lowlight", {}).get("enable", False):
            results["lowlight"] = self.apply_lowlight(image, seed=self._offset_seed(seed, 300))
        
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
            results = pipeline.apply_all(image)
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
