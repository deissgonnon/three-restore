"""Synthèse de faible luminosité (albumentations + configuration progressive)."""

import numpy as np
import albumentations as A


class LowlightGenerator:
    """Génère des images de faible luminosité via albumentations avec paramètres aléatoires."""

    @classmethod
    def apply_with_range(cls, image: np.ndarray, brightness_limit: tuple, contrast_limit: tuple,
                        shot_noise_scale: float = 0.0, saturation: float = 1.0, seed: int = 42) -> np.ndarray:
        """Apply lowlight with parameters sampled from ranges."""
        transforms = [
            A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=1.0)
        ]
        if shot_noise_scale > 0:
            transforms.append(A.ShotNoise(scale_range=(shot_noise_scale, shot_noise_scale), p=1.0))
        if saturation < 1.0:
            transforms.append(A.ColorJitter(saturation=(saturation, saturation), hue=0, p=1.0))
        
        transform = A.Compose(transforms)
        result = transform(image=image, random_state=seed)
        return result["image"]
