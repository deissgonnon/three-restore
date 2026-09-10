"""Synthèse de faible luminosité (albumentations + configuration progressive)."""

import numpy as np
import albumentations as A


class LowlightGenerator:
    """Génère des images de faible luminosité via albumentations avec paramètres aléatoires."""

    @classmethod
    def apply_with_range(
        cls,
        image: np.ndarray,
        gamma_limit: tuple | float | int | None = None,
        brightness_limit: tuple | None = None,
        contrast_limit: tuple | None = None,
        shot_noise_scale: float = 0.0,
        saturation: float = 1.0,
        seed: int | None = 42,
    ) -> np.ndarray:
        """Apply lowlight with parameters sampled from ranges."""
        transforms = []

        if gamma_limit is not None:
            if isinstance(gamma_limit, (int, float)):
                g = float(gamma_limit)
                g_val = int(round(g * 100)) if g <= 10.0 else int(round(g))
                g_tuple = (g_val, g_val)
            else:
                g_min, g_max = float(gamma_limit[0]), float(gamma_limit[1])
                g_min_val = int(round(g_min * 100)) if g_min <= 10.0 else int(round(g_min))
                g_max_val = int(round(g_max * 100)) if g_max <= 10.0 else int(round(g_max))
                g_tuple = (g_min_val, g_max_val)
            transforms.append(A.RandomGamma(gamma_limit=g_tuple, p=1.0))
        elif brightness_limit is not None and contrast_limit is not None:
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=brightness_limit,
                    contrast_limit=contrast_limit,
                    p=1.0,
                )
            )
        else:
            # Valeur par défaut si aucun paramètre de luminosité/gamma n'est passé
            transforms.append(A.RandomGamma(gamma_limit=(150, 250), p=1.0))

        if shot_noise_scale > 0:
            transforms.append(A.ShotNoise(scale_range=(shot_noise_scale, shot_noise_scale), p=1.0))
        if saturation < 1.0:
            transforms.append(A.ColorJitter(saturation=(saturation, saturation), hue=0, p=1.0))

        transform = A.Compose(transforms)
        result = transform(image=image, random_state=seed)
        return result["image"]
