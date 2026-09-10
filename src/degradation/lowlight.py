"""Synthèse de faible luminosité (albumentations + configuration progressive)."""

import numpy as np
import albumentations as A


class LowlightGenerator:
    """Génère des images de faible luminosité via albumentations avec paramètres aléatoires."""

    @staticmethod
    def compute_mean_luminance(image: np.ndarray) -> float:
        """Calcule la luminance moyenne perçue (BT.601) de l'image (0.0 - 255.0)."""
        if image.ndim == 2:
            return float(np.mean(image))
        return float(np.mean(0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]))

    @classmethod
    def adapt_gamma(
        cls,
        gamma: float,
        image: np.ndarray,
        ref_luminance: float = 120.0,
        power: float = 0.5,
        min_gamma: float = 1.3,
        max_gamma: float = 3.5,
    ) -> float:
        """Adapte le gamma en fonction de la luminance moyenne de l'image.

        - Image très claire (> ref_luminance) : gamma augmenté pour assombrir davantage.
        - Image déjà sombre (< ref_luminance) : gamma diminué pour préserver les détails.
        """
        mean_luma = cls.compute_mean_luminance(image)
        if mean_luma <= 1e-3 or ref_luminance <= 1e-3:
            return gamma

        is_scaled = gamma > 10.0
        g_raw = gamma / 100.0 if is_scaled else gamma

        ratio = (mean_luma / ref_luminance) ** power
        g_adapted = g_raw * ratio
        g_adapted = float(np.clip(g_adapted, min_gamma, max_gamma))

        return g_adapted * 100.0 if is_scaled else g_adapted

    @classmethod
    def apply_with_range(
        cls,
        image: np.ndarray,
        gamma_limit: tuple | float | int | None = None,
        brightness_limit: tuple | None = None,
        contrast_limit: tuple | None = None,
        shot_noise_scale: float = 0.0,
        saturation: float = 1.0,
        adaptive_gamma: bool = False,
        ref_luminance: float = 120.0,
        seed: int | None = 42,
    ) -> np.ndarray:
        """Apply lowlight with parameters sampled from ranges."""
        transforms = []

        if gamma_limit is not None:
            if isinstance(gamma_limit, (int, float)):
                g = float(gamma_limit)
                if adaptive_gamma:
                    g = cls.adapt_gamma(g, image, ref_luminance=ref_luminance)
                g_val = int(round(g * 100)) if g <= 10.0 else int(round(g))
                g_tuple = (g_val, g_val)
            else:
                g_min, g_max = float(gamma_limit[0]), float(gamma_limit[1])
                if adaptive_gamma:
                    g_min = cls.adapt_gamma(g_min, image, ref_luminance=ref_luminance)
                    g_max = cls.adapt_gamma(g_max, image, ref_luminance=ref_luminance)
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
            default_g = 200
            if adaptive_gamma:
                default_g = cls.adapt_gamma(default_g, image, ref_luminance=ref_luminance)
            transforms.append(A.RandomGamma(gamma_limit=(int(round(default_g)), int(round(default_g))), p=1.0))

        if shot_noise_scale > 0:
            transforms.append(A.ShotNoise(scale_range=(shot_noise_scale, shot_noise_scale), p=1.0))
        if saturation < 1.0:
            transforms.append(A.ColorJitter(saturation=(saturation, saturation), hue=0, p=1.0))

        transform = A.Compose(transforms)
        result = transform(image=image, random_state=seed)
        return result["image"]
