"""Synthèse de pluie avec les profils Albumentations retenus."""

import numpy as np
import albumentations as A


class RainGenerator:
    """Génère la pluie avec ``A.RandomRain`` comme dans le notebook."""

    RAIN_TYPES = ("drizzle", "default", "heavy", "torrential")

    @classmethod
    def apply(
        cls,
        image: np.ndarray,
        rain_type: str = "default",
        drop_length: int | None = None,
        drop_width: int = 1,
        slant_range: tuple[float, float] = (-10, 10),
        seed: int = 42,
    ) -> np.ndarray:
        """Apply one of the four notebook rain profiles to an RGB uint8 image."""
        if rain_type not in cls.RAIN_TYPES:
            raise ValueError(f"Unknown rain type: {rain_type}. Expected one of {cls.RAIN_TYPES}.")

        transform = A.Compose(
            [
                A.RandomRain(
                    rain_type=rain_type,
                    drop_length=drop_length,
                    drop_width=drop_width,
                    slant_range=slant_range,
                    p=1.0,
                )
            ],
            seed=seed,
        )
        return transform(image=image)["image"]
