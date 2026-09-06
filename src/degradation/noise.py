"""Fractal noise generation (Perlin-like via interpolation)."""

import numpy as np
import cv2


def fractal_noise(shape: tuple, octaves: int = 6, persistence: float = 0.55,
                  base_scale: int = 8, seed: int = 0) -> np.ndarray:
    """Generate fractal noise field for synthetic depth maps."""
    rng = np.random.default_rng(seed)
    h, w = shape
    noise = np.zeros(shape, dtype=float)
    amplitude, total = 1.0, 0.0
    
    for o in range(octaves):
        scale = base_scale * (2 ** o)
        gh, gw = max(h // scale, 2), max(w // scale, 2)
        grid = rng.random((gh, gw))
        layer = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)[:h, :w]
        noise += amplitude * layer
        total += amplitude
        amplitude *= persistence
    
    noise /= total
    noise = cv2.GaussianBlur(noise.astype(np.float32), (0, 0), sigmaX=2.0)
    noise -= noise.min()
    noise /= (noise.max() + 1e-8)
    return np.clip(noise, 0, 1)
