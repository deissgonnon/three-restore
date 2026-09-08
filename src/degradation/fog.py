"""Synthèse de brouillard réaliste (Koschmieder + Depth Anything V2)."""

import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class FogGenerator:
    """Génère du brouillard synthétique via Koschmieder + profondeur estimée."""

    def __init__(self, device: str = "cuda", depth_model: str = None):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.depth_model_name = depth_model or "depth-anything/Depth-Anything-V2-Small-hf"
        self.processor = None
        self.model = None

    def _load_depth_model(self):
        if self.processor is None:
            self.processor = AutoImageProcessor.from_pretrained(self.depth_model_name)
            self.model = AutoModelForDepthEstimation.from_pretrained(self.depth_model_name).to(self.device)
            self.model.eval()

    def _estimate_depth(self, image_rgb: np.ndarray) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        pil_img = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            depth = torch.nn.functional.interpolate(
                outputs.predicted_depth.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth = depth.cpu().numpy()
        p1, p99 = np.percentile(depth, [1, 99])
        depth = (depth - p1) / (p99 - p1 + 1e-8)
        depth = np.clip(depth, 0, 1)
        depth = 1.0 - depth
        depth = cv2.GaussianBlur(depth.astype(np.float32), (0, 0), sigmaX=1.5)
        return np.clip(depth, 0, 1)

    def _fog_field(self, h: int, w: int, strength: float, heterogeneity: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 1, (h, w)).astype(np.float32)
        sigma = max(20, min(h, w) / 12)
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma)
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

        noise2 = rng.normal(0, 1, (h, w)).astype(np.float32)
        sigma2 = max(50, min(h, w) / 5)
        noise2 = cv2.GaussianBlur(noise2, (0, 0), sigmaX=sigma2)
        noise2 = (noise2 - noise2.min()) / (noise2.max() - noise2.min() + 1e-8)
        print(f"Fog field noise stats: min={noise.min():.4f}, max={noise.max():.4f}, mean={noise.mean():.4f}, std={noise.std():.4f}")
        print(f"Fog field noise2 stats: min={noise2.min():.4f}, max={noise2.max():.4f}, mean={noise2.mean():.4f}, std={noise2.std():.4f}")

        field = 0.70 * noise + 0.30 * noise2
        print(f"Fog field combined stats: min={field.min():.4f}, max={field.max():.4f}, mean={field.mean():.4f}, std={field.std():.4f}")
        field -= field.mean()
        field /= (np.std(field) + 1e-8)
        print(f"Fog field normalized stats: min={field.min():.4f}, max={field.max():.4f}, mean={field.mean():.4f}, std={field.std():.4f}")
        field = 1.0 + heterogeneity * field
        # On verifie si le cip coupe assez de valeurs 
        print(f"Fog field stats: min={field.min():.4f}, max={field.max():.4f}, mean={field.mean():.4f}, std={field.std():.4f}")
        field = np.clip(field, 1.0 - 1.5 * heterogeneity, 1.0 + 1.5 * heterogeneity)
        return field.astype(np.float32)

    def _density_map(self, depth: np.ndarray, strength: float, heterogeneity: float, seed: int) -> np.ndarray:
        h, w = depth.shape
        spatial_field = self._fog_field(h, w, strength, heterogeneity, seed)
        base_density = 0.05 + 0.9 * strength
        depth_component = depth ** 1.35
        depth_density = (0.1 + 2.0 * strength) * depth_component
        density = base_density + depth_density
        density *= spatial_field
        return np.clip(density, 0, 8.0).astype(np.float32)

    def _transmission(self, depth: np.ndarray, strength: float, heterogeneity: float, seed: int) -> np.ndarray:
        density = self._density_map(depth, strength, heterogeneity, seed)
        effective_distance = 0.18 + 0.82 * depth
        optical_depth = density * effective_distance
        t = np.exp(-optical_depth)
        return np.clip(t, 0.02, 1.0).astype(np.float32)

    def _atmospheric_light(self, image: np.ndarray, strength: float) -> np.ndarray:
        img = image.astype(np.float32) / 255.0
        brightness = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        threshold = np.percentile(brightness, 97)
        mask = brightness >= threshold
        
        if np.sum(mask) > 100:
            a = img[mask].mean(axis=0)
        else:
            a = np.array([0.82, 0.85, 0.88], dtype=np.float32)

        daylight = np.array([0.84, 0.87, 0.90], dtype=np.float32)
        a = 0.70 * a + 0.30 * daylight
        a = (1 - 0.20 * strength) * a + (0.20 * strength) * daylight
        return np.clip(a, 0, 1)


    def apply(self, image: np.ndarray, strength: float, heterogeneity: float, seed: int = 42) -> np.ndarray:
        """Apply fog to image. Image should be RGB uint8."""
        self._load_depth_model()
        
        depth = self._estimate_depth(image)
        t = self._transmission(depth, strength, heterogeneity, seed)
        a = self._atmospheric_light(image, strength)

        j = image.astype(np.float32) / 255.0
        t3 = t[:, :, None]
        foggy = j * t3 + a[None, None, :] * (1.0 - t3)
        foggy = (np.clip(foggy, 0, 1) * 255).astype(np.uint8)

        return foggy
