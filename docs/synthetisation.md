# Report — Synthesis Methods for the Three Degradations

**Project:** DEP-RNet · Object Detection under Atmospheric Degradations (VisDrone)  
**Date:** 2026-09-13

This report describes the three degraded-image synthesis methods used to generate the training/validation/test dataset from clear VisDrone images: **fog**, **rain**, and **low light**.

---

## 1. Overview

| Degradation | File | Method | Random parameters |
|---|---|---|---|
| Fog | `src/degradation/fog.py` | Koschmieder physical model + estimated depth (Depth Anything V2) | strength, heterogeneity |
| Rain | `src/degradation/rain.py` | Albumentations `RandomRain` (linear streaks) | profile, drop length, slant |
| Low light | `src/degradation/lowlight.py` | Albumentations: adaptive `RandomGamma` + `ShotNoise` + `ColorJitter` | gamma, photon noise, saturation |

**Common principles** (`src/degradation/degradation_pipeline.py`):

- Each degradation is applied **independently to the clear image** (no composition between degradations) — `DegradationPipeline.apply_all` returns `{clear, fog, rain, lowlight}`.
- **No geometric transformation**: bounding boxes from annotations remain valid; annotations are simply copied (`degradation_pipeline.py:203-206`).
- Parameters are sampled from a **uniform distribution** within `[min, max]` ranges defined in `configs/degradation.yaml` (`_sample_param`, `degradation_pipeline.py:56-64`; commit `2c6f0af` replaced the triangular distribution with the uniform distribution).
- **Reproducibility**: `seed_base: 42` feeds a NumPy generator from which a seed is drawn per image, with offsets per degradation (+100 fog, +200 rain, +300 low light) and per parameter (+1, +2, +3) (`degradation_pipeline.py:152-164`). Two runs of the pipeline produce identical images (verified by `tests/degradation/test_degradation_pipeline.py::test_degradation_pipeline_seed_reproducibility_and_diversity`).
- **Input data**: daytime clear images only (night images are discarded by the SAM3 day/night classifier, `src/preprocessing/day_night_classifier.py`).

**Current configuration** (`configs/degradation.yaml`):

```yaml
fog:
  depth_model: "depth-anything/Depth-Anything-V2-Small-hf"
  strength_range: [0.50, 1.0]
  heterogeneity_range: [0.0, 0.35]
  blur_sigma: 1.5
rain:
  rain_types: [default, heavy, torrential]
  drop_length: [10, 50]
  drop_width: 1
  slant_range: [-20, 20]
lowlight:
  gamma_range: [200, 320]
  adaptive_gamma: true
  ref_luminance: 120.0
  shot_noise_range: [0.0, 0.03]
  saturation_range: [0.4, 1.0]
seed_base: 42
```

---

## 2. Fog — Koschmieder Model + Estimated Depth

`src/degradation/fog.py` — `FogGenerator`

### 2.1 Physical model

Fog is synthesized using the **Koschmieder** atmospheric scattering model:

```
I(x) = J(x) · t(x) + A · (1 − t(x))
```

where `J` is the clear image, `A` is the atmospheric light (color of the fog), and `t` is the transmission map, `t(x) = exp(−β · d(x))` (`fog.py:111-114`).

### 2.2 Depth estimation (`fog.py:26-46`)

1. Monocular depth estimated by **Depth Anything V2 Small** (`depth-anything/Depth-Anything-V2-Small-hf`) via HuggingFace Transformers, on GPU if available.
2. Bicubic resizing to the image size.
3. Robust percentile normalization: `(d − p1) / (p99 − p1)`.
4. Inversion (`depth = 1 − depth`) then smoothing by Gaussian blur (`blur_sigma = 1.5`).
5. Result: a **smoothed** map in `[0, 1]`.

### 2.3 Spatial heterogeneity (`fog.py:48-67`)

A **two-scale fractal noise spatial field** modulates density to avoid a uniform veil:

- two Gaussian noise fields smoothed by Gaussian blur at large scales (σ ≈ `min(h,w)/12` and σ ≈ `min(h,w)/5`), weighted 70% / 30%;
- standardization (zero mean, unit standard deviation), scaling by the `heterogeneity` parameter, then clipping to `±1.5 · heterogeneity`.

### 2.4 Density and transmission (`fog.py:69-84`)

```
density(x) = (0.05 + 0.9·s) + (0.1 + 2.0·s) · depth(x)^1.35  × field(x)
d_eff(x)   = 0.18 + 0.82 · depth(x)
t(x)       = exp(− density(x) · d_eff(x))   ∈ [0.02, 1.0]
```

where `s` = fog strength. Transmission can never reach 0 (clipped to `0.02`): no pixel is completely opaque.

### 2.5 Atmospheric light (`fog.py:86-100`)

- `A` is estimated as the average color of the brightest pixels (97th percentile of BT.601 luminance);
- mixed 70/30 with a constant "day" color `(0.84, 0.87, 0.90)`, then pulled further toward this constant as strength increases;
- fallback to the constant if fewer than 100 bright pixels.

### 2.6 Random parameters

| Parameter | Range (config) | Role |
|---|---|---|
| `strength` | [0.50, 1.0] | base density and depth component |
| `heterogeneity` | [0.0, 0.35] | amplitude of the spatial density field |

---

## 3. Rain — Linear Streaks (Albumentations `RandomRain`)

`src/degradation/rain.py` — `RainGenerator` (albumentations 2.0.8)

### 3.1 Principle

`A.RandomRain` draws **raindrops as straight line segments**:

1. The number of drops depends on the profile:
   `drizzle` = h/4, `default` = h/3, `heavy` = h, `torrential` = 2h
   (h = image height).
2. Each drop starts at a uniform random point `(x1, y1)` and ends at
   `(x2, y2) = (x1 + L·sin(θ), y1 + L·cos(θ))`, where `L` is the drop length and `θ` is the slant angle sampled uniformly from `slant_range`.
3. Segments are drawn with `cv2.line` in a light color `(200, 200, 200)` and width `drop_width`.

The retained "notebook" profiles are restricted by the configuration to the three most intense: `default`, `heavy`, `torrential`.

### 3.2 Random parameters

| Parameter | Range (config) | Role |
|---|---|---|
| `rain_type` | {default, heavy, torrential} | number of drops (h/3, h, 2h) |
| `drop_length` | [10, 50] px | streak length |
| `slant` | [−20°, +20°] | streak slant |
| `drop_width` | 1 (fixed) | streak thickness |
| `drop_color` | (200, 200, 200) (default) | streak color |

---

## 4. Low Light — Adaptive Gamma + Photon Noise + Desaturation

`src/degradation/lowlight.py` — `LowlightGenerator` (albumentations 2.0.8)

### 4.1 Gamma correction (`A.RandomGamma`)

- A gamma is sampled uniformly from `[200, 320]` (albumentations convention: `γ = value/100`, i.e., **γ ∈ [2.0, 3.2]**, γ > 1 darkens), then applied deterministically (`gamma_limit = (g, g)`).
- **Adaptive gamma** (`adapt_gamma`, `lowlight.py:17-43`, enabled by `adaptive_gamma: true`):
  - mean BT.601 luminance of the image compared to `ref_luminance = 120` with power 0.5: `γ' = γ · (luma/120)^0.5`, clipped to **[1.3, 3.5]**;
  - a **clear** image (luma > 120) is darkened more (gamma increased), while an **already dark** image receives a gentler gamma to preserve details (behavior tested in `tests/degradation/test_lowlight.py`).

### 4.2 Photon noise (`A.ShotNoise`)

- Poisson noise (sensor photon noise), scale sampled uniformly from `[0.0, 0.03]` — a very low range in the albumentations convention (0.1 = low noise), i.e., mild night/underexposure noise.

### 4.3 Desaturation (`A.ColorJitter`)

- Saturation sampled uniformly from `[0.4, 1.0]` (never oversaturation), to imitate color loss in low light.

### 4.4 Random parameters

| Parameter | Range (config) | Role |
|---|---|---|
| `gamma` | [2.0, 3.2] (adapted → [1.3, 3.5]) | darkening |
| `shot_noise` | [0.0, 0.03] | photon noise |
| `saturation` | [0.4, 1.0] | desaturation |

The generator also supports an alternative `RandomBrightnessContrast` mode (not used by the current configuration).

---

## 5. Pipeline Outputs

`python -m src.degradation.degradation_pipeline` reads `configs/degradation.yaml`, iterates over images in `data/splits/{train,val,test}`, and writes for each enabled degradation:

```
data/synthetic/{fog,rain,lowlight}/{split}/images/*.jpg   (degraded image)
data/synthetic/{fog,rain,lowlight}/{split}/annotations/*.txt (identical copy)
```

The degraded images are then randomly mixed during training of the reference models (e.g., MoCE-IR) via `MixedDegradationDataset`, with the clear images as targets.

---

## 6. References

- Koschmieder model: Koschmieder, H. (1924), *Theorie der horizontalen Sichtweite*.
- Depth estimation: Yang et al., *Depth Anything V2* (`depth-anything/Depth-Anything-V2-Small-hf` on Hugging Face).
- Augmentation library: albumentations **2.0.8** (`A.RandomRain`, `A.RandomGamma`, `A.ShotNoise`, `A.ColorJitter`).
- Code: `src/degradation/{fog,rain,lowlight,degradation_pipeline}.py`, `configs/degradation.yaml`, tests `tests/degradation/*`.