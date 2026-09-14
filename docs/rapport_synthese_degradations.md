# Rapport — Méthodes de synthèse des 3 dégradations

**Projet :** DEP-RNet · Détection d'objets sous dégradations atmosphériques (VisDrone)
**Date :** 2026-09-13

Ce rapport décrit les trois méthodes de synthèse d'images dégradées utilisées
pour générer le jeu de données d'entraînement/validation/test à partir des
images claires de VisDrone : **brouillard**, **pluie** et **faible luminosité**.

---

## 1. Vue d'ensemble

| Dégradation | Fichier | Méthode | Paramètres aléatoires |
|---|---|---|---|
| Brouillard | `src/degradation/fog.py` | Modèle physique de Koschmieder + profondeur estimée (Depth Anything V2) | force, hétérogénéité |
| Pluie | `src/degradation/rain.py` | Albumentations `RandomRain` (stries linéaires) | profil, longueur de goutte, inclinaison |
| Faible luminosité | `src/degradation/lowlight.py` | Albumentations : `RandomGamma` adaptatif + `ShotNoise` + `ColorJitter` | gamma, bruit de photons, saturation |

**Principes communs** (`src/degradation/degradation_pipeline.py`) :

- Chaque dégradation est appliquée **indépendamment sur l'image claire**
  (pas de composition entre dégradations) — `DegradationPipeline.apply_all`
  renvoie `{clear, fog, rain, lowlight}`.
- **Aucune transformation géométrique** : les boîtes englobantes des
  annotations restent valides, les annotations sont simplement copiées
  (`degradation_pipeline.py:203-206`).
- Les paramètres sont tirés selon une **distribution uniforme** dans des
  plages `[min, max]` définies dans `configs/degradation.yaml`
  (`_sample_param`, `degradation_pipeline.py:56-64` ; commit `2c6f0af` ayant
  remplacé la distribution triangulaire par l'uniforme).
- **Reproductibilité** : `seed_base: 42` alimente un générateur NumPy dont on
  tire une graine par image, avec des décalages par dégradation
  (+100 brouillard, +200 pluie, +300 faible luminosité) et par paramètre
  (+1, +2, +3) (`degradation_pipeline.py:152-164`). Deux exécutions du
  pipeline produisent des images identiques (vérifié par
  `tests/degradation/test_degradation_pipeline.py::test_degradation_pipeline_seed_reproducibility_and_diversity`).
- **Données d'entrée** : images claires de jour uniquement (les images de
  nuit sont écartées par le classifieur jour/nuit SAM3,
  `src/preprocessing/day_night_classifier.py`).

**Configuration actuelle** (`configs/degradation.yaml`) :

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

## 2. Brouillard — modèle de Koschmieder + profondeur estimée

`src/degradation/fog.py` — `FogGenerator`

### 2.1 Modèle physique

Le brouillard est synthétisé avec le modèle de diffusion atmosphérique de
**Koschmieder** :

```
I(x) = J(x) · t(x) + A · (1 − t(x))
```

où `J` est l'image claire, `A` la lumière atmosphérique (couleur du
brouillard) et `t` la carte de transmission, `t(x) = exp(−β · d(x))`
(`fog.py:111-114`).

### 2.2 Estimation de la profondeur (`fog.py:26-46`)

1. Profondeur monoscopique estimée par **Depth Anything V2 Small**
   (`depth-anything/Depth-Anything-V2-Small-hf`) via HuggingFace
   Transformers, sur GPU si disponible.
2. Redimensionnement bicubique à la taille de l'image.
3. Normalisation robuste par percentiles : `(d − p1) / (p99 − p1)`.
4. Inversion (`depth = 1 − depth`) puis lissage par flou gaussien
   (`blur_sigma = 1.5`).
5. Résultat : une carte **lissée** dans `[0, 1]`.


### 2.3 Hétérogénéité spatiale (`fog.py:48-67`)

Un **champ spatial de bruit fractal** à deux échelles module la densité pour
éviter un voile uniforme :

- deux champs de bruit gaussien lissés par flou gaussien à grande échelle
  (σ ≈ `min(h,w)/12` et σ ≈ `min(h,w)/5`), pondérés 70 % / 30 % ;
- standardisation (moyenne nulle, écart-type 1), mise à l'échelle par le
  paramètre `heterogeneity`, puis écrêtage à `±1.5 · heterogeneity`.

### 2.4 Densité et transmission (`fog.py:69-84`)

```
densité(x)  = (0.05 + 0.9·s) + (0.1 + 2.0·s) · profondeur(x)^1.35  × champ(x)
d_eff(x)    = 0.18 + 0.82 · profondeur(x)
t(x)        = exp(− densité(x) · d_eff(x))   ∈ [0.02, 1.0]
```

avec `s` = force du brouillard. La transmission ne peut jamais atteindre 0
(écrêtage à `0.02`) : aucun pixel n'est totalement opaque.

### 2.5 Lumière atmosphérique (`fog.py:86-100`)

- `A` est estimée comme la couleur moyenne des pixels les plus lumineux
  (percentile 97 de la luminance BT.601) ;
- mélangée à 70/30 avec une couleur « jour » constante `(0.84, 0.87, 0.90)`,
  puis d'autant plus tirée vers cette constante que la force augmente ;
- repli sur la constante si moins de 100 pixels lumineux.

### 2.6 Paramètres aléatoires

| Paramètre | Plage (config) | Rôle |
|---|---|---|
| `strength` | [0.50, 1.0] | densité de base et composante profondeur |
| `heterogeneity` | [0.0, 0.35] | amplitude du champ spatial de densité |

---

## 3. Pluie — stries linéaires (Albumentations `RandomRain`)

`src/degradation/rain.py` — `RainGenerator` (albumentations 2.0.8)

### 3.1 Principe

`A.RandomRain` dessine des **gouttes sous forme de segments de droite** :

1. Un nombre de gouttes dépendant du profil :
   `drizzle` = h/4, `default` = h/3, `heavy` = h, `torrential` = 2h
   (h = hauteur de l'image).
2. Chaque goutte part d'un point aléatoire uniforme `(x1, y1)` et se termine
   en `(x2, y2) = (x1 + L·sin(θ), y1 + L·cos(θ))`, où `L` est la longueur de
   goutte et `θ` l'angle d'inclinaison tiré uniformément dans `slant_range`.
3. Les segments sont tracés avec `cv2.line` en couleur claire
   `(200, 200, 200)` et largeur `drop_width`.

Les profils « notebook » retenus sont restreints par la configuration aux
trois plus intenses : `default`, `heavy`, `torrential`.

### 3.2 Paramètres aléatoires

| Paramètre | Plage (config) | Rôle |
|---|---|---|
| `rain_type` | {default, heavy, torrential} | nombre de gouttes (h/3, h, 2h) |
| `drop_length` | [10, 50] px | longueur des stries |
| `slant` | [−20°, +20°] | inclinaison des stries |
| `drop_width` | 1 (fixe) | épaisseur des stries |
| `drop_color` | (200, 200, 200) (défaut) | couleur des stries |

---

## 4. Faible luminosité — gamma adaptatif + bruit de photons + désaturation

`src/degradation/lowlight.py` — `LowlightGenerator` (albumentations 2.0.8)

### 4.1 Correction gamma (`A.RandomGamma`)

- Un gamma est tiré uniformément dans `[200, 320]` (convention
  albumentations : `γ = valeur/100`, soit **γ ∈ [2.0, 3.2]**, γ > 1
  assombrit), puis appliqué de façon déterministe (`gamma_limit = (g, g)`).
- **Gamma adaptatif** (`adapt_gamma`, `lowlight.py:17-43`, activé par
  `adaptive_gamma: true`) :
  - luminance moyenne BT.601 de l'image comparée à `ref_luminance = 120`
    avec une puissance 0.5 : `γ' = γ · (luma/120)^0.5`, écrêté à
    **[1.3, 3.5]** ;
  - une image **claire** (luma > 120) est assombrie davantage (γ augmenté),
    une image **déjà sombre** reçoit un γ plus doux pour préserver les
    détails (comportement testé dans `tests/degradation/test_lowlight.py`).

### 4.2 Bruit de photons (`A.ShotNoise`)

- Bruit de Poisson (bruit de photons du capteur), échelle tirée
  uniformément dans `[0.0, 0.03]` — plage très faible dans la convention
  albumentations (0.1 = bruit faible), donc un bruit léger de nuit/sous-exposition.

### 4.3 Désaturation (`A.ColorJitter`)

- Saturation tirée uniformément dans `[0.4, 1.0]` (jamais de sursaturation),
  pour imiter la perte de couleur en faible éclairage.

### 4.4 Paramètres aléatoires

| Paramètre | Plage (config) | Rôle |
|---|---|---|
| `gamma` | [2.0, 3.2] (adapté → [1.3, 3.5]) | assombrissement |
| `shot_noise` | [0.0, 0.03] | bruit de photons |
| `saturation` | [0.4, 1.0] | désaturation |

Le générateur supporte aussi un mode alternatif
`RandomBrightnessContrast` (non utilisé par la configuration actuelle).

---

## 5. Sorties du pipeline

`python -m src.degradation.degradation_pipeline` lit
`configs/degradation.yaml`, parcourt les images de `data/splits/{train,val,test}`
et écrit pour chaque dégradation activée :

```
data/synthetic/{fog,rain,lowlight}/{split}/images/*.jpg   (image dégradée)
data/synthetic/{fog,rain,lowlight}/{split}/annotations/*.txt (copie identique)
```

Les images dégradées sont ensuite mélangées aléatoirement à
l'entraînement des modèles de référence (ex. MoCE-IR) via
`MixedDegradationDataset`, avec les images claires comme cibles.

---

## 6. Références

- Modèle de Koschmieder : Koschmieder, H. (1924), *Theorie der horizontalen
  Sichtweite*.
- Estimation de profondeur : Yang et al., *Depth Anything V2*
  (`depth-anything/Depth-Anything-V2-Small-hf` sur Hugging Face).
- Bibliothèque d'augmentations : albumentations **2.0.8**
  (`A.RandomRain`, `A.RandomGamma`, `A.ShotNoise`, `A.ColorJitter`).
- Code : `src/degradation/{fog,rain,lowlight,degradation_pipeline}.py`,
  `configs/degradation.yaml`, tests `tests/degradation/*`.
