# DEP-RNet — Drone Image Restoration

Image restoration network for VisDrone under adverse weather conditions (fog, rain, low light) that preserves detection cues for vehicles and pedestrians. See `docs/suivi_projet.md` for the detailed plan.

## Project structure

```
dep-rnet/
├── configs/            # All modifiable values/parameters (YAML), nothing hardcoded in the code
├── data/               # Raw and generated data (not versioned, except splits/)
│   ├── raw/visdrone/           # Original VisDrone dataset (never modified)
│   ├── synthetic/{fog,rain,lowlight}/  # Synthetically degraded images
│   ├── splits/                 # Fixed train/val/test lists (versioned)
│   └── real_lowlight_test/     # Real night/low-light images — robustness testing only
├── src/
│   ├── datasets/       # VisDrone loading, category filtering, split management
│   ├── preprocessing/  # Day/night classification (SAM3) and other pre-training filters
│   ├── degradation/    # Fog synthesis (Koschmieder+fractal noise), rain, low light
│   ├── models/
│   │   ├── restoration/   # Main restoration network (DEP-RNet)
│   │   ├── dep_branch/    # Lightweight detection evidence preservation branch
│   │   └── losses/        # SOD loss (Scale-aware Object Detail) + classic losses
│   ├── detection/      # Fixed detector wrapper (YOLO) + mAP metrics computation
│   ├── training/       # Training loop, checkpointing
│   ├── evaluation/     # Restoration metrics (PSNR/SSIM/LPIPS) + joint evaluation
│   ├── baselines/      # Wrappers for AirNet, PromptIR, MoCE-IR, Restormer, etc.
│   ├── visualization/  # Figures: degradation examples, failure cases, visual comparisons
│   └── utils/          # Seeds, logging, path management
├── scripts/            # Executable entry points (train.py, evaluate_baseline.py, ...)
├── tests/              # Mirror of src/, one test file per module (pytest)
├── experiments/        # One subfolder per run, with frozen config + results
├── outputs/            # Final artifacts (checkpoints, exported figures)
├── logs/               # Raw training/evaluation logs
├── notebooks/          # Exploratory notebooks (preparatory work before integration into src/)
└── docs/               # Tracking document, decision log
```

## Basic rules

- **No hardcoded values in the code**: all modifiable parameters live in `configs/`.
- **`data/raw/visdrone/` is never modified**: degradations and filters generate new files elsewhere.
- **Train/val/test splits are fixed** once defined in `data/splits/`.
- **Each module in `src/` has its mirror in `tests/`.**
- Real night/low-light images are never included in the main benchmark — they are used only for robustness testing (see `docs/suivi_projet.md`).

## Installation

```bash
pip install -r requirements.txt
```

## Download a dataset

Datasets are defined in `configs/datasets.yaml`. The script recognizes Kaggle, Hugging Face, Google Drive URLs, and direct archive URLs:

```bash
python scripts/download_dataset.py
```

To change the dataset, only modify `dataset_name` in the `main` function.

## Run tests

```bash
pytest tests/
```

### Test the three degradations

The tests below use `data/degradation_test/original.jpg`, verify that the output keeps the same size and `uint8` type, then save an image for visual inspection.

```bash
pytest tests/degradation/test_fog.py -s
```

Generates `data/degradation_test/foggy.jpg`. The fog test loads Depth Anything the first time and may therefore be longer.

```bash
pytest tests/degradation/test_rain.py -s
```

Generates `data/degradation_test/rainy.jpg` with the `torrential` profile, a drop length of `20` and a seed `42`.

```bash
pytest tests/degradation/test_lowlight.py -s
```

Generates `data/degradation_test/lowlight.jpg` with brightness `-0.4`, contrast `-0.1`, saturation `0.8` and seed `42`.

To run all three tests:

```bash
pytest tests/degradation/test_fog.py tests/degradation/test_rain.py tests/degradation/test_lowlight.py -s
```

These are isolated image tests and use their own parameters. To test the values in `configs/degradation.yaml`, use the synthesis below.

## Run degradation synthesis

Day/night classification and synthesis are two separate steps. First run the classification:

```bash
export HF_TOKEN="hf_..."
python -m src.preprocessing.day_night_classifier
```

`facebook/sam3` is a protected Hugging Face model. You must first request and accept access on the model page, then provide a Hugging Face token with read role. The token can also be stored in a Kaggle secret named `HF_TOKEN`. It is never written to the configuration or the repository.

It copies daytime images to `data/splits/` and nighttime images to `data/real_lowlight_test/`. Then run the synthesis:

The classification uses batches of `8` images, according to `batch_size` in `configs/preprocessing.yaml`. Reduce this value if GPU memory is insufficient.

```bash
python -m src.degradation.degradation_pipeline
```

The synthesis reads `configs/degradation.yaml` and writes images and annotations to `data/synthetic/{fog,rain,lowlight}/`.
Modify the parameters in `configs/degradation.yaml` if necessary. But the values present there are already optimal.

## Tune low light interactively

Launch the Gradio interface with the preloaded test image (a free port is chosen automatically):

```bash
python scripts/lowlight_ui.py
```

The sliders update the preview instantly. To force port `7860`, use `python scripts/lowlight_ui.py --port 7860`. For access from another machine, add `--host 0.0.0.0`.