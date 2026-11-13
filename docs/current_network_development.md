# Current Network Development

**Project:** DEP-RNet — Image Restoration Preserving Detection Evidence (VisDrone)
**Date:** 2026-09-17

This report describes the current state of the network development: the baseline
model (MoCE-IR), the degradation settings under test, the parts that are working,
the parts that are still running or incomplete, the technical problems encountered,
and the two proposed components (SOD Loss, DEP Branch) currently under
implementation.

Status labels used throughout: **Completed**, **Ongoing**, **Planned**.

---

## 1. Baseline model: MoCE-IR

### 1.1 Which version of MoCE-IR is being used

MoCE-IR (*Mixture of Contextual Experts for All-in-One Image Restoration*) is
re-implemented from the official codebase in `src/baselines/moce_ir.py`
(`MoCEIR` class, faithful port: encoder/decoder transformer, FFT-domain attention
experts, frequency embedding, sparse expert routing with importance + load-balancing
auxiliary losses). The repository does not use pretrained weights — the model is
trained from scratch on the synthetic VisDrone data (`weights_path: null` in
`configs/baselines.yaml`).

The configured instance (`configs/baselines.yaml → all_in_one.moce_ir`) is:

| Hyperparameter | Value |
|---|---|
| `dim` / `levels` | 48 / 4 |
| `heads` | [1, 2, 4, 8] |
| `num_blocks` (encoder) | [4, 6, 6, 8] |
| `num_dec_blocks` (decoder) | [2, 4, 4] |
| `ffn_expansion_factor` / `num_refinement_blocks` | 2 / 4 |
| `num_experts` / `rank` / `topk` | 4 / 2 (spread) / 1 |
| `LayerNorm_type` / `bias` | WithBias / false |
| Expert layer | FFTAttention (patch sizes 4→32, kernels 3→9) |

Measured complexity (CUDA, `outputs/eval_moce_ir_test.txt`):
**25,353,796 parameters · 92.26 GFLOPs @ 256×256 · 84.3 ms inference @ 256×256**.

### 1.2 Degradation settings currently being tested

`configs/degradation.yaml` (deterministic, `seed_base: 42`, per-image seed offsets;
no geometric transformation, so VisDrone bounding boxes remain valid):

| Degradation | Method | Random parameters |
|---|---|---|
| Fog | Koschmieder model + depth from Depth Anything V2 Small | strength U[0.50, 1.0], heterogeneity U[0.0, 0.35], blur σ 1.5 |
| Rain | Albumentations `RandomRain` (linear streaks) | profiles {default, heavy, torrential}, drop length U[10, 50] px, slant U[−20°, +20°], width 1 |
| Low light | Adaptive `RandomGamma` + `ShotNoise` + `ColorJitter` | γ U[2.0, 3.2] (adaptive, clipped [1.3, 3.5], ref. luminance 120), shot noise U[0.0, 0.03], saturation U[0.4, 1.0] |

- **Data coverage (Completed):** 6 728 daytime images — train 5 200 / val 533 /
  test 995, each degraded into the three conditions (`data/synthetic/{fog,rain,lowlight}`).
  1 901 real night/low-light images (`data/real_lowlight_test/`) are reserved for
  robustness testing only and never enter the main benchmark.
- **Training protocol:** `MixedDegradationDataset` draws inputs randomly from
  {fog, rain, lowlight} with the clear image as target. Loss = **L1** +
  **0.01 × MoCE-IR routing auxiliary loss** (importance + load balancing).
  Optimizer AdamW (lr 2e-4), cosine schedule, 5 warmup epochs, 100 epochs planned.
  (Perceptual loss is not wired yet — `configs/loss.yaml` exists but is not read
  by `scripts/train.py`.)

### 1.3 Which parts of the baseline code are working

| Component | Status | Notes |
|---|---|---|
| Day/night classification (SAM3) + manual correction | Completed | Produces frozen `data/splits/` |
| Degradation synthesis pipeline (fog, rain, lowlight) | Completed | 100 % coverage, reproducible, unit-tested |
| MoCE-IR implementation | Completed | `src/baselines/moce_ir.py`, model metrics measured |
| MoCE-IR training | Completed (partial) | First run stopped after **20/100 epochs** (2026-09-14, checkpoint `outputs/baselines/best_moce_ir.pth`) |
| MoCE-IR evaluation on test split | Completed | Preliminary results with the 20-epoch checkpoint (see below) |
| Restore-to-disk pipeline | Completed | `scripts/restore_test.py` → `data/restored/moce_ir/{fog,rain,lowlight}` (2 352 images) |
| Training harness | Completed | `src/training/trainer.py`: DataParallel multi-GPU, tqdm progress, wandb logging, per-epoch visual samples, checkpointing every 5 epochs |
| Detector interface (fixed reference) | Completed | Common interface + factory (`src/detection/`): YOLO & RT-DETR; 3 detectors trained: yolov8l, yolo26l, rtdetr_l (`outputs/checkpoints/detector/`) |
| Restoration metrics | Completed | PSNR/SSIM/LPIPS (+ optional NIQE/BRISQUE), params/FLOPs/inference time (`src/evaluation/restoration_metrics.py`) |

**Preliminary baseline results** (20-epoch checkpoint, full test split,
2026-09-16, `outputs/eval_moce_ir_test.txt`):

| Degradation | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Fog | 16.42 | 0.776 | 0.159 |
| Rain | 31.52 | 0.939 | 0.048 |
| Low light | 23.66 | 0.767 | 0.181 |

*These numbers are preliminary: the checkpoint was interrupted at epoch 20 of the
100 planned.*

### 1.4 Tests still running or incomplete

- **Ongoing — MoCE-IR retraining:** a new run started 2026-09-17 (batch 8,
  100 epochs, same protocol); at the time of writing it is at **epoch 4/100**
  (ETA ≈ 25 h).
- **Ongoing — uncommitted evaluation work:** `scripts/evaluate_baseline.py`,
  `scripts/restore_test.py`, `src/evaluation/restore.py` and
  `src/evaluation/restoration_metrics.py` are modified/untracked in the working
  tree (not yet committed).
- **Incomplete — failing placeholder tests:** 6 tests still assert
  `NotImplementedError` while the implementation now exists and must be updated:
  `tests/baselines/test_moce_ir.py::test_restore_not_implemented`,
  `tests/training/test_trainer.py::{test_train_one_epoch_not_implemented,
  test_validate_not_implemented}`,
  `tests/evaluation/test_restoration_metrics.py::{test_psnr_not_implemented,
  test_ssim_not_implemented, test_lpips_not_implemented}`.
- **Incomplete — collection errors:** `tests/datasets/test_category_filter.py` and
  `tests/datasets/test_split_manager.py` import modules that do not exist yet
  (`src.datasets.category_filter`, `src.datasets.split_manager`).
- **Planned — unimplemented modules (stubs):** other baselines (AirNet, PromptIR,
  Restormer), task-specific baselines (dehaze/derain/lowlight), joint
  restoration+detection evaluation (`src/evaluation/joint_evaluation.py`),
  detection metrics `compute_detection_metrics` (`src/detection/metrics.py`),
  `src/datasets/visdrone_loader.py`, visualization modules.

### 1.5 Technical problems encountered

1. **CUDA out-of-memory at batch size 16.** A run launched on 2026-09-17 crashed
   inside `SparseDispatcher.combine` (`torch.OutOfMemoryError`, 23.56 GiB GPU
   exhausted). **Mitigation:** retry at batch size 8 (current run). Memory remains
   the main constraint: 20 epochs ≈ 5 h, so 100 epochs ≈ 25 h.
2. **Early wandb crashes at startup** (runs of 2026-09-14 lasting 2–9 s).
   **Resolved:** `setup_wandb()` in `scripts/train.py` now handles API keys from
   `configs/training.yaml` or a `.env` file.
3. **Degradation pipeline reproducibility bug** — fixed in commit `547e5c5`;
   parameter sampling switched from triangular to uniform distribution (commit
   `2c6f0af`).
4. **RT-DETR non-determinism on CUDA** (`grid_sample`) — requires
   `deterministic: false`, documented in `configs/detection.yaml` / README.
5. **Stale tests** (section 1.4) — placeholder tests were not updated after their
   modules were implemented.
6. Minor: `torch.load(weights_only=False)` FutureWarnings (MoCE-IR and LPIPS) —
   cosmetic, to be cleaned up later.

---

## 2. Proposed components (under implementation)

Both components below are **proposals under implementation**. Their intended
functions are described here, but **no performance improvement is claimed at this
stage** — all claims are deferred until the corresponding experiments are
completed.

### 2.1 Scale-aware Object Detail Loss (SOD Loss)

- **Intended purpose.** Strengthen the training supervision on small object
  regions — vehicles and pedestrians — so that restoration preserves the edges,
  contours and textures that degradations (fog, rain, low light) erase from small
  objects. The weighting is *scale-aware*: small regions receive boosted
  supervision (`configs/loss.yaml`: `small_object_scale_threshold: 32` px,
  `boost_factor: 2.0`, global weight `sod_loss: 0.5`).
- **Where it will be added.** `src/models/losses/sod_loss.py`
  (`ScaleAwareObjectDetailLoss`); it will be combined with the restoration losses
  (L1, and later perceptual) through the total-loss weighting defined in
  `configs/loss.yaml`.
- **Supervision.** The forward signature already declares `object_masks`:
  pixel-wise supervision on object regions, computed from the VisDrone
  annotations (boxes of vehicles/pedestrians), comparing the restored image to
  the clear target. The mask construction from annotations is not implemented yet.
- **Status: Planned.** Only the class skeleton exists;
  `forward()` raises `NotImplementedError`. The corresponding test
  (`tests/models/test_sod_loss.py`) is a placeholder that asserts the
  unimplemented state.

### 2.2 Lightweight Detection-Evidence Preservation Branch (DEP Branch)

- **Intended purpose.** A lightweight auxiliary branch whose role is to preserve
  *detection evidence* during restoration: object edges, local contrast and
  high-frequency details for vehicles and pedestrians — the cues a downstream
  detector relies on.
- **Where it will be added.** `src/models/dep_branch/dep_branch.py`
  (`DEPBranch`); it will be integrated into the DEP-RNet backbone
  (`src/models/restoration/dep_rnet.py`) as an auxiliary module, driven by
  `configs/model.yaml → dep_branch` (`enabled: true`, `channels: 32`,
  `num_layers: 3`).
- **Supervision.** Not yet specified in code. Candidates under consideration:
  high-frequency/edge targets derived from the clear image, object-mask-guided
  constraints (shared with the SOD Loss), or feature-level guidance from the fixed
  detector (`src/detection/detector_wrapper.py`). The exact scheme will be fixed
  before implementation.
- **Status: Planned.** Only the class skeleton exists (`TODO` in `__init__`;
  `forward()` raises `NotImplementedError`). The DEP-RNet backbone itself
  (`src/models/restoration/dep_rnet.py`) is also a skeleton (**Planned**), as is
  its test (`tests/models/test_dep_branch.py`).

---

## 3. Summary of status

| Item | Status |
|---|---|
| Degradation synthesis + splits + day/night filter | Completed |
| MoCE-IR implementation, training (20/100 epochs), preliminary evaluation | Completed (partial) |
| MoCE-IR retraining, 100 epochs, batch 8 | Ongoing (epoch ~4/100, ETA ≈ 25 h) |
| Test suite alignment (6 stale placeholder tests + 2 collection errors) | Ongoing |
| Joint restoration + detection evaluation (mAP on restored images) | Planned |
| Other baselines (AirNet, PromptIR, Restormer, task-specific) | Planned |
| **SOD Loss** | **Planned** (skeleton only) |
| **DEP Branch** | **Planned** (skeleton only) |
| DEP-RNet backbone | Planned (skeleton only) |

**Next steps:** finish the running MoCE-IR training → finalize the baseline
evaluation (restoration + detection metrics on the three degradations) → implement
and ablate the SOD Loss → implement and integrate the DEP Branch → full
DEP-RNet experiments.
