---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  :root {
    --ink: #0b0b0b;
    --ink2: #52514e;
    --blue: #2a78d6;
    --orange: #eb6834;
    --aqua: #1baf7a;
    --red: #e34948;
    --surface: #fcfcfb;
    --line: #c3c2b7;
  }
  section { font-family: "Segoe UI", "DejaVu Sans", sans-serif; color: var(--ink); background: var(--surface); padding: 48px 56px; }
  h1 { color: var(--blue); font-size: 1.9em; margin-bottom: 6px; }
  h2 { font-size: 1.15em; color: var(--ink2); margin-top: 0; }
  h3 { color: var(--ink); font-size: 1.05em; margin-bottom: 4px; }
  strong { color: var(--ink); }
  .badge { display: inline-block; font-size: 0.72em; font-weight: bold; padding: 1px 8px; border-radius: 10px; color: #fff; vertical-align: middle; }
  .b-highest { background: var(--orange); }
  .b-second  { background: var(--blue); }
  .b-opt     { background: #898781; }
  .b-planned { background: var(--aqua); }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
  .formula { font-family: "Cambria Math", "DejaVu Serif", serif; font-size: 1.15em; font-weight: bold; color: var(--blue); }
  .key { border: 2px solid var(--orange); border-radius: 10px; padding: 10px 16px; margin: 8px 0; }
  .key h3 { color: var(--orange); margin: 0 0 2px 0; }
  table { font-size: 0.82em; border-collapse: collapse; width: 100%; }
  th { background: #eef3fa; color: var(--ink); border-bottom: 2px solid var(--blue); }
  th, td { padding: 7px 10px; border-bottom: 1px solid var(--line); text-align: left; }
  .foot { position: absolute; bottom: 40px; left: 56px; right: 56px; font-size: 0.78em; color: var(--ink2); border-top: 1px solid var(--line); padding-top: 8px; }
  ul { margin: 4px 0 8px 0; }
  li { margin: 3px 0; font-size: 0.86em; }
  p { font-size: 0.9em; margin: 4px 0; }
---

# DEP-RNet — Mid-Term Presentation
## Restoration of degraded UAV images preserving detection evidence

**Task** — VisDrone UAV images degraded by **fog, rain, low light** are restored so that downstream
detection of **vehicles and pedestrians** keeps working. Degradations are synthesized from clear
images; bounding boxes remain valid (no geometric transforms).

**Baseline: MoCE-IR** (25.4 M params, from-scratch re-implementation, `src/baselines/moce_ir.py`)

| Data (synthetic VisDrone) | Preliminary test PSNR (20/100 epochs) |
|---|---|
| 6 728 day images × {fog, rain, lowlight} — train 5 200 / val 533 / test 995 | Fog **16.4 dB** · Rain **31.5 dB** · Low light **23.7 dB** |

<div class="key">

### The two key problems addressed by the proposed improvements

1. **Insufficient spatial information for small objects** — a 12×8 px pedestrian spans less than
   one feature position at 1/16 scale: downsampling weakens boundaries, background features dominate.
2. **Unreliable details caused by degradation** — strong high-frequency responses can be rain streaks
   or noise; weak edges can be real pedestrians under haze or low light.

</div>

<div class="foot">Retraining of MoCE-IR is in progress (100 epochs, batch 8, started 2026-09-17). All proposed designs below are <b>candidates — not yet validated improvements, not confirmed novel contributions</b>.</div>

---

# From Image Characteristics to Module Design

| Image characteristic | Main difficulty | Corresponding design | Priority |
|---|---|---|---|
| Small, densely distributed vehicles & pedestrians | Downsampling weakens boundaries; background features dominate | **Object-guided high-resolution residual refinement** | <span class="badge b-highest">Highest</span> |
| Spatially varying visibility | Some details preservable, others need contextual reconstruction | **Local reliability-guided detail–context fusion** | <span class="badge b-second">Second</span> |
| Thin, directional rain streaks | General high-frequency enhancement may amplify rain | Directional rain-residual suppression | <span class="badge b-opt">Optional</span> |
| Low light: weak contrast + noise | Brightness enhancement may amplify noise | Separate illumination adjustment & noise restoration | <span class="badge b-opt">Optional</span> |
| Spatially varying contrast attenuation (haze) | Local edge enhancement lacks scene context | Broad context with local transmission guidance | <span class="badge b-opt">Optional</span> |

- **Do not implement all five designs — prioritize the first two.**
- Local reliability gating (2nd row) starts as an **optional fusion strategy inside the refinement branch**.
- Contribution must concern **small-object-specific detail selection** — gating principles already exist
  (SNR-Aware LLIE, Retinexformer); degradation-aware MoE for UAV detection already exists (DAME-Net).

<div class="foot">Status labels: <span class="badge b-planned">Planned</span> / <span class="badge b-highest">Under implementation</span> — nothing here is claimed as an achieved improvement yet.</div>

---

# Recommended Architecture
## MoCE-IR backbone + Object-Guided High-Resolution Residual Refinement + SOD Loss

<div class="columns">
<div>

**Object-guided high-resolution residual refinement**
*(replaces the previous DEP Branch description)*

- **Reuse** MoCE-IR shallow features + final decoder features — **no new encoder**
- Project to ~**16 channels** → **2 lightweight depthwise-separable residual blocks**
- Predict a **correction ΔI** and a **soft object-region map M** (vehicle/pedestrian areas)
- <span class="formula">Î = Î<sub>baseline</sub> + M ⊙ ΔI</span>
- **Training:** M supervised by soft regions built from VisDrone boxes (coarse region supervision, not precise contours)
- **Inference:** M is **predicted by the network** — no ground-truth boxes
- **Risk control:** if M misses small objects under severe degradation, only the *refinement* is affected — the baseline output stays intact

</div>
<div>

**Refined SOD Loss** *(replaces box-wide uniform boosting)*

- **Per-object normalized** pixel loss: average within each object region before combining objects
- Smaller objects get **higher, bounded** weights
- **Gradient loss matched to the clean reference** — not "preserve every edge of the degraded input"
- Include **limited surrounding context** to reduce discontinuities at box boundaries

**Research route**

1. **Verify preprocessing first** — crop strategy, box transforms, object sizes after resizing (if pedestrians shrink to a few pixels, no module can fix it)
2. Implement refinement + SOD Loss on MoCE-IR
3. Ablate: with/without local reliability gating (optional), no directional rain module yet
4. Identify what small-object information the baseline loses, then test whether the refinement recovers it

</div>
</div>

<div class="foot"><b>All proposed components: Planned / Under Implementation — no performance claims until experiments are completed.</b></div>
