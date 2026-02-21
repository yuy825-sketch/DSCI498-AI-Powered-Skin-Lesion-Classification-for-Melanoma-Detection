# Results Report

This report summarizes the project setup, key experimental results, and the conclusions supported by the tracked artifacts in `results/`.

## Table of contents

- [What “targets” are achieved?](#what-targets-are-achieved)
- [Dataset and splits](#dataset-and-splits)
- [Methods](#methods)
- [Best runs (snapshot)](#best-runs-snapshot)
- [Experiments and results](#experiments-and-results)
- [Threshold-based melanoma detection (one-vs-rest)](#threshold-based-melanoma-detection-one-vs-rest)
- [Qualitative interpretability (Grad-CAM)](#qualitative-interpretability-grad-cam)
- [Dataset visualizations](#dataset-visualizations-for-reportpresentation)
- [Demo app](#demo-app)
- [Limitations and ethical note](#limitations-and-ethical-note)
- [Reproducibility](#reproducibility)

## What “targets” are achieved?

The milestone targets in `info.md` are **not strict KPIs**, but I tried to reach them anyway:

- **Test accuracy > 0.85**: achieved by the accuracy-focused EfficientNet-B2 @ 260px run.  
  See `summary_effnetb2_260_acc.md`.
- **Melanoma sensitivity (recall for `mel`) > 0.85**: achieved by the sensitivity-first run (top-1 recall), and also achievable by **thresholding `P(mel)`** (one-vs-rest) to pick a high-recall operating point.  
  See `summary_effnetb2_260_mel_sampler.md` and the threshold tables/plots.

Important nuance:
- These two targets are achieved by **different operating modes** (different runs / settings). A single model that simultaneously has both very high overall accuracy and very high top-1 melanoma recall is harder; that is why I also report the **threshold trade-off** (PR/threshold curves) as a more “medical-style” decision rule.

## Dataset and splits

- Dataset: HAM10000 (7 classes, `dx`) with metadata fields such as `age`, `sex`, and `localization`.
- Size: 10,015 images.
- Split strategy: lesion-wise grouped split (by `lesion_id`) into train/val/test with a fixed seed (see `train.py` + each run’s `split.json` under `runs/`, which is local-only).

Tracked dataset visualizations:
- Class imbalance: `dataset/class_distribution.png`
- Metadata overview: `dataset/metadata_stats.png`
- Sample thumbnails: `dataset/samples_grid.png`

## Methods

### CNN classifier

I train a CNN image classifier on HAM10000 (7 classes) using EfficientNet backbones. Metrics reported:
- accuracy
- macro-F1 (more informative under class imbalance)
- per-class recall (melanoma sensitivity = recall for `mel`)

### Handling class imbalance

I explore multiple strategies:
- class-weighted cross entropy (baseline)
- weighted sampling
- melanoma-weighted loss (explicitly prioritizing `mel`)

### Melanoma thresholding (one-vs-rest)

In addition to top-1 multiclass predictions, I treat `P(mel)` as a melanoma detection score and sweep thresholds to select an operating point (e.g., to enforce recall ≥ 0.85).

### Interpretability (Grad-CAM)

Grad-CAM overlays are used as qualitative explanations of model attention for selected examples.

### Optional generative augmentation (cVAE)

I include a conditional VAE to generate synthetic samples for an augmentation ablation. The tracked qualitative grid is `vae_samples_grid.png`.

## Best runs (snapshot)

### Accuracy-focused (best overall accuracy)

- Config: `configs/effnetb2_260_acc_select.json`
- Summary: `summary_effnetb2_260_acc.md`
- Confusion matrix: `confusion_matrix_effnetb2_260_acc.png`
- Key metrics (test):
  - Accuracy: **0.8614**
  - Macro-F1: **0.7386**
  - Melanoma recall (top-1): **0.5403**
- Takeaway: This run best matches the “>85% accuracy” style target, but its top-1 melanoma sensitivity is not high. If melanoma sensitivity is the priority, use a threshold on `P(mel)` (see below) or a sensitivity-first training setup.

### Sensitivity-first (best melanoma recall under top-1)

- Config: `configs/effnetb2_260_mel_sampler_select.json`
- Summary: `summary_effnetb2_260_mel_sampler.md`
- Confusion matrix: `confusion_matrix_effnetb2_260_mel_sampler.png`
- Key metrics (test):
  - Accuracy: **0.5374**
  - Macro-F1: **0.5666**
  - Melanoma recall (top-1): **0.8548**
- Takeaway: This run reaches the “>85% melanoma sensitivity” target under top-1 classification, but at a heavy cost: many non-melanoma samples are pushed toward melanoma (low precision and low overall accuracy). This is useful as a demonstration of a sensitivity-first bias, not as a balanced classifier.

## Experiments and results

This table is a compact index of the main tracked runs and where to find their artifacts (all values are from the corresponding `results/summary_*.md` files).

| Run | Test acc | Test macro-F1 | Mel recall | Key artifacts |
|---|---:|---:|---:|---|
| Baseline (EffNet-B0, class-weighted CE) | 0.7637 | 0.6650 | 0.7581 | `summary_baseline.md`, `confusion_matrix_baseline.png` |
| Tuned (EffNet-B2, mel-selected ckpt) | 0.7697 | 0.6958 | 0.7823 | `summary_effnetb2.md`, `confusion_matrix_effnetb2.png`, `mel_threshold_effnetb2.md` |
| Accuracy-focused (EffNet-B2@260, acc-selected ckpt) | **0.8614** | **0.7386** | 0.5403 | `summary_effnetb2_260_acc.md`, `confusion_matrix_effnetb2_260_acc.png`, `mel_threshold_effnetb2_260_acc.md` |
| Sensitivity-first (EffNet-B2@260 + sampler + mel-weight) | 0.5374 | 0.5666 | **0.8548** | `summary_effnetb2_260_mel_sampler.md`, `confusion_matrix_effnetb2_260_mel_sampler.png`, `mel_threshold_effnetb2_260_mel_sampler.md` |

Interpretation:
- “Accuracy-focused” provides the best overall classifier by accuracy and macro-F1 among the tracked runs.
- “Sensitivity-first” demonstrates melanoma recall > 0.85 under top-1 classification, at a large cost to overall accuracy and expected precision.
- Threshold analysis provides a clean way to pick a sensitivity-first operating point without forcing the whole multiclass classifier to behave like “always melanoma”.

## Threshold-based melanoma detection (one-vs-rest)

Instead of using the model’s top-1 class as the decision rule, I also treat `P(mel)` as a **detection score** and sweep thresholds.

Key artifacts:
- PR curves: `mel_pr_curve_effnetb2.png`, `mel_pr_curve_effnetb2_260_acc.png`, `mel_pr_curve_effnetb2_260_mel_sampler.png`
- Threshold sweep tables: `mel_threshold_effnetb2.md`, `mel_threshold_effnetb2_260_acc.md`, `mel_threshold_effnetb2_260_mel_sampler.md`
- Threshold trade-off plots: `mel_threshold_curve_effnetb2.png`, `mel_threshold_curve_effnetb2_260_acc.png`, `mel_threshold_curve_effnetb2_260_mel_sampler.png`

How to interpret:
- Lower threshold ⇒ higher melanoma recall (sensitivity) but lower precision (more false positives).
- This provides a clear “operating point” narrative for presentation/poster: *choose sensitivity first, accept precision cost*.

## Qualitative interpretability (Grad-CAM)

Grad-CAM overlays and per-image notes:
- Images: `results/gradcam/*.png`
- Per-image truth/pred/probability and conclusion: `results/gradcam/FIGURES.md`

Why it matters:
- It shows typical “attention” behavior for correct vs incorrect predictions.
- It also provides evidence for the main claim: high confidence does not guarantee correctness, and missed melanoma errors are the most critical to analyze.

## Visualizations count (tracked)

Currently tracked visualizations under `results/`:
- Total images: **30** (`.png`)
- Grad-CAM overlays: **5** (`results/gradcam/*.png`)

Full figure meanings and conclusions are indexed in:
- `FIGURES.md`
- `gradcam/FIGURES.md`

## Dataset visualizations (for report/presentation)

I added dataset-level plots and qualitative sample thumbnails for presentation use:
- Class distribution: `dataset/class_distribution.png`
- Metadata overview (age/sex/localization): `dataset/metadata_stats.png`
- Sample thumbnails (2 per class): `dataset/samples_grid.png`

Notes:
- HAM10000 is strongly imbalanced (dominant `nv`), which is why macro-F1 and melanoma-oriented metrics are necessary.
- Sample thumbnails are included as low-resolution examples for qualitative inspection and follow the dataset’s license/attribution (see dataset links in the repository `README.md`).
- These figures were generated by `python scripts/dataset_viz.py` (see `results/FIGURES.md` for exact commands).

## Demo app

The Streamlit demo app is implemented in `app/app.py` and supports:
- image upload
- top-k prediction display
- Grad-CAM overlay visualization

Tracked demo screenshot:
- `streamlit_demo.png`

This screenshot is generated in a local “demo mode” (so it can run without a manual upload during capture), but the app still supports normal user uploads during real use.

If you want to reproduce the screenshot locally, a helper script is provided:
- `bash scripts/capture_streamlit_screenshot.sh`

## Limitations and ethical note

- This is a course project and is **not** clinically validated.
- Metrics are reported on a single dataset split; generalization across populations/devices is not guaranteed.
- In a medical context, false negatives for melanoma are safety-critical; this is why I present threshold-based sensitivity trade-offs instead of only reporting overall accuracy.

## Reproducibility

Large artifacts are intentionally not committed:
- `data/` (dataset)
- `runs/` (full run directories, checkpoints, per-epoch history)

What is committed for reproducibility and grading:
- configs: `configs/*.json`
- tracked outputs: `results/` (summaries, plots, Grad-CAM examples, and this report)

To reproduce a run locally, use a config from `configs/` and run:
- `python train.py --config <config> --run-name <name>`
- then export tracked figures/tables via scripts in `scripts/` (see `results/FIGURES.md`).

## What to show in the final deliverable

For a course demo/report, the most defensible story is:
- Show the **high-accuracy model** as the “balanced baseline for general classification”
- Then show **melanoma threshold tuning** as the “sensitivity-first operating mode”
- Use Grad-CAM as qualitative interpretability evidence

The recommended “presentation bundle” is:
- `confusion_matrix_effnetb2_260_acc.png`
- `mel_threshold_curve_effnetb2.png` (or `*_260_acc.png`)
- one representative Grad-CAM overlay + its note entry
- `dataset/class_distribution.png`
- `dataset/samples_grid.png`
- `streamlit_demo.png`
