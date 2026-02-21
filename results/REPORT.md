# Results Report (Current)

This report summarizes the **current best results**, key conclusions, and how to interpret the trade-offs for this HAM10000 (7-class) skin lesion classification course project.

## What “targets” are achieved?

The milestone targets in `info.md` are **not strict KPIs**, but I tried to reach them anyway:

- **Test accuracy > 0.85**: achieved by the accuracy-focused EfficientNet-B2 @ 260px run.  
  See `summary_effnetb2_260_acc.md`.
- **Melanoma sensitivity (recall for `mel`) > 0.85**: achieved by the sensitivity-first run (top-1 recall), and also achievable by **thresholding `P(mel)`** (one-vs-rest) to pick a high-recall operating point.  
  See `summary_effnetb2_260_mel_sampler.md` and the threshold tables/plots.

Important nuance:
- These two targets are achieved by **different operating modes** (different runs / settings). A single model that simultaneously has both very high overall accuracy and very high top-1 melanoma recall is harder; that is why I also report the **threshold trade-off** (PR/threshold curves) as a more “medical-style” decision rule.

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
- Total images: **31** (`.png`)
- Grad-CAM overlays: **5** (`results/gradcam/*.png`)

Full figure meanings and conclusions are indexed in:
- `FIGURES.md`
- `gradcam/FIGURES.md`

## Dataset visualizations (for report/presentation)

I added dataset-level plots and qualitative sample thumbnails for presentation use:
- Dataset overview (recommended for README): `dataset/overview.png`
- Class distribution: `dataset/class_distribution.png`
- Metadata overview (age/sex/localization): `dataset/metadata_stats.png`
- Sample thumbnails (2 per class): `dataset/samples_grid.png`

Notes:
- HAM10000 is strongly imbalanced (dominant `nv`), which is why macro-F1 and melanoma-oriented metrics are necessary.
- Sample thumbnails are included as low-resolution examples for qualitative inspection and follow the dataset’s license/attribution (see dataset links in the repository `README.md`).

## Streamlit demo screenshot

To support the “webapp/demo” deliverable, I captured a demo screenshot:
- `streamlit_demo.png`

This screenshot is generated in a local “demo mode” (so it can run without a manual upload during capture), but the app still supports normal user uploads during real use.

## What to show in the final deliverable

For a course demo/report, the most defensible story is:
- Show the **high-accuracy model** as the “balanced baseline for general classification”
- Then show **melanoma threshold tuning** as the “sensitivity-first operating mode”
- Use Grad-CAM as qualitative interpretability evidence

The recommended “presentation bundle” is:
- `confusion_matrix_effnetb2_260_acc.png`
- `mel_threshold_curve_effnetb2.png` (or `*_260_acc.png`)
- one representative Grad-CAM overlay + its note entry
- `dataset/overview.png`
- `streamlit_demo.png`
