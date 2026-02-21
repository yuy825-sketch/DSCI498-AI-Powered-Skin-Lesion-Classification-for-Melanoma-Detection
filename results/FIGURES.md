# Figures and Visualizations (Index)

This file documents each visualization in `results/`, what it shows, and the conclusion it supports.

## Confusion matrices

### `confusion_matrix_baseline.png`
- **What**: 7-class confusion matrix on the test split for the baseline CNN.
- **How generated**: `python scripts/export_results.py --run-dir runs/<baseline_run> --out-dir results --tag baseline`
- **Conclusion**: Baseline has the strongest melanoma recall among the quick ablations (see `summary_baseline.md`).

### `confusion_matrix_sampler.png`
- **What**: Confusion matrix for the weighted-sampler ablation.
- **Conclusion**: Overall accuracy improved in this run, but melanoma recall did not improve compared to baseline.

### `confusion_matrix_melweight3.png`
- **What**: Confusion matrix for the melanoma-weighted loss ablation.
- **Conclusion**: Shifts recall distribution across classes; melanoma recall remains below baseline in this run.

### `confusion_matrix_synth.png`
- **What**: Confusion matrix for the “cVAE synthetic augmentation” ablation.
- **Conclusion**: Macro-F1 improved in this run, but melanoma recall decreased vs baseline (trade-off).

### `confusion_matrix_effnetb2.png`
- **What**: Confusion matrix for the tuned EfficientNet-B2 run.
- **Conclusion**: This run achieved the best macro-F1 among the committed experiments and slightly improved melanoma recall vs the baseline.

## Metric summaries

Each `summary_*.md` records:
- test accuracy
- test macro-F1
- test per-class recall (including melanoma sensitivity = recall for `mel`)

## Generative model visualization

### `vae_samples_grid.png`
- **What**: A small grid of cVAE-generated synthetic samples (first few per class).
- **How generated**: `python scripts/vae_sample_grid.py --synth-root artifacts/synth --out results/vae_samples_grid.png`
- **Conclusion**: The cVAE produces plausible-looking low-resolution lesion-like images (qualitative), enabling a synthetic augmentation ablation.

## Interpretability visualizations

See `gradcam/README.md` and `gradcam/FIGURES.md` for the Grad-CAM gallery and per-image notes.

## Training dynamics

### `training_curves_effnetb2.png`
- **What**: Train loss + validation accuracy/macro-F1/melanoma recall across epochs.
- **How generated**: `python scripts/plot_training.py --run-dir runs/<effnetb2_run> --out results/training_curves_effnetb2.png`
- **Conclusion**: Shows training stability and how validation melanoma recall evolves (checkpoint selection uses val melanoma recall for some configs).

## Melanoma detection operating point (one-vs-rest)

### `mel_pr_curve_effnetb2.png`
- **What**: Precision-Recall curve for melanoma (`mel`) one-vs-rest on the test split.
- **How generated**: `python scripts/plot_mel_pr.py --run-dir runs/<effnetb2_run> --out results/mel_pr_curve_effnetb2.png`
- **Conclusion**: Visualizes the sensitivity/precision trade-off when using `P(mel)` as a detection score.

### `mel_threshold_effnetb2.md`
- **What**: Threshold sweep table for melanoma one-vs-rest, including a suggested operating point achieving recall ≥ 0.85.
- **How generated**: `python scripts/mel_threshold_analysis.py --run-dir runs/<effnetb2_run> --out-md results/mel_threshold_effnetb2.md --min-recall 0.85`
- **Conclusion**: Demonstrates that melanoma sensitivity > 0.85 is achievable by lowering the detection threshold (with reduced precision).

### `mel_threshold_curve_effnetb2.png`
- **What**: Precision and recall vs threshold plot for melanoma one-vs-rest.
- **How generated**: `python scripts/plot_mel_threshold_curve.py --run-dir runs/<effnetb2_run> --out results/mel_threshold_curve_effnetb2.png --min-recall 0.85`
- **Conclusion**: Makes the threshold trade-off visually clear for presentation/poster use.

## Additional (exploratory) plots

These are included for completeness when comparing tuning attempts:
- `summary_effnetb0_long.md`, `confusion_matrix_effnetb0_long.png`, `training_curves_effnetb0_long.png`, `mel_pr_curve_effnetb0_long.png`, `mel_threshold_effnetb0_long.md`
