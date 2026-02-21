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

