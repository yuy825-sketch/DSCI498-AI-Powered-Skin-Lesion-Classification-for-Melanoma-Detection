# Figures and Visualizations (Index)

This file documents each visualization in `results/`, what it shows, and the conclusion it supports.

## Confusion matrices

### `confusion_matrix.png` (latest export)
- **What**: The most recently exported confusion matrix without a tag.
- **Conclusion**: Use the tagged confusion matrices (below) for the specific experiments referenced in the README.

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

### `confusion_matrix_effnetb2_260_acc.png`
- **What**: Confusion matrix for the EfficientNet-B2 @ 260px run (checkpoint selected by validation accuracy).
- **How generated**: `python scripts/export_results.py --run-dir runs/<effnetb2_260_acc_run> --out-dir results --tag effnetb2_260_acc`
- **Conclusion**: Achieves the **highest overall accuracy** among the tracked runs (meets the >85% accuracy target), but melanoma recall is relatively low under top-1 classification.

### `confusion_matrix_effnetb2_260_mel_sampler.png`
- **What**: Confusion matrix for the EfficientNet-B2 @ 260px run with weighted sampling + melanoma-weighted loss (checkpoint selected by validation melanoma recall).
- **How generated**: `python scripts/export_results.py --run-dir runs/<effnetb2_260_mel_sampler_run> --out-dir results --tag effnetb2_260_mel_sampler`
- **Conclusion**: Achieves **melanoma sensitivity > 85%** under top-1 classification, but accuracy drops significantly due to an aggressive sensitivity-first bias (many non-melanoma samples are pushed toward melanoma).

## Metric summaries

Each `summary_*.md` (and `summary.md` as a “latest export”) records:
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

## Dataset visualizations

### `dataset/class_distribution.png`
- **What**: Bar chart of `dx` class counts with percentages.
- **How generated**: `python scripts/dataset_viz.py --root data/ham10000 --out-dir results/dataset --seed 42 --per-class 2`
- **Conclusion**: Confirms severe class imbalance (especially `nv` vs minority classes), motivating imbalance handling and melanoma-sensitivity analysis.

### `dataset/metadata_stats.png`
- **What**: 3-panel overview of metadata: age histogram, sex distribution, and top-10 localizations.
- **How generated**: `python scripts/dataset_viz.py --root data/ham10000 --out-dir results/dataset --seed 42 --per-class 2`
- **Conclusion**: Shows non-image covariate distributions that can contribute to dataset bias; results should be interpreted as dataset-specific.

### `dataset/samples_grid.png`
- **What**: Random sample thumbnails (2 per class) for qualitative inspection.
- **How generated**: `python scripts/dataset_viz.py --root data/ham10000 --out-dir results/dataset --seed 42 --per-class 2`
- **Conclusion**: Lesion appearances overlap visually across classes, explaining common confusions and the need for threshold-based sensitivity tuning.

### `dataset/samples_strip.png`
- **What**: One random example per class in a single row (compact “sample examples” figure).
- **How generated**: `python scripts/dataset_viz.py --root data/ham10000 --out-dir results/dataset --seed 42 --per-class 2`
- **Conclusion**: A compact qualitative overview suitable for the README (shows class variety without taking excessive vertical space).

## Demo app screenshot

### `streamlit_demo.png`
- **What**: Screenshot of the Streamlit demo page showing predictions + Grad-CAM (demo mode).
- **How generated**: `bash scripts/capture_streamlit_screenshot.sh` (requires a local run dir; see `app/app.py` demo env vars).
- **Conclusion**: Demonstrates a working “upload → predict → Grad-CAM” pipeline suitable for course deliverables.

## Training dynamics

### `training_curves_effnetb2.png`
- **What**: Train loss + validation accuracy/macro-F1/melanoma recall across epochs.
- **How generated**: `python scripts/plot_training.py --run-dir runs/<effnetb2_run> --out results/training_curves_effnetb2.png`
- **Conclusion**: Shows training stability and how validation melanoma recall evolves (checkpoint selection uses val melanoma recall for some configs).

### `training_curves_effnetb2_260_acc.png`
- **What**: Training curves for the EfficientNet-B2 @ 260px accuracy-selected run.
- **Conclusion**: Shows that validation accuracy can reach the mid/high-0.85 range during training, motivating the selected checkpoint for the final test evaluation.

### `training_curves_effnetb2_260_mel_sampler.png`
- **What**: Training curves for the EfficientNet-B2 @ 260px sensitivity-first (mel-sampler) run.
- **Conclusion**: Shows the dynamics and instability you can get when optimizing strongly for melanoma recall; this helps justify why the final project reports multiple operating modes/metrics instead of a single “best” number.

## Melanoma detection operating point (one-vs-rest)

### `mel_pr_curve_effnetb2.png`
- **What**: Precision-Recall curve for melanoma (`mel`) one-vs-rest on the test split.
- **How generated**: `python scripts/plot_mel_pr.py --run-dir runs/<effnetb2_run> --out results/mel_pr_curve_effnetb2.png`
- **Conclusion**: Visualizes the sensitivity/precision trade-off when using `P(mel)` as a detection score.

### `mel_pr_curve_effnetb2_260_acc.png`
- **What**: Melanoma PR curve for the EfficientNet-B2 @ 260px accuracy-selected run.
- **Conclusion**: Even when top-1 melanoma recall is low, `P(mel)` can still be used as a **detection score**; this curve visualizes that trade-off for threshold-based sensitivity tuning.

### `mel_pr_curve_effnetb2_260_mel_sampler.png`
- **What**: Melanoma PR curve for the EfficientNet-B2 @ 260px sensitivity-first run.
- **Conclusion**: Shows a high-recall regime with low precision, consistent with the model’s aggressive melanoma bias (useful for presenting a sensitivity-first operating mode).

### `mel_threshold_effnetb2.md`
- **What**: Threshold sweep table for melanoma one-vs-rest, including a suggested operating point achieving recall ≥ 0.85.
- **How generated**: `python scripts/mel_threshold_analysis.py --run-dir runs/<effnetb2_run> --out-md results/mel_threshold_effnetb2.md --min-recall 0.85`
- **Conclusion**: Demonstrates that melanoma sensitivity > 0.85 is achievable by lowering the detection threshold (with reduced precision).

### `mel_threshold_effnetb2_260_acc.md`
- **What**: Threshold sweep table for the EfficientNet-B2 @ 260px accuracy-selected run.
- **Conclusion**: Provides an explicit “high sensitivity” operating point (if available) and a full sweep table for reporting and visualization.

### `mel_threshold_effnetb2_260_mel_sampler.md`
- **What**: Threshold sweep table for the EfficientNet-B2 @ 260px sensitivity-first run.
- **Conclusion**: Quantifies the precision/sensitivity trade-off for a model that already achieves >85% melanoma recall under top-1, and helps pick a more reasonable operating point if desired.

### `mel_threshold_curve_effnetb2.png`
- **What**: Precision and recall vs threshold plot for melanoma one-vs-rest.
- **How generated**: `python scripts/plot_mel_threshold_curve.py --run-dir runs/<effnetb2_run> --out results/mel_threshold_curve_effnetb2.png --min-recall 0.85`
- **Conclusion**: Makes the threshold trade-off visually clear for presentation/poster use.

### `mel_threshold_curve_effnetb2_260_acc.png`
- **What**: Precision/recall vs threshold for melanoma detection (EfficientNet-B2 @ 260px accuracy-selected run).
- **Conclusion**: Visualizes how much sensitivity you can gain by lowering the melanoma threshold, and what precision you give up.

### `mel_threshold_curve_effnetb2_260_mel_sampler.png`
- **What**: Precision/recall vs threshold for melanoma detection (EfficientNet-B2 @ 260px sensitivity-first run).
- **Conclusion**: Visualizes a high-sensitivity operating region and highlights the expected precision cost.
## Additional (exploratory) plots

These are included for completeness when comparing tuning attempts:
- `summary_effnetb0_long.md`, `confusion_matrix_effnetb0_long.png`, `training_curves_effnetb0_long.png`, `mel_pr_curve_effnetb0_long.png`, `mel_threshold_effnetb0_long.md`
