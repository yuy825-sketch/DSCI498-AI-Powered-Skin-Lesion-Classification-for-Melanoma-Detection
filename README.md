# AI-Powered Skin Lesion Classification for Melanoma Detection (DSCI 498)

This repository contains an end-to-end deep learning project for **skin lesion image classification** (HAM10000, 7 classes), with an emphasis on **melanoma sensitivity** (recall) and **model interpretability** (Grad-CAM), plus a simple **Streamlit demo app**.

## Project goals
- Train a strong CNN classifier on HAM10000
- Improve performance with augmentation and class-imbalance handling
- Add explainability (Grad-CAM) to visualize model attention
- Deliver a small Streamlit web app for image upload + predictions

## Disclaimer
This is a course project for educational purposes only. It is **not** medical advice and should not be used for clinical decision-making.

## Dataset
I use **HAM10000** (Human Against Machine with 10000 training images), a dermatoscopic image dataset with 7 diagnostic categories.

Recommended sources:
- Harvard Dataverse (DOI): `doi:10.7910/DVN/DBW86T` (link: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- Kaggle mirror: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

Expected local layout (not committed):
```
data/ham10000/
  HAM10000_metadata.csv
  ham10000_images_part_1/   # or a single images/ folder (either is fine)
  ham10000_images_part_2/
```

## Setup
1) Create an environment (conda/venv) and install dependencies:
```
pip install -r requirements.txt
pip install -e .
```

2) Run a quick smoke test (CPU):
```
python -m dsci498_skin.smoke
```

Note: On this machine, PyTorch may require:
```
export LD_PRELOAD=$CONDA_PREFIX/lib/libittnotify.so
```

## Train / Evaluate
The training scripts will save:
- checkpoints locally (not committed)
- run metadata + metrics under `runs/` (not committed)
- a small, presentation-ready results summary under `results/` (committed)

Example (after the dataset is available under `data/ham10000/`):
```
python train.py --config configs/baseline.json --run-name baseline-effnetb0
python scripts/export_latest.py
```

## Generative augmentation (VAE)
I include a simple **conditional VAE (cVAE)** that can generate class-conditioned synthetic samples.

Workflow (high level):
```
python train_vae.py --config configs/vae.json --run-name cvae64
python scripts/generate_synthetic.py --run-dir runs/<cvae_run_dir> --out-root artifacts/synth --per-class 200
python train.py --config configs/baseline_with_synth.json --run-name baseline+synthetic
```

## Demo app (Streamlit)
The Streamlit app supports:
- uploading an image
- showing top-k predictions
- displaying a Grad-CAM overlay

Run it:
```
streamlit run app/app.py
```

## Results
I report:
- accuracy, macro-F1
- per-class recall
- melanoma sensitivity (recall for `mel`)
- confusion matrices and a small Grad-CAM gallery

### Experiment snapshots

| Experiment | Test accuracy | Test macro-F1 | Melanoma sensitivity | Config | Outputs |
|---|---:|---:|---:|---|---|
| Baseline (EffNet-B0 + class-weighted CE, 10 epochs) | 0.7637 | 0.6650 | 0.7581 | [`configs/baseline.json`](configs/baseline.json) | `results/summary_baseline.md`, `results/confusion_matrix_baseline.png` |
| Imbalance sampler (EffNet-B0 + weighted sampler, 15 epochs) | 0.8006 | 0.6756 | 0.7016 | [`configs/imbalance_sampler.json`](configs/imbalance_sampler.json) | `results/summary_sampler.md`, `results/confusion_matrix_sampler.png` |
| Melanoma-weighted loss (EffNet-B0 + mel multiplier=3, 15 epochs) | 0.7597 | 0.6285 | 0.7419 | [`configs/mel_sensitive.json`](configs/mel_sensitive.json) | `results/summary_melweight3.md`, `results/confusion_matrix_melweight3.png` |
| cVAE synthetic augmentation (EffNet-B0 + extra synthetic images, 10 epochs) | 0.7926 | 0.6979 | 0.6694 | [`configs/baseline_with_synth.json`](configs/baseline_with_synth.json) | `results/summary_synth.md`, `results/confusion_matrix_synth.png`, `results/vae_samples_grid.png` |
| Tuned backbone (EffNet-B2, checkpoint selected by val melanoma recall) | 0.7697 | 0.6958 | 0.7823 | [`configs/effnetb2_mel_select.json`](configs/effnetb2_mel_select.json) | `results/summary_effnetb2.md`, `results/confusion_matrix_effnetb2.png`, `results/training_curves_effnetb2.png`, `results/mel_pr_curve_effnetb2.png`, `results/mel_threshold_effnetb2.md`, `results/mel_threshold_curve_effnetb2.png` |
| Accuracy-focused (EffNet-B2 @ 260px, checkpoint selected by val accuracy) | 0.8614 | 0.7386 | 0.5403 | [`configs/effnetb2_260_acc_select.json`](configs/effnetb2_260_acc_select.json) | `results/summary_effnetb2_260_acc.md`, `results/confusion_matrix_effnetb2_260_acc.png`, `results/training_curves_effnetb2_260_acc.png`, `results/mel_pr_curve_effnetb2_260_acc.png`, `results/mel_threshold_effnetb2_260_acc.md`, `results/mel_threshold_curve_effnetb2_260_acc.png` |
| Sensitivity-first (EffNet-B2 @ 260px + sampler + mel-weight, selected by val melanoma recall) | 0.5374 | 0.5666 | 0.8548 | [`configs/effnetb2_260_mel_sampler_select.json`](configs/effnetb2_260_mel_sampler_select.json) | `results/summary_effnetb2_260_mel_sampler.md`, `results/confusion_matrix_effnetb2_260_mel_sampler.png`, `results/training_curves_effnetb2_260_mel_sampler.png`, `results/mel_pr_curve_effnetb2_260_mel_sampler.png`, `results/mel_threshold_effnetb2_260_mel_sampler.md`, `results/mel_threshold_curve_effnetb2_260_mel_sampler.png` |

Grad-CAM examples: `results/gradcam/README.md`.

### Conclusions (current)
- The dataset is highly imbalanced, so **overall accuracy can be misleading**; I prioritize **melanoma sensitivity** and macro-F1.
- In my quick ablations, the **baseline class-weighted cross-entropy** achieved the best melanoma sensitivity among the early settings.
- I can reach **>85% test accuracy** with an EfficientNet-B2 @ 260px model (see `results/summary_effnetb2_260_acc.md`).
- I can also reach **>85% melanoma sensitivity** (top-1 recall for `mel`) with an explicitly sensitivity-first training setup, but accuracy drops sharply; this trade-off is important to show in a medical-style setting.
- Even without sensitivity-first training, I can achieve **melanoma sensitivity ≥ 0.85** by using **thresholding on `P(mel)`** (one-vs-rest) at a chosen operating point (see `results/mel_threshold_effnetb2.md` and related plots).

### Representative visualizations

High-accuracy confusion matrix (EffNet-B2 @ 260px):

![High-accuracy confusion matrix](results/confusion_matrix_effnetb2_260_acc.png)

Melanoma detection threshold trade-off (EffNet-B2, one-vs-rest):

![Melanoma threshold curve](results/mel_threshold_curve_effnetb2.png)

Example Grad-CAM overlay:

![Grad-CAM overlay example](results/gradcam/ISIC_0025964_overlay.png)

Full results and figure notes:
- Results folder: `results/README.md`
- Figure index: `results/FIGURES.md`
- Grad-CAM notes: `results/gradcam/FIGURES.md`

### Limitations
- This is a course project and not a clinically validated system.
- Performance may not generalize beyond HAM10000 (dataset bias, acquisition differences, labeling noise).
