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
We use **HAM10000** (Human Against Machine with 10000 training images), a dermatoscopic image dataset with 7 diagnostic categories.

Recommended sources:
- Harvard Dataverse (DOI): `doi:10.7910/DVN/DBW86T`
- Kaggle mirror: `skin-cancer-mnist-ham10000`

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

## Train / Evaluate (planned)
The training scripts will save:
- checkpoints locally (not committed)
- run metadata + metrics under `runs/` (not committed)
- a small, presentation-ready results summary under `results/` (committed)

Example (after the dataset is available under `data/ham10000/`):
```
python train.py --config configs/baseline.json --run-name baseline-effnetb0
python scripts/export_latest.py
```

## Generative augmentation (VAE) (planned)
We include a simple **conditional VAE (cVAE)** that can generate class-conditioned synthetic samples.

Workflow (high level):
```
python train_vae.py --config configs/vae.json --run-name cvae64
python scripts/generate_synthetic.py --run-dir runs/<cvae_run_dir> --out-root artifacts/synth --per-class 200
python train.py --config configs/baseline_with_synth.json --run-name baseline+synthetic
```

## Demo app (planned)
The Streamlit app supports:
- uploading an image
- showing top-k predictions
- displaying a Grad-CAM overlay

Run it:
```
streamlit run app/app.py
```

## Results (to be filled)
We will report (at minimum):
- accuracy, macro-F1
- per-class recall
- melanoma sensitivity (recall for melanoma)
- confusion matrix and a small Grad-CAM gallery
