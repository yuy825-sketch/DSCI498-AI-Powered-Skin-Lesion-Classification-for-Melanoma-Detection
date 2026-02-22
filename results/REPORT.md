# Project Report — AI-Powered Skin Lesion Classification for Melanoma Detection (DSCI 498)

This report is written for course submission. It documents the dataset, methods, experiments, results, analysis, and a runnable demo app.

## Abstract

I built an end-to-end deep learning system to classify dermatoscopic skin lesion images from the **HAM10000** dataset (7 diagnostic categories). The project emphasizes both overall performance under severe class imbalance and **melanoma sensitivity** (recall for `mel`) as a safety-critical metric. I evaluate multiple imbalance-handling strategies, report confusion matrices and per-class recall, add threshold-based melanoma detection analysis using `P(mel)`, and provide qualitative interpretability via Grad-CAM. A Streamlit demo app demonstrates real-time image upload, prediction, and Grad-CAM visualization.

## Table of contents

- [1. Problem statement](#1-problem-statement)
- [2. Dataset](#2-dataset)
- [3. Methods](#3-methods)
- [4. Experiments](#4-experiments)
- [5. Results and analysis](#5-results-and-analysis)
- [6. Interpretability (Grad-CAM)](#6-interpretability-grad-cam)
- [7. Demo app (Streamlit)](#7-demo-app-streamlit)
- [8. Limitations and ethical note](#8-limitations-and-ethical-note)
- [9. Reproducibility](#9-reproducibility)
- [10. Conclusion](#10-conclusion)
- [11. References](#11-references)

## 1. Problem statement

Skin lesion classification is a high-impact application of computer vision. The key challenge in this dataset is **class imbalance** (a large majority of benign nevi) and visually overlapping lesion appearance. For melanoma detection, false negatives are particularly concerning, so I explicitly track melanoma sensitivity and also analyze threshold-based operating points using the model’s `P(mel)` output.


## 2. Dataset

I use **HAM10000** (Human Against Machine with 10,000 training images), a widely used dataset of **dermatoscopic skin lesion images** with diagnostic labels and basic patient/lesion metadata.

Dataset sources:
- Harvard Dataverse (DOI): `doi:10.7910/DVN/DBW86T`
- Kaggle mirror: “Skin Cancer MNIST: HAM10000”

HAM10000 images are dermoscopic photographs captured under clinical imaging setups. The dataset aggregates multiple sources and label acquisition methods, which makes it more diverse than many single-source collections, but also introduces potential dataset shift and label noise considerations [1].

### 2.1 Labels (`dx`) and their meaning

The classification target is the `dx` column (7 classes). I use the following label codes and human-readable names:

| `dx` code | Meaning |
|---|---|
| `akiec` | Actinic keratoses |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic nevi |
| `vasc` | Vascular lesions |

### 2.2 Metadata variables (columns)

The project relies on images for training, but the metadata is useful for dataset understanding and reporting. Key columns in the dataset metadata file include:

| Column | Meaning (high level) |
|---|---|
| `image_id` | Image identifier (used to locate the `.jpg` file) |
| `lesion_id` | Lesion identifier (multiple images can belong to the same lesion) |
| `dx` | Diagnostic label (target class) |
| `dx_type` | How the diagnosis was obtained (e.g., histopathology, follow-up, consensus, confocal) |
| `age` | Patient age (years; has a small amount of missingness) |
| `sex` | Patient sex |
| `localization` | Anatomical site |
| `dataset` | Source subset within HAM10000 |

### 2.3 Dataset statistics (class imbalance and sources)

Class counts (N=10,015):

| Class (`dx`) | Count | Fraction |
|---|---:|---:|
| `akiec` | 327 | 3.27% |
| `bcc` | 514 | 5.13% |
| `bkl` | 1099 | 10.97% |
| `df` | 115 | 1.15% |
| `mel` | 1113 | 11.11% |
| `nv` | 6705 | 66.95% |
| `vasc` | 142 | 1.42% |

Additional dataset descriptors (from metadata):
- `dx_type` distribution: `histo` (5340), `follow_up` (3704), `consensus` (902), `confocal` (69)
- `dataset` subsets: `vidir_molemax` (3954), `vidir_modern` (3363), `rosendahl` (2259), `vienna_dias` (439)
- Missingness: `age` missing rate is ~0.57%; `sex`/`localization` have no missing values in this CSV.

<img src="dataset/class_distribution.png" width="520" alt="HAM10000 class distribution">

*Figure 1. Class distribution of HAM10000 (`dx`). The dataset is highly imbalanced (dominant `nv`), motivating macro-F1 and melanoma-focused evaluation in addition to accuracy.*

<img src="dataset/metadata_stats.png" width="650" alt="HAM10000 metadata overview">

*Figure 2. Metadata overview (age, sex, and top-10 localizations). These covariates can contribute to dataset bias and should be considered when interpreting results.*

### 2.4 Sample examples

![HAM10000 sample strip](dataset/samples_strip.png)

*Figure 3. One random example per class (qualitative). Visual overlap across classes explains why misclassifications occur and why thresholding can be useful for sensitivity-first operation.*

### 2.5 Split strategy (leakage-aware)

HAM10000 contains multiple images per lesion (`lesion_id`). To reduce train/test leakage, I use a **lesion-wise grouped split** (grouped by `lesion_id`) into train/val/test with a fixed seed.

## 3. Methods

### 3.1 CNN classifier

I train a CNN image classifier using EfficientNet backbones [2] with ImageNet-style normalization. Given an input image \(x\), the network produces logits \(z \in \mathbb{R}^K\) for \(K=7\) classes, and class probabilities are obtained via softmax:

\[
p_k = \mathrm{softmax}(z)_k = \frac{\exp(z_k)}{\sum_{j=1}^{K}\exp(z_j)}.
\]

I use transfer learning (ImageNet-pretrained backbone) and fine-tune on HAM10000. EfficientNet provides a strong accuracy/efficiency trade-off via compound scaling of depth/width/resolution [2].

### 3.2 Handling class imbalance

I evaluate multiple imbalance-handling strategies:
- **Class-weighted cross-entropy** (baseline)
- **Weighted sampling** (to rebalance minibatches)
- **Melanoma-weighted loss** (explicitly prioritizing `mel` recall)

For class-weighted cross-entropy, the loss for a sample with true label \(y\) is:

\[
\mathcal{L}_{\mathrm{WCE}} = - w_y \log(p_y),
\]

where \(w_y\) is a class weight (typically larger for minority classes). When prioritizing melanoma sensitivity, I additionally upweight melanoma by a multiplier \(m>1\) (i.e., \(w_{\text{mel}} \leftarrow m \cdot w_{\text{mel}}\)).

Weighted sampling approximates rebalanced training by drawing examples with probability inversely proportional to class frequency. This can increase minority-class exposure but may reduce calibration and overall accuracy if over-applied.

### 3.3 Threshold-based melanoma detection (one-vs-rest)

Beyond top-1 multiclass prediction, I treat `P(mel)` as a melanoma detection score and sweep thresholds to obtain precision/recall trade-offs and a suggested operating point (e.g., recall ≥ 0.85).

Formally, define the melanoma score \(s(x) = p_{\text{mel}}(x)\). For a threshold \(t\),
\[
\hat{y}_{\text{mel}}(x;t) = \mathbb{1}[s(x)\ge t].
\]
On a test set, precision and recall are:
\[
\mathrm{Precision}(t) = \frac{TP(t)}{TP(t)+FP(t)}, \quad
\mathrm{Recall}(t) = \frac{TP(t)}{TP(t)+FN(t)}.
\]

### 3.4 Interpretability (Grad-CAM)

Grad-CAM provides qualitative heatmaps showing image regions most influencing a predicted class [3]. Let \(A^k\) be the \(k\)-th feature map in the chosen convolutional layer and \(y^c\) be the logit for class \(c\). Grad-CAM computes weights:

\[
\alpha_k^c = \frac{1}{Z}\sum_{i}\sum_{j}\frac{\partial y^c}{\partial A_{ij}^{k}},
\]

and the class activation map:

\[
L_{\mathrm{Grad\text{-}CAM}}^c = \mathrm{ReLU}\left(\sum_k \alpha_k^c A^k\right).
\]

This produces a coarse localization map highlighting regions that most increase the score for class \(c\).

### 3.5 Optional generative augmentation (cVAE)

I include a conditional variational autoencoder (cVAE) for a synthetic augmentation ablation. A VAE optimizes the evidence lower bound (ELBO):

\[
\mathcal{L}_{\mathrm{ELBO}} =
\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]
- \mathrm{KL}\left(q_\phi(z|x)\,\|\,p(z)\right),
\]

and the conditional version incorporates a class condition \(y\) (i.e., \(q_\phi(z|x,y)\), \(p_\theta(x|z,y)\)).

### 3.6 Preprocessing and augmentation

For all classifier runs, images are resized to a fixed square resolution (e.g., 224 or 260 pixels depending on the config), converted to tensors, and normalized using ImageNet mean/std. Training-time augmentation includes common geometric and color perturbations (e.g., random flips/rotation and light color jitter). This aims to reduce overfitting and improve robustness to minor appearance changes.

### 3.7 Optimization and checkpoint selection

I use AdamW optimization [4] with mixed precision (AMP) enabled for GPU training. A “best checkpoint” is selected based on a validation metric, such as:
- validation accuracy (accuracy-focused run)
- validation melanoma recall (sensitivity-focused runs)

### 3.8 Evaluation metrics (what I report and why)

Because HAM10000 is imbalanced, I report multiple metrics:
- **Accuracy**: overall fraction correct; can be inflated by the dominant `nv` class.
- **Macro-F1**: unweighted average F1 across classes; more informative for minority classes.
- **Per-class recall**: for each class, recall = TP / (TP + FN). In particular, **melanoma sensitivity** is recall for `mel`.

For a class \(c\), precision and recall are:
\[
\mathrm{Precision}_c = \frac{TP_c}{TP_c+FP_c}, \quad
\mathrm{Recall}_c = \frac{TP_c}{TP_c+FN_c},
\]
and the class F1 score is:
\[
F1_c = \frac{2\,\mathrm{Precision}_c\,\mathrm{Recall}_c}{\mathrm{Precision}_c+\mathrm{Recall}_c}.
\]
Macro-F1 is the mean over classes:
\[
\mathrm{MacroF1} = \frac{1}{K}\sum_{c=1}^{K} F1_c.
\]

## 4. Experiments

I report:
- test accuracy
- test macro-F1
- test per-class recall (including melanoma sensitivity = recall for `mel`)

All figures and quantitative results needed for this submission are included directly in this report.

### 4.1 Experiment set (ablations and tuning)

I ran a small set of experiments designed to answer:
1) How much do imbalance-handling strategies affect melanoma recall?
2) Can I reach high overall accuracy while still documenting a high-sensitivity operating mode?

The tracked experiments include:
- **Baseline**: EfficientNet-B0 with class-weighted cross-entropy.
- **Weighted sampler**: rebalance minibatches using a weighted sampler.
- **Melanoma-weighted loss**: increase the effective weight of melanoma to encourage sensitivity.
- **cVAE synthetic augmentation**: add generated samples for a controlled augmentation ablation.
- **Backbone tuning**: EfficientNet-B2 (including a higher-resolution 260px setup) with different checkpoint selection criteria (accuracy vs melanoma recall).

## 5. Results and analysis

### 5.1 Evaluation objectives

Given the dataset’s imbalance and the safety-critical nature of melanoma, I structure the evaluation around three objectives:
1) **Strong overall performance** (accuracy and macro-F1) under class imbalance
2) **High melanoma sensitivity** (melanoma recall), even if it requires a different operating point
3) **Interpretability + demo** for a complete end-to-end deliverable

As a result, I present both (i) an accuracy-focused multiclass classifier and (ii) sensitivity-first operating modes via thresholding or targeted training.

### 5.2 Summary table (key tracked runs)

| Run | Test acc | Test macro-F1 | Mel recall (top-1) |
|---|---:|---:|---:|
| Baseline (EffNet-B0, class-weighted CE) | 0.7637 | 0.6650 | 0.7581 |
| Tuned (EffNet-B2, melanoma-aware selection) | 0.7697 | 0.6958 | 0.7823 |
| Accuracy-focused (EffNet-B2 @ 260px, accuracy selection) | **0.8614** | **0.7386** | 0.5403 |
| Sensitivity-first (EffNet-B2 @ 260px + sampler + melanoma weighting) | 0.5374 | 0.5666 | **0.8548** |

Key conclusions from the table:
- The **accuracy-focused** model achieves the best overall accuracy and macro-F1, which is expected given the dominant `nv` class, but it has low melanoma recall under top-1.
- The **sensitivity-first** model meets high melanoma recall under top-1, but overall accuracy degrades sharply, suggesting that forcing sensitivity via training alone can create an impractical classifier.
- The best presentation is therefore to combine a strong multiclass model with **threshold-based** sensitivity tuning for melanoma.

### 5.3 Accuracy-focused model (best overall accuracy)

<img src="confusion_matrix_effnetb2_260_acc.png" width="360" alt="Confusion matrix (accuracy-focused)">

*Figure 4. Confusion matrix for the accuracy-focused EfficientNet-B2@260 run. Accuracy is strong due to the dominant `nv` class, while minority classes (especially `mel`) remain challenging.*

Key observations from Figure 4:
- The model is very strong on the majority class `nv`, which boosts overall accuracy.
- Many melanoma (`mel`) samples are confused with visually similar pigmented lesion classes (e.g., `nv`, `bkl`), which reduces top-1 melanoma recall in this operating mode.

<img src="training_curves_effnetb2_260_acc.png" width="680" alt="Training curves (accuracy-focused)">

*Figure 5. Training dynamics for the accuracy-focused run. The curves show validation metrics across epochs and support checkpoint selection by validation accuracy.*

Analysis: This run achieves **>0.85 test accuracy** and the best macro-F1 among the tracked experiments, which makes it a good “general classifier” baseline for the submission. However, accuracy alone is not sufficient to assess safety for melanoma detection.

### 5.4 Melanoma sensitivity vs precision (threshold analysis)

Using `P(mel)` as a one-vs-rest detection score yields a sensitivity/precision trade-off:

<img src="mel_threshold_curve_effnetb2.png" width="680" alt="Melanoma threshold trade-off">

*Figure 6. Precision/recall vs threshold for melanoma detection (EffNet-B2). Lower thresholds increase sensitivity but reduce precision (more false positives).*

This analysis supports a “medical-style” narrative: select an operating point by a sensitivity requirement instead of relying only on top-1 multiclass predictions.

To make this concrete, the threshold sweep tables (e.g., `mel_threshold_effnetb2.md`, `mel_threshold_effnetb2_260_acc.md`) record operating points such as “recall ≥ 0.85” and the associated precision cost. This is the most interpretable way to present the trade-off in a course setting.

### 5.5 Sensitivity-first training (high melanoma recall under top-1)

<img src="confusion_matrix_effnetb2_260_mel_sampler.png" width="360" alt="Confusion matrix (sensitivity-first)">

*Figure 7. Confusion matrix for the sensitivity-first run (EffNet-B2@260 with sampler + melanoma-weighted loss). Melanoma recall exceeds 0.85 under top-1, but many non-melanoma samples are pulled toward `mel`, which hurts overall accuracy.*

Analysis: This run demonstrates that explicitly optimizing for melanoma sensitivity can meet a sensitivity target, but it can create an impractical classifier if not paired with a thresholding policy or additional calibration. The main value of this experiment is to illustrate the sensitivity/precision trade-off and why a single metric is not sufficient.

## 6. Interpretability (Grad-CAM)

Grad-CAM overlays provide qualitative explanations for individual predictions:

![Grad-CAM example](gradcam/ISIC_0025964_overlay.png)

*Figure 8. Example Grad-CAM overlay. Grad-CAM is qualitative; it highlights image regions that most influence the model’s prediction.*

Analysis note: Grad-CAM is qualitative. It is most useful for sanity-checking whether the model focuses on the lesion region and for illustrating failure modes (e.g., missed melanoma despite seemingly lesion-focused attention). In this particular example, the image is a melanoma case that the baseline model misclassified, illustrating why sensitivity-oriented evaluation is necessary.

## 7. Demo app (Streamlit)

The Streamlit demo app provides an interactive submission deliverable:
- Upload an image
- View top-k predictions
- View a Grad-CAM overlay

![Streamlit demo screenshot](streamlit_demo.png)

*Figure 9. Streamlit demo screenshot (upload → predict → Grad-CAM). This supports a live demo during grading/presentation.*

## 8. Limitations and ethical note

- This is a course project and is **not** clinically validated.
- Results are reported on a fixed split of HAM10000; generalization to other populations, acquisition devices, and clinical settings is not guaranteed.
- In medical applications, decision thresholds should be selected with domain constraints, cost-sensitive evaluation, and external validation.

## 9. Reproducibility

The code is provided in this repository. To reproduce results, download the dataset from the DOI above, use the same split strategy (lesion-wise grouped split), and train with the described objectives and metrics. I recommend recording random seeds and exporting the same plots/metrics reported in this document to ensure comparability across runs.

## 10. Conclusion

This project demonstrates an end-to-end deep learning workflow on HAM10000 with:
- strong **overall accuracy** achievable under imbalance (EffNet-B2@260 reaches >0.85 test accuracy)
- explicit analysis of **melanoma sensitivity** and the **precision/sensitivity trade-off** via thresholding `P(mel)`
- qualitative interpretability via Grad-CAM and a runnable Streamlit demo for interactive presentation

## 11. References

1. P. Tschandl, C. Rosendahl, and H. Kittler. “The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.” *Scientific Data*, 2018. DOI: 10.1038/sdata.2018.161.
2. M. Tan and Q. V. Le. “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.” *ICML*, 2019.
3. R. R. Selvaraju et al. “Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based Localization.” *ICCV*, 2017.
4. I. Loshchilov and F. Hutter. “Decoupled Weight Decay Regularization.” *ICLR*, 2019.
5. Z. Liu et al. “A ConvNet for the 2020s (ConvNeXt).” *CVPR*, 2022.
6. A. Esteva et al. “Dermatologist-level classification of skin cancer with deep neural networks.” *Nature*, 2017. DOI: 10.1038/nature21056.
7. H. Xu et al. “Transformer-aided skin cancer classification using VGG19-based feature encoding.” *Scientific Reports*, 2025. DOI: 10.1038/s41598-025-24081-w.
8. A. M. Kamal and M. A. Al-Ahmar. “A Novel Vision Transformer Model for Skin Cancer Classification.” *Neural Processing Letters*, 2023.
9. J.-Y. Choi, M.-J. Song, and Y.-J. Shin. “Enhancing Skin Lesion Classification Performance with the ABC Ensemble Model.” *Applied Sciences*, 2024. DOI: 10.3390/app142210294.
