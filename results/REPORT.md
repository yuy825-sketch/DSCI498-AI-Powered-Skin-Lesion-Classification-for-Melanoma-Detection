# Project Report — AI-Powered Skin Lesion Classification for Melanoma Detection (DSCI 498)

This report is written for course submission. It documents the dataset, methods, experiments, results, analysis, and a runnable demo app.

## Abstract

This project implements an end-to-end deep learning system to classify dermatoscopic skin lesion images from the **HAM10000** dataset (7 diagnostic categories) [1]. The work emphasizes (i) strong overall multiclass performance under severe class imbalance and (ii) **melanoma sensitivity** (recall for `mel`) as a safety-critical metric. Multiple imbalance-handling strategies are evaluated (class-weighted loss, weighted sampling, melanoma upweighting, and a small synthetic augmentation ablation). Confusion matrices and per-class recall are reported, and melanoma detection operating points are analyzed by thresholding the model’s `P(mel)` output.

On the fixed lesion-wise split used throughout the report, the best accuracy-focused model (EfficientNet-B2 @ 260px) reaches **0.8614** test accuracy and **0.7386** macro-F1, while a sensitivity-first training setup reaches **0.8548** top-1 melanoma recall at the cost of overall accuracy. A practical sensitivity-first operating mode can also be obtained via thresholding: for an EfficientNet-B2 melanoma-aware model, selecting a threshold that enforces **recall ≥ 0.85** yields **precision ≈ 0.322** with **recall ≈ 0.855** on the test set. Qualitative interpretability via Grad-CAM [3] and a Streamlit demo app for interactive upload → prediction → heatmap visualization are included.

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

Skin lesion classification is a high-impact application of computer vision. The key challenge in this dataset is **class imbalance** (a large majority of benign nevi) and visually overlapping lesion appearance. For melanoma detection, false negatives are particularly concerning, so melanoma sensitivity is explicitly tracked and threshold-based operating points are analyzed using the model’s `P(mel)` output.

### 1.1 Context and brief related work

Deep learning has reached dermatologist-level performance on certain curated dermatology image classification settings [5], but performance depends on data curation, acquisition conditions, and evaluation protocol. Surveys and systematic reviews emphasize common pitfalls for medical imaging ML (imbalance, dataset shift, and limited external validation) [8, 10]. Dermatoscopic datasets are often imbalanced and multi-source [1], which motivates evaluation beyond accuracy (e.g., macro-F1 and per-class recall) and motivates threshold-based decision policies for safety-critical targets such as melanoma.

### 1.2 Submission deliverables (what is shown and graded)

This project is designed as a complete, demonstrable course submission:
- A trained multiclass classifier on HAM10000 with tracked runs and plots.
- Multiple imbalance-handling ablations and a clear analysis of trade-offs.
- Melanoma detection threshold analysis (precision/recall operating points).
- Qualitative interpretability via Grad-CAM.
- A Streamlit demo app with screenshots suitable for presentation.


## 2. Dataset

This project uses **HAM10000** (Human Against Machine with 10,000 training images), a widely used dataset of **dermatoscopic skin lesion images** with diagnostic labels and basic patient/lesion metadata.

Dataset sources:
- Harvard Dataverse (DOI): `doi:10.7910/DVN/DBW86T` [17]
- Kaggle mirror: “Skin Cancer MNIST: HAM10000” [16]

HAM10000 images are **dermoscopic photographs** captured under clinical imaging setups (magnified, standardized lighting). The dataset aggregates multiple sources and multiple label acquisition methods, which makes it more diverse than many single-source collections, but also introduces potential dataset shift and label uncertainty considerations [1]. In this project, images are the primary learning signal; metadata is used mainly for dataset understanding, visualization, and reporting rather than as a model input.

### 2.1 Labels (`dx`) and their meaning

The classification target is the `dx` column (7 classes). The following label codes and human-readable names are used:

| `dx` code | Meaning |
|---|---|
| `akiec` | Actinic keratoses |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic nevi |
| `vasc` | Vascular lesions |

Clinically, this is a mix of malignant (`mel`, `bcc`) and benign or benign-like categories (`nv`, `bkl`, `df`, `vasc`) as well as pre-malignant lesions (`akiec`). Because several classes can present as pigmented lesions with similar color/texture patterns under dermoscopy, misclassifications are expected, especially between `mel`, `nv`, and `bkl`.

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

The `dx_type` field is particularly important for interpretation because it reflects label provenance:
- `histo`: diagnosis via histopathology (biopsy)
- `follow_up`: diagnosis via follow-up (stable benign appearance over time)
- `consensus`: expert consensus
- `confocal`: confocal microscopy

This heterogeneity is a realistic aspect of medical datasets, but it also means that “label quality” is not uniform across all samples, which can affect achievable performance and error interpretation.

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

<img src="dataset/class_distribution.png" width="440" alt="HAM10000 class distribution">

*Figure 1. Class distribution of HAM10000 (`dx`). The dataset is highly imbalanced (dominant `nv`), motivating macro-F1 and melanoma-focused evaluation in addition to accuracy.*

<img src="dataset/metadata_stats.png" width="520" alt="HAM10000 metadata overview">

*Figure 2. Metadata overview (age, sex, and top-10 localizations). These covariates can contribute to dataset bias and should be considered when interpreting results.*

### 2.4 Sample examples

![HAM10000 sample strip](dataset/samples_strip.png)

*Figure 3. One random example per class (qualitative). Visual overlap across classes explains why misclassifications occur and why thresholding can be useful for sensitivity-first operation.*

### 2.5 Split strategy (leakage-aware)

HAM10000 contains multiple images per lesion (`lesion_id`). To reduce train/test leakage, a **lesion-wise grouped split** (grouped by `lesion_id`) is used to form train/val/test with a fixed seed.

On my fixed split, the image counts are:
- Train: 8012 images
- Validation: 1000 images
- Test: 1003 images

This split strategy is important because if images from the same lesion appear in both train and test, accuracy can be artificially inflated (the model effectively sees near-duplicates of the same lesion).

### 2.6 Dataset challenges that affect modeling

Several dataset properties directly influence model design and evaluation:
- **Severe imbalance**: `nv` dominates (~67%), so accuracy can be misleading; macro-F1 and per-class recall are necessary.
- **Heterogeneous label acquisition** (`dx_type`): histopathology (`histo`) is generally the strongest ground truth, while follow-up and consensus labeling can introduce different uncertainty profiles [1].
- **Multi-source composition** (`dataset` column): multi-source data improves diversity but increases the risk of source-specific cues (device, lighting, annotation policy) that a model might overfit.
- **Risk asymmetry**: for melanoma detection, false negatives are costlier than false positives, motivating operating-point analysis instead of relying only on top-1 labels.

## 3. Methods

### 3.1 CNN classifier

A CNN image classifier is trained using EfficientNet backbones [2] with ImageNet-style normalization. Given an input image `x`, the network produces logits `z` for `K=7` classes, and class probabilities are obtained via softmax:

```math
p_k = \mathrm{softmax}(z)_k = \frac{\exp(z_k)}{\sum_{j=1}^{K}\exp(z_j)}.
```

The models use transfer learning (ImageNet-pretrained backbone) and are fine-tuned on HAM10000. EfficientNet provides a strong accuracy/efficiency trade-off via compound scaling of depth/width/resolution [2].

In implementation terms, the model consists of (i) an EfficientNet backbone, (ii) a dropout layer, and (iii) a linear classification head to produce 7 logits. During training, mixed precision (AMP) is enabled when a CUDA GPU is available, which speeds up training while keeping memory usage manageable.

### 3.2 Handling class imbalance

Several imbalance-handling strategies are evaluated:
- **Class-weighted cross-entropy** (baseline)
- **Weighted sampling** (to rebalance minibatches)
- **Melanoma-weighted loss** (explicitly prioritizing `mel` recall)

For class-weighted cross-entropy, the loss for a sample with true label `y` is:

```math
\mathcal{L}_{\mathrm{WCE}} = - w_y \log(p_y),
```

where `w_y` is a class weight (typically larger for minority classes). When prioritizing melanoma sensitivity, the melanoma weight is additionally upweighted by a multiplier `m>1` (i.e., `w_mel ← m · w_mel`).

In this project, the base class weights are computed from training-set counts `n_c` as:
```math
w_c \propto \frac{\sum_{j=1}^{K} n_j}{n_c + \epsilon},
```
and then normalized so that the mean weight is 1 (to keep the overall loss scale stable across runs).

Weighted sampling approximates rebalanced training by drawing examples with probability inversely proportional to class frequency. Intuitively, this changes the *effective* training distribution to reduce minibatch dominance by `nv`. If `pi_i` denotes the sampling probability for training example `i` with class `y_i`, then:
```math
\pi_i \propto w_{y_i}.
```

Two additional knobs that can interact with imbalance are also included:
- **Label smoothing** (used in the accuracy-focused EfficientNet-B2@260 run) can improve generalization and reduce overconfidence.
- **Focal loss** (available as an option) down-weights easy examples to focus learning on harder cases [6].
- **Class-balanced losses** (not used in my final runs) provide an alternative reweighting scheme based on the “effective number of samples” [7].

### 3.3 Threshold-based melanoma detection (one-vs-rest)

Beyond top-1 multiclass prediction, `P(mel)` is treated as a melanoma detection score and thresholds are swept to obtain precision/recall trade-offs and suggested operating points (e.g., recall ≥ 0.85).

Formally, define the melanoma score `s(x) = p_mel(x)`. For a threshold `t`,
```math
\hat{y}_{\text{mel}}(x;t) = \mathbf{1}[s(x)\ge t].
```
On a test set, precision and recall are:
```math
\mathrm{Precision}(t) = \frac{TP(t)}{TP(t)+FP(t)}, \quad
\mathrm{Recall}(t) = \frac{TP(t)}{TP(t)+FN(t)}.
```

This threshold analysis is intentionally decision-oriented: it makes explicit how the same model can be used under different objectives (high-accuracy multiclass classification vs high-sensitivity melanoma screening) without pretending a single top-1 metric captures clinical risk.

### 3.4 Interpretability (Grad-CAM)

Grad-CAM provides qualitative heatmaps showing image regions most influencing a predicted class [3]. Let `A^k` be the `k`-th feature map in the chosen convolutional layer and let `y^c` be the logit for class `c`. Grad-CAM computes weights:

```math
\alpha_k^c = \frac{1}{Z}\sum_{i}\sum_{j}\frac{\partial y^c}{\partial A_{ij}^{k}},
```

and the class activation map:

```math
L_{\mathrm{GradCAM}}^c = \mathrm{ReLU}\left(\sum_k \alpha_k^c A^k\right).
```

This produces a coarse localization map highlighting regions that most increase the score for class `c`.

### 3.5 Optional generative augmentation (cVAE)

A conditional variational autoencoder (cVAE) is included for a synthetic augmentation ablation. A VAE optimizes the evidence lower bound (ELBO):

```math
\mathcal{L}_{\mathrm{ELBO}} =
\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]
- \mathrm{KL}\left(q_\phi(z|x)\,\|\,p(z)\right),
```

and the conditional version incorporates a class condition `y` (i.e., `q_φ(z|x,y)`, `p_θ(x|z,y)`).

### 3.6 Preprocessing and augmentation

For all classifier runs, images are resized to a fixed square resolution (e.g., 224 or 260 pixels depending on the configuration), converted to tensors, and normalized using ImageNet mean/std.

Training-time augmentation includes:
- random horizontal/vertical flips
- random rotations (up to ~25 degrees)
- light color jitter (brightness/contrast/saturation)
- optional random resized crop (enabled in the 260px EfficientNet-B2 runs)

These augmentations aim to reduce overfitting and improve robustness to plausible acquisition variation (lighting, framing, and small rotations).

### 3.7 Optimization and checkpoint selection

AdamW optimization [4] is used with mixed precision (AMP) enabled for GPU training. A “best checkpoint” is selected based on a validation metric, such as:
- validation accuracy (accuracy-focused run)
- validation melanoma recall (sensitivity-focused runs)

This deliberate choice of selection metric is part of the experimental design: selecting by validation accuracy tends to favor the dominant class, while selecting by melanoma recall explicitly prioritizes sensitivity. In all cases, the final reported test metrics are computed using the selected best checkpoint.

### 3.8 Evaluation metrics (what is reported and why)

Because HAM10000 is imbalanced, multiple metrics are reported:
- **Accuracy**: overall fraction correct; can be inflated by the dominant `nv` class.
- **Macro-F1**: unweighted average F1 across classes; more informative for minority classes.
- **Per-class recall**: for each class, recall = TP / (TP + FN). In particular, **melanoma sensitivity** is recall for `mel`.

For a class `c`, precision and recall are:
```math
\mathrm{Precision}_c = \frac{TP_c}{TP_c+FP_c}, \quad
\mathrm{Recall}_c = \frac{TP_c}{TP_c+FN_c},
```
and the class F1 score is:
```math
F1_c = \frac{2\,\mathrm{Precision}_c\,\mathrm{Recall}_c}{\mathrm{Precision}_c+\mathrm{Recall}_c}.
```
Macro-F1 is the mean over classes:
```math
\mathrm{MacroF1} = \frac{1}{K}\sum_{c=1}^{K} F1_c.
```

## 4. Experiments

All experiments are evaluated on the same held-out test split (lesion-wise grouped). The evaluation reports:
- **test accuracy** (overall correctness)
- **test macro-F1** (imbalance-aware summary)
- **test per-class recall**, with emphasis on **melanoma sensitivity** (recall for `mel`)

For each tracked run, the following artifacts are produced and summarized in this report:
- confusion matrix (multiclass error structure)
- training curves (training dynamics and checkpoint selection behavior)
- melanoma threshold curve (precision/recall vs threshold using `P(mel)`)

The goal is not only to present a single “best score”, but to explain *why* certain objectives (overall accuracy vs melanoma recall) pull the model toward different operating points.

### 4.1 Experiment set (ablations and tuning)

A focused set of experiments is designed to answer:
1) How much do imbalance-handling strategies affect melanoma recall and macro-F1?
2) Can a strong overall classifier be obtained while also documenting a high-sensitivity screening operating point?

Experimental protocol (consistent across runs):
- The same lesion-wise grouped train/val/test split is used for all runs.
- Each run trains for a fixed number of epochs (typically 10–30 depending on the run) with AdamW and standard image augmentations described in Section 3.
- Model selection is performed on the validation split using a single chosen selection metric (validation accuracy or validation melanoma recall), and the selected checkpoint is evaluated once on the test split.

This design makes the comparisons interpretable: changes in test behavior can be attributed primarily to the imbalance strategy (sampler vs weighting), backbone capacity (EffNet-B0 vs EffNet-B2), input resolution (224 vs 260), and checkpoint selection criterion (accuracy vs melanoma recall).

The tracked experiments include:
- **Baseline**: EfficientNet-B0 with class-weighted cross-entropy.
- **Weighted sampler**: rebalance minibatches using a weighted sampler.
- **Melanoma-weighted loss**: increase the effective weight of melanoma to encourage sensitivity.
- **cVAE synthetic augmentation**: add generated samples for a controlled augmentation ablation.
- **Backbone tuning**: EfficientNet-B2 (including a higher-resolution 260px setup) with different checkpoint selection criteria (accuracy vs melanoma recall).

### 4.2 Key run configurations (what changed across runs)

To make the experiments easier to interpret, Table 1 summarizes the most important configuration differences (backbone, input size, and imbalance strategy). All runs use the same lesion-wise split and the same evaluation metrics.

| Run name (short) | Backbone | Image size | Imbalance strategy | Checkpoint selection |
|---|---|---:|---|---|
| Baseline | EffNet-B0 | 224 | class-weighted CE | default (val macro-F1) |
| Sampler | EffNet-B0 | 224 | weighted sampler | default (val macro-F1) |
| Mel-upweight (x3) | EffNet-B0 | 224 | class weights + `mel` multiplier | default (val macro-F1) |
| Baseline + synth | EffNet-B0 | 224 | class weights + synthetic images | default (val macro-F1) |
| Long mel-select | EffNet-B0 | 224 | class weights + `mel` multiplier | val melanoma recall |
| EffNet-B2 mel-select | EffNet-B2 | 224 | class weights + `mel` multiplier | val melanoma recall |
| EffNet-B2@260 acc-select | EffNet-B2 | 260 | label smoothing + stronger crop | val accuracy |
| EffNet-B2@260 mel-sampler | EffNet-B2 | 260 | class weights + sampler + `mel` multiplier | val melanoma recall |

## 5. Results and analysis

### 5.1 Evaluation objectives

Given the dataset’s imbalance and the safety-critical nature of melanoma, the evaluation is structured around three objectives:
1) **Strong overall performance** (accuracy and macro-F1) under class imbalance
2) **High melanoma sensitivity** (melanoma recall), even if it requires a different operating point
3) **Interpretability + demo** for a complete end-to-end deliverable

As a result, the report presents both (i) an accuracy-focused multiclass classifier and (ii) sensitivity-first operating modes via thresholding or targeted training.

These objectives are intentionally complementary rather than mutually exclusive. Accuracy and macro-F1 summarize general multiclass utility, but they do not fully describe melanoma risk. Conversely, maximizing melanoma recall in training can distort the multiclass decision boundary and produce too many false positives. The report therefore treats melanoma screening as an operating-point selection problem (Section 5.5), and uses interpretability/demonstration artifacts (Sections 6–7) to make the system understandable and presentable in a course setting.

### 5.2 Summary table (key tracked runs)

| Run | Test acc | Test macro-F1 | Mel recall (top-1) |
|---|---:|---:|---:|
| Baseline (EffNet-B0, class-weighted CE) | 0.7637 | 0.6650 | 0.7581 |
| Sampler (EffNet-B0, weighted sampler) | 0.8006 | 0.6756 | 0.7016 |
| Mel-upweight x3 (EffNet-B0) | 0.7597 | 0.6285 | 0.7419 |
| Baseline + synthetic augmentation (EffNet-B0) | 0.7926 | 0.6979 | 0.6694 |
| Long mel-select (EffNet-B0) | 0.7009 | 0.6218 | 0.7500 |
| Tuned (EffNet-B2, mel-select @224) | 0.7697 | 0.6958 | 0.7823 |
| Accuracy-focused (EffNet-B2 @ 260px, acc-select) | **0.8614** | **0.7386** | 0.5403 |
| Sensitivity-first (EffNet-B2 @ 260px, mel-sampler) | 0.5374 | 0.5666 | **0.8548** |

Key conclusions from the table:
- The **accuracy-focused** model achieves the best overall accuracy and macro-F1, but it has low top-1 melanoma recall. This is consistent with a model that strongly optimizes overall correctness and the majority class.
- Several imbalance interventions (sampler, synthetic augmentation) can improve macro-F1 relative to the baseline, but they do not automatically improve melanoma recall; improvements can move performance across classes rather than monotonically improving all objectives.
- The **sensitivity-first** model reaches the highest melanoma recall under top-1, but overall accuracy collapses, indicating that pushing sensitivity via training alone can over-predict melanoma and harm multiclass utility.
- A practically useful submission story is to pair a strong multiclass model with **threshold-based** sensitivity tuning for melanoma screening, making the precision/recall trade-off explicit.

### 5.3 Per-class recall breakdown (why the trade-off happens)

Table 3 shows per-class recall for three representative EfficientNet-B2 runs: an accuracy-focused model, a melanoma-aware model at 224px, and a sensitivity-first model. This clarifies how overall accuracy can coexist with weak melanoma recall, and how improving melanoma recall often reduces `nv` performance.

| Class | EffNet-B2@260 (acc-select) | EffNet-B2@224 (mel-select) | EffNet-B2@260 (mel-sampler) |
|---|---:|---:|---:|
| `akiec` | 0.605 | 0.632 | 0.763 |
| `bcc` | 0.769 | 0.789 | 0.789 |
| `bkl` | 0.768 | 0.642 | 0.663 |
| `df` | 0.727 | 0.727 | 0.636 |
| `mel` | 0.540 | 0.782 | 0.855 |
| `nv` | 0.964 | 0.794 | 0.421 |
| `vasc` | 0.538 | 0.692 | 0.846 |

Interpretation:
- The accuracy-focused run has very high `nv` recall (0.964), which drives accuracy upward, but melanoma recall is relatively low (0.540).
- The melanoma-aware run improves melanoma recall (0.782) while keeping `nv` recall reasonable (0.794), yielding a more balanced classifier.
- The sensitivity-first run increases melanoma recall to 0.855, but `nv` recall collapses to 0.421; this is consistent with a model that labels many benign nevi as melanoma.

### 5.4 Accuracy-focused model (best overall accuracy)

<img src="confusion_matrix_effnetb2_260_acc.png" width="300" alt="Confusion matrix (accuracy-focused)">

*Figure 4. Confusion matrix for the accuracy-focused EfficientNet-B2@260 run. Accuracy is strong due to the dominant `nv` class, while minority classes (especially `mel`) remain challenging.*

Key observations from Figure 4:
- The model is very strong on the majority class `nv`, which boosts overall accuracy.
- Many melanoma (`mel`) samples are confused with visually similar pigmented lesion classes (e.g., `nv`, `bkl`), which reduces top-1 melanoma recall in this operating mode.

<img src="training_curves_effnetb2_260_acc.png" width="520" alt="Training curves (accuracy-focused)">

*Figure 5. Training dynamics for the accuracy-focused run. The curves show validation metrics across epochs and support checkpoint selection by validation accuracy.*

Analysis: This run achieves **>0.85 test accuracy** and the best macro-F1 among the tracked experiments, which makes it a strong “general classifier” baseline for the submission. However, the confusion matrix and Table 3 show that this operating mode misses a substantial fraction of melanoma cases under top-1, motivating sensitivity-aware evaluation and threshold analysis.

### 5.5 Melanoma sensitivity vs precision (threshold analysis)

Using `P(mel)` as a one-vs-rest detection score yields a sensitivity/precision trade-off:

<img src="mel_threshold_curve_effnetb2.png" width="520" alt="Melanoma threshold trade-off">

*Figure 6. Precision/recall vs threshold for melanoma detection (EffNet-B2). Lower thresholds increase sensitivity but reduce precision (more false positives).*

This analysis supports a decision-oriented narrative: select an operating point by a sensitivity requirement (e.g., “recall ≥ 0.85”) instead of relying only on top-1 multiclass predictions.

To make this concrete, Table 4 lists operating points obtained by sweeping thresholds on the test set and choosing the **largest threshold** that still satisfies a recall constraint (which typically maximizes precision under that constraint).

| Model | Recall constraint | Selected threshold `t` | Precision | Recall |
|---|---:|---:|---:|---:|
| EffNet-B2@224 (mel-select) | ≥ 0.85 | 0.13 | 0.322 | 0.855 |
| EffNet-B2@224 (mel-select) | ≥ 0.90 | 0.06 | 0.299 | 0.911 |
| EffNet-B2@224 (mel-select) | ≥ 0.95 | 0.01 | 0.265 | 0.952 |
| EffNet-B2@260 (acc-select) | ≥ 0.85 | 0.01 | 0.133 | 0.992 |
| EffNet-B2@260 (mel-sampler) | ≥ 0.85 | 0.34 | 0.222 | 0.855 |
| EffNet-B2@260 (mel-sampler) | ≥ 0.90 | 0.21 | 0.199 | 0.911 |
| EffNet-B2@260 (mel-sampler) | ≥ 0.95 | 0.13 | 0.178 | 0.952 |

Interpretation:
- The accuracy-focused model achieves very high recall for melanoma at very low thresholds, but precision is extremely low (many false positives). This indicates that `P(mel)` is not well-separated for melanoma in that model, even though its top-1 accuracy is strong.
- The melanoma-aware EfficientNet-B2@224 provides a more favorable screening trade-off (higher precision at the same recall constraint).
- The sensitivity-first model shifts probability mass toward melanoma; it can satisfy high-recall constraints at higher thresholds, but precision remains limited due to increased false positives.

### 5.6 Sensitivity-first training (high melanoma recall under top-1)

<img src="confusion_matrix_effnetb2_260_mel_sampler.png" width="300" alt="Confusion matrix (sensitivity-first)">

*Figure 7. Confusion matrix for the sensitivity-first run (EffNet-B2@260 with sampler + melanoma-weighted loss). Melanoma recall exceeds 0.85 under top-1, but many non-melanoma samples are pulled toward `mel`, which hurts overall accuracy.*

Analysis: This run demonstrates that explicitly optimizing for melanoma sensitivity can meet a sensitivity target, but it can create an impractical classifier if not paired with a thresholding policy or additional calibration. In terms of course-report narrative, the main value of this experiment is to make the safety trade-off visible and to motivate why “best model” depends on the objective.

### 5.7 Error patterns and practical operating recommendation

Two recurring patterns explain most of the observed trade-offs:

1) **Melanoma is most frequently confused with `nv` and `bkl`.**  
In the accuracy-focused EfficientNet-B2@260 run, out of 124 melanoma test cases, 67 are correctly classified (recall 0.540), while 42 are predicted as `nv` and 12 as `bkl`. This is consistent with the visual overlap between pigmented lesions and helps explain why accuracy can remain high even when melanoma recall is low (because the majority class dominates).

2) **Sensitivity-first training increases false positives on `nv`.**  
In the melanoma-aware EfficientNet-B2@224 run, 112 `nv` images are predicted as `mel` under top-1. In the sensitivity-first EfficientNet-B2@260 run, this increases further (328 `nv` → `mel`), which explains the large drop in overall accuracy and the need to communicate precision/recall operating points.

Recommended presentation for a course submission:
- Use the **accuracy-focused EfficientNet-B2@260** run as the headline multiclass model (best overall metrics).
- Use the **melanoma-aware EfficientNet-B2@224** run for thresholded melanoma screening, because it offers a better precision/recall trade-off at recall constraints such as 0.85–0.95 (Table 4).

## 6. Interpretability (Grad-CAM)

Grad-CAM overlays provide qualitative explanations for individual predictions:

![Grad-CAM example](gradcam/ISIC_0025964_overlay.png)

*Figure 8. Example Grad-CAM overlay. Grad-CAM is qualitative; it highlights image regions that most influence the model’s prediction.*

Analysis note: Grad-CAM is qualitative. It is most useful for sanity-checking whether the model focuses on the lesion region and for illustrating failure modes (e.g., missed melanoma despite seemingly lesion-focused attention). In this particular example, the image is a melanoma case that the baseline model misclassified, illustrating why sensitivity-oriented evaluation is necessary.
Analysis note: Grad-CAM is qualitative. It is most useful for sanity-checking whether the model focuses on the lesion region and for illustrating failure modes (e.g., missed melanoma despite seemingly lesion-focused attention). It does not guarantee correctness and can sometimes highlight spurious correlations (e.g., surrounding skin texture or imaging artifacts).

In Figure 8, the ground-truth label is melanoma, but the baseline model predicts `vasc` with high confidence (top-1 probability ≈ 0.548) and assigns a relatively low melanoma probability (`P(mel) ≈ 0.074`). This is a clinically relevant failure mode: a confident false negative for melanoma. It reinforces two points already visible in the quantitative section: (i) top-1 accuracy can look acceptable under imbalance while still missing melanoma, and (ii) melanoma-oriented evaluation/thresholding is essential to communicate the sensitivity/precision trade-off.

## 7. Demo app (Streamlit)

The Streamlit demo app [15] provides an interactive submission deliverable:
- Upload an image
- View top-k predictions
- View a Grad-CAM overlay

From a course-grading perspective, the demo app serves two purposes:
1) It demonstrates that the model can be packaged into a usable interface (not just offline notebooks/scripts).
2) It helps communicate model uncertainty and failure modes (e.g., when `P(mel)` is non-trivial even for benign-appearing cases), especially when paired with the threshold analysis in Section 5.

The interface includes a sidebar for model/run selection and inference settings, a main-panel preview of the uploaded image, a probability table for the 7 classes, and an optional Grad-CAM overlay for the predicted class.

![Streamlit demo screenshot](streamlit_demo.png)

*Figure 9. Streamlit demo screenshot (upload → predict → Grad-CAM). This supports a live demo during grading/presentation.*

## 8. Limitations and ethical note

- This is a course project and is **not** clinically validated.
- Results are reported on a fixed split of HAM10000; generalization to other populations, acquisition devices, and clinical settings is not guaranteed.
- In medical applications, decision thresholds should be selected with domain constraints, cost-sensitive evaluation, and external validation.

Additional discussion points:
- **Label uncertainty**: different `dx_type` values correspond to different diagnostic processes; this can blur an achievable ceiling on performance for certain classes [1].
- **Population and device bias**: demographic representation and acquisition protocols can affect generalization; systematic reviews highlight the need for external validation and careful reporting for clinical deployment [10].
- **Explainability limitations**: Grad-CAM can be misleading if it highlights spurious regions; it should be used as a qualitative sanity check rather than a proof of correctness [3].
- **No clinical decision support claim**: this demo should not be used to guide diagnosis or treatment; it is intended purely as a course project artifact.

## 9. Reproducibility

The code is provided in this repository. To reproduce results end-to-end:
- Download HAM10000 from the DOI above (or the Kaggle mirror) and place it under the configured dataset root.
- Use the same lesion-wise grouped split (seeded) to avoid leakage across train/val/test.
- Train a model using one of the provided JSON configs (baseline, EfficientNet-B2 variants, sampler variants).
- Evaluate on the held-out test split and export the same plots reported here (confusion matrices, training curves, and melanoma threshold curves).

To make the report self-contained, this document focuses on the scientific/analytic narrative, while the repository code captures implementation details (configs, run metadata, and saved metrics).

## 10. Conclusion

This project demonstrates an end-to-end deep learning workflow on HAM10000 with:
- strong **overall accuracy** achievable under imbalance (EffNet-B2@260 reaches >0.85 test accuracy)
- explicit analysis of **melanoma sensitivity** and the **precision/sensitivity trade-off** via thresholding `P(mel)`
- qualitative interpretability via Grad-CAM and a runnable Streamlit demo for interactive presentation

Most importantly, the experiments show that “best model” depends on the objective: an accuracy-focused classifier can be strong overall while still missing melanoma under top-1, whereas sensitivity-first training can increase melanoma recall but may produce too many false positives. Presenting both multiclass metrics and melanoma operating points provides a clearer, more realistic summary of system behavior under class imbalance.

Future work for a more clinically oriented system would include external validation on independent datasets, explicit calibration (so that probabilities are more decision-ready), and evaluation across demographic and acquisition subgroups.

## 11. References

1. P. Tschandl, C. Rosendahl, and H. Kittler, “The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions,” *Scientific Data*, 2018. DOI: 10.1038/sdata.2018.161.
2. M. Tan and Q. V. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” *ICML*, 2019. arXiv:1905.11946.
3. R. R. Selvaraju et al., “Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,” *ICCV*, 2017. DOI: 10.1109/ICCV.2017.74.
4. I. Loshchilov and F. Hutter, “Decoupled Weight Decay Regularization,” *ICLR*, 2019. arXiv:1711.05101.
5. A. Esteva et al., “Dermatologist-level classification of skin cancer with deep neural networks,” *Nature*, 2017. DOI: 10.1038/nature21056.
6. T.-Y. Lin et al., “Focal Loss for Dense Object Detection,” *ICCV*, 2017. DOI: 10.1109/ICCV.2017.324.
7. Y. Cui et al., “Class-Balanced Loss Based on Effective Number of Samples,” *CVPR*, 2019. DOI: 10.1109/CVPR.2019.00949.
8. M. A. Khan et al., “Deep learning techniques for skin lesion analysis and melanoma cancer detection: a survey of state-of-the-art,” *Artificial Intelligence Review*, 2021. DOI: 10.1007/s10462-020-09865-y.
9. A. A. A. Elngar et al., “Deep learning approach of skin lesion classification from dermoscopy and clinical images along with patient clinical information,” *Physica Medica*, 2022. DOI: 10.1016/S1120-1797(22)03168-4.
10. F. M. B. Melo et al., “Diagnosis and prognosis of melanoma from dermoscopy images using machine learning and deep learning: a systematic review,” *BMC Cancer*, 2025. DOI: 10.1186/s12885-024-13423-y.
11. J.-Y. Choi, M.-J. Song, and Y.-J. Shin, “Enhancing Skin Lesion Classification Performance with the ABC Ensemble Model,” *Applied Sciences*, 2024. DOI: 10.3390/app142210294.
12. H. Xu et al., “Transformer-aided skin cancer classification using VGG19-based feature encoding,” *Scientific Reports*, 2025. DOI: 10.1038/s41598-025-24081-w.
13. “Improving skin lesion classification through saliency-guided loss functions,” *Computers in Biology and Medicine*, 2025. DOI: 10.1016/j.compbiomed.2025.110299.
14. A. Paszke et al., “PyTorch: An Imperative Style, High-Performance Deep Learning Library,” *NeurIPS*, 2019. arXiv:1912.01703.
15. Streamlit Documentation. https://docs.streamlit.io/
16. Kaggle Dataset: “Skin Cancer MNIST: HAM10000.” https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
17. Harvard Dataverse, “HAM10000: Human Against Machine with 10000 training images.” DOI: 10.7910/DVN/DBW86T.
