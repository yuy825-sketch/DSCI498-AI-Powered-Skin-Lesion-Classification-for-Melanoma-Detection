# Grad-CAM Gallery Notes

This file documents Grad-CAM overlays in this folder.

For each image:
- **Truth** refers to the dataset label (`dx`) from `HAM10000_metadata.csv`.
- **Prediction** refers to the model’s top-1 class for that image.
- Grad-CAM is **qualitative**; it helps interpret what regions influence the prediction, but it does not guarantee correctness.

## Model used for these overlays

All overlays and the per-image probabilities below were generated with the baseline checkpoint:
- Run: `runs/20260220_192847__baseline-effnetb0-s1` (EfficientNet-B0, class-weighted CE)
- Script: `python scripts/gradcam_image.py --run-dir <run_dir> --image <path_to_image> --out results/gradcam/<image_id>_overlay.png`

## Images

- `ISIC_0025964_overlay.png`
  - Truth: `mel` (melanoma)
  - Prediction (top-1): `vasc` (0.5477); `P(mel)=0.0740`
  - Top-3: `vasc:0.5477`, `nv:0.3680`, `mel:0.0740`
  - Meaning: **false negative melanoma** example (melanoma missed under argmax).
  - Conclusion: Even when the model’s attention appears lesion-related, the classifier can still confidently mislabel melanoma; this motivates reporting **melanoma sensitivity** and using a **melanoma threshold** operating point in addition to top-1 accuracy.

- `ISIC_0030623_overlay.png`
  - Truth: `mel` (melanoma)
  - Prediction (top-1): `vasc` (0.8719); `P(mel)=0.1230`
  - Top-3: `vasc:0.8719`, `mel:0.1230`, `nv:0.0029`
  - Meaning: Another **false negative melanoma** with very high confidence for a wrong class.
  - Conclusion: This is a high-risk failure mode (missed melanoma). It supports the project focus on **sensitivity-first** evaluation and highlights that “high confidence” does not imply correctness.

- `ISIC_0024698_overlay.png`
  - Truth: `nv` (melanocytic nevi)
  - Prediction (top-1): `nv` (0.6598); `P(mel)=0.3281`
  - Top-3: `nv:0.6598`, `mel:0.3281`, `bkl:0.0111`
  - Meaning: Correct common-class reference example, but with a relatively high melanoma probability.
  - Conclusion: Some benign nevi can look melanoma-like to the model (high `P(mel)`), which may increase false positives when tuning for high melanoma sensitivity.

- `ISIC_0032212_overlay.png`
  - Truth: `nv` (melanocytic nevi)
  - Prediction (top-1): `nv` (1.0000); `P(mel)=0.0000`
  - Top-3: `nv:1.0000`, `vasc:0.0000`, `mel:0.0000`
  - Meaning: Very confident correct prediction example.
  - Conclusion: Serves as a sanity-check that Grad-CAM overlays can align with a stable decision when the model is confident.

- `ISIC_0028155_overlay.png`
  - Truth: `bcc` (basal cell carcinoma)
  - Prediction (top-1): `bcc` (0.9990); `P(mel)=0.0004`
  - Top-3: `bcc:0.9990`, `mel:0.0004`, `bkl:0.0003`
  - Meaning: Correct non-melanoma example for comparing attention patterns across lesion types.
  - Conclusion: Demonstrates that the model can strongly separate some non-melanoma classes, while melanoma remains the most safety-critical class requiring special handling.
