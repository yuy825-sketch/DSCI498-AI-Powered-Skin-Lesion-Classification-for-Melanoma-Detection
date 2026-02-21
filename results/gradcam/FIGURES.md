# Grad-CAM Gallery Notes

This file documents Grad-CAM overlays in this folder.

For each image:
- **Truth** refers to the dataset label (`dx`) from `HAM10000_metadata.csv`.
- **Prediction** refers to the model’s top-1 class for that image.
- Grad-CAM is **qualitative**; it helps interpret what regions influence the prediction, but it does not guarantee correctness.

## Images

- `ISIC_0025964_overlay.png` — melanoma sample (truth: `mel`). Check if the heatmap emphasizes lesion region rather than background.
- `ISIC_0030623_overlay.png` — melanoma sample (truth: `mel`). Used to sanity-check attention focus.
- `ISIC_0024698_overlay.png` — nevus sample (truth: `nv`). Used as a common-class reference.
- `ISIC_0032212_overlay.png` — nevus sample (truth: `nv`). Used as a common-class reference.
- `ISIC_0028155_overlay.png` — basal cell carcinoma sample (truth: `bcc`). Used to compare attention patterns across lesion types.

