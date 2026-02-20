from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from dsci498_skin.infer import default_eval_transform, load_run_model, predict_topk
from dsci498_skin.interpret.gradcam import gradcam, infer_target_layer


def _overlay(image_rgb: np.ndarray, heatmap01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import cv2

    h, w = image_rgb.shape[:2]
    heatmap = (heatmap01 * 255).astype(np.uint8)
    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap_color + (1 - alpha) * image_rgb).astype(np.uint8)
    return overlay


@st.cache_resource
def _load_model(run_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_run_model(Path(run_dir), device=device)
    return loaded, device


def main() -> None:
    st.set_page_config(page_title="DSCI498 Skin Lesion Classifier", layout="wide")
    st.title("Skin Lesion Classification (HAM10000) — Demo")
    st.caption("Educational demo only. Not medical advice.")

    run_dir = st.text_input("Run directory (must contain best.pt + classes.json)", value="runs/<your_run_dir>")
    image_size = st.number_input("Model image size", min_value=64, max_value=512, value=224, step=16)
    topk = st.slider("Top-k predictions", min_value=1, max_value=7, value=3)

    uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if not uploaded:
        st.info("Upload an image to see predictions and Grad-CAM.")
        return

    image = Image.open(BytesIO(uploaded.read())).convert("RGB")
    st.image(image, caption="Input image", use_container_width=True)

    try:
        loaded, device = _load_model(run_dir)
    except Exception as e:
        st.error(f"Failed to load model from {run_dir}: {e}")
        return

    transform = default_eval_transform(int(image_size))
    preds = predict_topk(
        model=loaded.model,
        image=image,
        transform=transform,
        idx_to_class=loaded.idx_to_class,
        device=device,
        k=int(topk),
    )

    st.subheader("Predictions")
    for label, prob in preds:
        st.write(f"- `{label}`: **{prob:.4f}**")

    st.subheader("Grad-CAM")
    with st.spinner("Computing Grad-CAM..."):
        x = transform(image).unsqueeze(0).to(device)
        target_layer = infer_target_layer(loaded.model)
        cam_res = gradcam(model=loaded.model, target_layer=target_layer, x=x, class_idx=None)

        overlay = _overlay(np.array(image), cam_res.heatmap, alpha=0.45)
        st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)


if __name__ == "__main__":
    main()

