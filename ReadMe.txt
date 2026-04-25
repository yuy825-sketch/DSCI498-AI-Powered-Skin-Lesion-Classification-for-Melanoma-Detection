AI-Powered Skin Lesion Classification for Melanoma Detection (DSCI 498)

Project description
This project builds an end-to-end deep learning pipeline for skin lesion image classification on the HAM10000 dataset. The work emphasizes both overall multiclass performance and melanoma sensitivity, and it also includes Grad-CAM interpretability, threshold-based melanoma screening analysis, confidence calibration diagnostics, subgroup analysis, and a Streamlit demo app.

Data source
- HAM10000 (Human Against Machine with 10000 training images)
- Harvard Dataverse DOI: 10.7910/DVN/DBW86T
- Kaggle mirror: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

Required packages
- Python 3.10+
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- scikit-learn
- Pillow
- streamlit

Install
1. pip install -r requirements.txt
2. pip install -e .

Dataset placement
Place the HAM10000 files under:
data/ham10000/
  HAM10000_metadata.csv
  ham10000_images_part_1/
  ham10000_images_part_2/

How to run the code
- Quick smoke test:
  python -m dsci498_skin.smoke

- Train a baseline classifier:
  python train.py --config configs/baseline.json --run-name baseline-effnetb0

- Launch the demo app:
  streamlit run app/app.py

- Unified entrypoint:
  python main.py --help

Main outputs
- Committed figures and summaries live under results/
- Article files live under paper/
- GitHub Pages demo site lives under docs/

Disclaimer
This project is for educational purposes only and is not medical advice.
