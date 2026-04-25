Dataset: HAM10000 (Human Against Machine with 10000 training images)

How to obtain the data
1. Download the dataset from one of these sources:
   - Harvard Dataverse DOI: 10.7910/DVN/DBW86T
   - Kaggle mirror: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

How to place the data
Put the downloaded files under:

data/ham10000/
  HAM10000_metadata.csv
  ham10000_images_part_1/
  ham10000_images_part_2/

Notes
- Do not commit the dataset itself to the repository.
- The code expects the metadata CSV and image folders to be available locally.
- A single merged images/ folder is also acceptable if the image paths are adapted consistently.
