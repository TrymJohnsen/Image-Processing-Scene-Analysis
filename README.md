# Image Processing Scene Analysis

This repository contains work for IT3212 Assignment 2, exploring image processing and scene analysis techniques.

## Running the code

Datasets are stored locally in `data/` and are not committed to Git.

1. Install the dependencies with `uv sync`, or with `pip install -r requirements.txt`.
2. Unpack the Intel Image Classification dataset so that the training images are in `data/intel-image-classification/seg_train/seg_train/<category>/`.
3. Run the scripts from the repository root:

```
uv run python src/fourier.py
uv run python src/pca.py
uv run python src/hog.py
uv run python src/lbp.py
uv run python src/blob.py
uv run python src/contour.py
```

## Building the report

`uv run report/build_pdf.py` converts `report/final_report.md` to `report/final_report.pdf` with headless Chrome.
