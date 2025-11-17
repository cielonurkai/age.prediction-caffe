# Age Prediction using DEX Caffe Model

This project uses the DEX (Deep EXpectation) age prediction model trained on the IMDB-WIKI dataset to predict ages from facial images.

## Overview

- Extracts 30 balanced face samples (15 male, 15 female) from WIKI dataset
- Runs age prediction using pre-trained DEX Caffe model
- Compares predicted ages with true ages for analysis

## Requirements
```bash
pip install opencv-python numpy pandas scipy
```

## Model Files

Download the following files from [IMDB-WIKI dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/):
- `dex_imdb_wiki.caffemodel` (pre-trained model weights)
- `age.prototxt.txt` (model architecture)
- WIKI faces dataset

## Usage

1. **Extract faces from WIKI dataset:**
```bash
python select_faces.py
```

2. **Run age predictions:**
```bash
python predict_ages.py
```

3. **Results saved to:**
   - `selected_faces/` - extracted face images
   - `faces_with_predictions.csv` - predictions and metadata

## Project Structure
```
age-prediction/
├── select_faces.py          # Extract balanced face samples
├── predict_ages.py          # Run age predictions
├── faces_with_predictions.csv
└── selected_faces/
    ├── face_01.jpg
    ├── face_02.jpg
    └── ...
```

## Notes

- Model files (.caffemodel) are not included due to size (>100MB)
- Face images are not included for privacy
- Update file paths in scripts to match your local setup