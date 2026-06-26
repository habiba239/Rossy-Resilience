# Task 2 — Multi-Modal Classification (Late Fusion)

> Part of the graduation project: **"A Clinical Decision Support System (CDSS) for Breast Cancer Diagnosis"**

This task builds a multi-modal malignancy classifier that fuses **mammogram** and **ultrasound** predictions. The core challenge: standard datasets are never paired — no patient has both modalities simultaneously. We solve this with a three-stage pipeline:

1. **Propensity Score Estimation** — Train a CatBoost tabular classifier per modality to compute a clinically-grounded malignancy probability for each sample.
2. **Optimal Transport Pairing** — Use the Hungarian algorithm to synthetically pair mammogram–ultrasound samples with similar propensity scores.
3. **Late Fusion Classification** — Pass each paired image through its fine-tuned deep learning model; fuse the two image-level probabilities with Logistic Regression.

---

## Pipeline

```
CBIS-DDSM metadata          BUS-BRA metadata
(tabular clinical features)  (tabular clinical features)
        │                           │
        ▼                           ▼
  CatBoost Mammogram          CatBoost Ultrasound
  Classifier (AUC: 0.81)      Classifier (AUC: 0.79)
        │                           │
        ▼                           ▼
    PS_mammo ∈ [0,1]            PS_us ∈ [0,1]
        │                           │
        └──────────┬────────────────┘
                   ▼
       Optimal Transport Pairing
       (Hungarian algorithm, within-class,
        PS_diff threshold < 0.15)
       → 1,681 synthetic pairs
         (1,141 train / 282 val / 258 test)
         mean PS_diff = 0.027
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
  EfficientNetV2L          ResNet50
  on Mammogram             on Ultrasound
  (456×456, CBIS-DDSM)     (224×224, BUS-BRA)
         │                    │
         ▼                    ▼
   PS_mammo_img          PS_us_img
         └─────────┬──────────┘
                   ▼
         Logistic Regression
         (5-fold CV, Youden-J threshold: 0.334)
                   │
                   ▼
           Final Prediction
           (Benign / Malignant)
```

---

## Datasets

| Dataset | Modality | Role | Access |
|---------|----------|------|--------|
| **CBIS-DDSM** | Mammogram | Tabular PS training + image model fine-tuning | [The Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM) |
| **BUS-BRA** | Ultrasound | Tabular PS training + image model fine-tuning | [Kaggle - orvile/bus-bra](https://www.kaggle.com/datasets/orvile/bus-bra-a-breast-ultrasound-dataset) |

**CBIS-DDSM features used:** abnormality type, breast density, mass shape, mass margins, calcification type, calcification distribution.

**BUS-BRA features used:** lesion dimensions (width, height), morphological scores (HOB, K5B, K10B, HOP, K5P, K10P).

---

## Results

### Propensity Score (CatBoost Classifiers)

| Model | Dataset | AUC (Test) |
|-------|---------|-----------|
| Mammogram CatBoost | CBIS-DDSM | **0.81** |
| Ultrasound CatBoost | BUS-BRA | **0.79** |

### OT Pairing Quality
- Total synthetic pairs: **1,681** (1,141 train / 282 val / 258 test)
- Mean PS difference: **0.027** (near-perfect cross-modal alignment)
- Pairs discarded (PS_diff > 0.15): filtered out to ensure quality

### Late Fusion Classification (Test set: 258 pairs)

| Model | AUC | Sensitivity | Specificity |
|-------|-----|-------------|-------------|
| Mammogram Only | 0.75 | 86.75% | 52.57% |
| Ultrasound Only | 0.89 | 79.52% | 78.86% |
| **Late Fusion (Ours)** | **0.92** | **84.34%** | **84.57%** |

> Decision threshold optimized using the **Youden J statistic** on the validation set → final threshold: **0.334**

The Late Fusion model outperforms both single-modality baselines, demonstrating the complementary diagnostic value of combining mammography (high sensitivity) with ultrasound (high specificity).

---

## Image Model Details

### EfficientNetV2L — Mammogram Classifier
- Input: 456×456, cubic interpolation
- Last 30 layers unfrozen
- Augmentation: horizontal/vertical flips, brightness jitter ±15%
- Class imbalance: sample weighting

### ResNet50 — Ultrasound Classifier
- Input: 224×224, RGB conversion
- Classification head: GlobalAveragePooling → Dense(256) → Dropout(0.4) → Binary output
- Last 30 layers unfrozen
- Class imbalance: balanced class weights

---

## Data Files (in `data/`)

| File | Description |
|------|-------------|
| `tabular_ft_train.csv` | Mammogram PS scores — training split |
| `tabular_ft_val.csv` | Mammogram PS scores — validation split |
| `tabular_final_test.csv` | Mammogram PS scores — test split |
| `train_predictions.csv` | US model predictions — training split |
| `validation_predictions.csv` | US model predictions — validation split |
| `test_predictions.csv` | US model predictions — test split |
| `all_pairs.csv` | Full OT-paired dataset (1,681 pairs) with PS values, labels, and splits |

---

## Repository Structure

```
Multi-Modal-Classification/
│
├── src/                                               # Python scripts
│   ├── mammogram_propensity_score_catboost.py         # CBIS-DDSM preprocessing + CatBoost PS
│   ├── ultrasound_propensity_score_catboost.py        # BUS-BRA preprocessing + CatBoost PS
│   ├── pairing_data_optimal_transport.py              # OT pairing (Hungarian algorithm)
│   ├── fine_tuning_efficientnet_mammogram.py          # EfficientNetV2L fine-tuning
│   ├── ultrasound_classification_resnet50.py          # ResNet50 fine-tuning
│   ├── late_fusion.py                                 # Late fusion inference
│   └── full_pipeline.py                              # End-to-end inference pipeline
│
├── notebooks/                                         # Original Jupyter notebooks
│   ├── mammogram-propensity-score-catboost.ipynb
│   ├── ultrasound-propensity-score-catboost.ipynb
│   ├── pairing-data-optimal-transport.ipynb
│   ├── fine-tuning-efficientnet-mammogram-cbis-ddsm.ipynb
│   ├── ultrasound-classification-resnet50.ipynb
│   ├── late-fusion.ipynb
│   └── full-pipline.ipynb
│
├── data/                                              # Propensity scores & paired data CSVs
│   ├── tabular_ft_train.csv
│   ├── tabular_ft_val.csv
│   ├── tabular_final_test.csv
│   ├── train_predictions.csv
│   ├── validation_predictions.csv
│   ├── test_predictions.csv
│   └── all_pairs.csv
│
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run end-to-end inference
```python
from src.full_pipeline import predict

result = predict(
    mammo_img_path="path/to/mammogram.jpg",
    us_img_path="path/to/ultrasound.png"
)
# returns: {'prediction': 'Malignant', 'confidence': 0.87}
```

### Training order (reproduce from scratch)
```bash
# Step 1: Generate propensity scores
python src/mammogram_propensity_score_catboost.py
python src/ultrasound_propensity_score_catboost.py

# Step 2: Pair the modalities
python src/pairing_data_optimal_transport.py

# Step 3: Fine-tune image models
python src/fine_tuning_efficientnet_mammogram.py
python src/ultrasound_classification_resnet50.py

# Step 4: Late fusion
python src/late_fusion.py
```
---
### Pretrained models
Models are hosted on Hugging Face Hub:
```
hab200/breast-cancer-late-fusion/tree/main/Models
```

## ⚠️ Medical Disclaimer

This project is developed for educational and research purposes only as part of a graduation project. 

**It is NOT a medical device and should not be used as a substitute for professional medical advice, diagnosis, or treatment.** The diagnostic predictions and segmentation results generated by this system are intended to assist clinical decision-making and should always be reviewed and verified by qualified radiologists or healthcare professionals. Never disregard professional medical advice or delay seeking it because of information provided by this system. The authors assume no responsibility for any clinical decisions made based on the output of this software.