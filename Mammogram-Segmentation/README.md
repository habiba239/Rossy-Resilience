# Task 1 — Mammogram Detection & Segmentation

> Part of the graduation project: **"A Clinical Decision Support System (CDSS) for Breast Cancer Diagnosis"**

This task implements a two-stage Computer-Aided Detection (CAD) pipeline that automatically **locates** and **delineates** breast masses in mammograms. Rather than running a segmentation model on the entire high-resolution image (computationally prohibitive and prone to class imbalance), we first detect the mass region with YOLO11s, then run U-Net only on the cropped Region of Interest (ROI).

---

## Approach: Detect-Then-Segment

```
Raw Mammogram
      │
      ▼
  Preprocessing
  ─ Border removal (crop 2.5% top/bottom, 1% sides)
  ─ Orientation normalization (flip to right-facing)
  ─ Background extraction & ROI crop
  ─ CLAHE contrast enhancement
  ─ Resize + zero-pad to 640×640
      │
      ▼
Stage 1 — YOLO11s Detection
  ─ Scans full mammogram
  ─ Outputs bounding box(es) around suspicious mass regions
      │
      ▼
Stage 2 — U-Net Segmentation
  ─ Receives 128×128 cropped ROI from YOLO box
  ─ MobileNetV3-Small encoder (ImageNet pre-trained)
  ─ Outputs binary pixel-level mask of the mass boundary
      │
      ▼
  Final Mask + Overlay
```

---

## Datasets

| Dataset | Role | Size | Source |
|---------|------|------|--------|
| **CBIS-DDSM** | Primary training & validation | 1,489 mammograms (974 train / 165 val / 350 test) | [The Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM) |
| **INbreast** | Fine-tuning (FFDM digital scans) | 106 images with XML polygon annotations | [Kaggle - martholi/inbreast](https://www.kaggle.com/datasets/martholi/inbreast) |
| **BCDR** | External test only (zero-shot) | 20 images | [bcdr.eu](https://bcdr.eu/) |

**Ground Truth Generation:**
- CBIS-DDSM: Bounding boxes derived algorithmically from extreme coordinates of physician-annotated pixel masks.
- INbreast: Custom XML parser extracted polygon vertices to build binary masks; multi-mass instances combined via bitwise OR.

---

## Results

### Stage 1 — YOLO11s Detection

**5-Fold Cross-Validation on INbreast** (conf=0.25):

| Fold | Train | Val | Precision | Recall | mAP@50 |
|------|-------|-----|-----------|--------|--------|
| 1 | 247 | 22 | 0.708 | 0.826 | 0.766 |
| 2 | 249 | 21 | 0.946 | 0.826 | 0.849 |
| 3 | 251 | 21 | 0.861 | 0.818 | 0.782 |
| 4 | 250 | 21 | 0.863 | 0.810 | 0.830 |
| 5 | 252 | 21 | 0.943 | 0.708 | 0.777 |
| **Mean** | — | — | **0.864** | **0.798** | **0.801** |
| **Std** | — | — | 0.097 | 0.051 | 0.036 |

**External Validation on BCDR** (unseen, zero-shot):

| Dataset | Images | Best Conf | Precision | Recall | mAP@50 |
|---------|--------|-----------|-----------|--------|--------|
| BCDR | 20 | 0.15 | 0.879 | 0.810 | **0.776** |

### Stage 2 — U-Net Segmentation

**5-Fold Cross-Validation on INbreast:**
- Mean Dice Score: **0.9228** ± 0.009
- Mean IoU: **0.8591** ± 0.009

**External Validation:**

| Dataset | Type | Dice Score | IoU |
|---------|------|-----------|-----|
| BCDR (Test) | Unseen External | **0.8619** | 0.7690 |
| CBIS-DDSM (Test) | Legacy Data Check | **0.8914** | 0.8069 |

---

## Model Architecture Details

### YOLO11s (Stage 1)
- **Architecture:** YOLO11s — 182 layers, ~9.4M parameters, 21.5 GFLOPs
- **Backbone:** CSP-based with multi-scale detection head
- **Input size:** 640×640
- **Transfer Learning:** 3-stage strategy — Turkish DigitalEye Mammography → CBIS-DDSM (mAP@50: 0.610) → INbreast fine-tune

**YOLO Fine-Tuning Hyperparameters:**

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 60 |
| Image Size | 640×640 |
| Batch Size | 8 |
| Optimizer | AdamW |
| Initial LR | 1×10⁻⁴ |
| Final LR Ratio | 0.01 |
| Weight Decay | 1×10⁻³ |
| Frozen Layers | 5 (backbone stem) |
| AMP | Enabled |

### U-Net (Stage 2)
- **Architecture:** U-Net + MobileNetV3-Small encoder (ImageNet pre-trained)
- **Input size:** 128×128 cropped ROI
- **Loss:** Pure Dice Loss (smooth=1.0)
- **Optimizer:** Adam + ReduceLROnPlateau (Factor: 0.6, Patience: 5)
- **Training:** Phase 1 — CBIS-DDSM pre-training (160 epochs, Dice: 0.904) → Phase 2 — INbreast fine-tuning (100 epochs, encoder frozen for first 10)
- **Data Augmentation:** Flip, Affine, ElasticTransform, Gamma, Noise

---

## Repository Structure

```
Mammogram-Segmentation/
│
├── src/                                              # Python scripts
│   ├── mammogram_preprocessing_baseline_yolo11.py   # Data prep + YOLO baseline training
│   ├── fine_tuning_yolo11_detection.py               # YOLO fine-tuning on INbreast
│   ├── breast_mass_segmentation_unet.py              # U-Net training
│   └── pipeline_detection_segmentation.py            # End-to-end inference
│
├── notebooks/                                        # Original Jupyter notebooks
│   ├── mammogram-preprocessing-baseline-yolo11.ipynb
│   ├── fine-tuning-yolo11-detection.ipynb
│   ├── breast-mass-segmentation-unet.ipynb
│   └── pipeline-detection-segmentation.ipynb
│
├── data/                                             # Dataset placeholder
│   └── README_data.md
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

### Run pipeline on a new mammogram
```python
from src.pipeline_detection_segmentation import load_models, full_pipeline

yolo_model, unet_model = load_models()  # loads from Hugging Face Hub

result = full_pipeline(
    img_path="path/to/mammogram.jpg",
    yolo_model=yolo_model,
    unet_model=unet_model,
    yolo_conf=0.15,
    yolo_iou=0.25
)

# result keys: 'final_mask', 'boxes', 'overlay'
```

### Training order (reproduce from scratch)
```bash
python src/mammogram_preprocessing_baseline_yolo11.py   # Step 1: data prep + YOLO baseline
python src/fine_tuning_yolo11_detection.py               # Step 2: YOLO fine-tuning
python src/breast_mass_segmentation_unet.py              # Step 3: U-Net training
```

### Pretrained models
Models are hosted on Hugging Face Hub:
```
hab200/Breast-Cancer-Mammogram-Detection-Segmentation_new
```
---

## ⚠️ Medical Disclaimer

This project is developed for educational and research purposes only as part of a graduation project. 

**It is NOT a medical device and should not be used as a substitute for professional medical advice, diagnosis, or treatment.** The diagnostic predictions and segmentation results generated by this system are intended to assist clinical decision-making and should always be reviewed and verified by qualified radiologists or healthcare professionals. Never disregard professional medical advice or delay seeking it because of information provided by this system. The authors assume no responsibility for any clinical decisions made based on the output of this software.