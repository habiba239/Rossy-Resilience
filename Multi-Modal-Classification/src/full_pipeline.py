#!/usr/bin/env python
# coding: utf-8
"""
Late Fusion Inference Pipeline
================================
End-to-end inference: given a mammogram and an ultrasound image,
returns a binary malignancy prediction and confidence score.

Models are downloaded automatically from HuggingFace:
  hab200/breast-cancer-late-fusion

Usage:
  python full_pipeline.py --mammo path/to/mammogram.jpg --us path/to/ultrasound.png
"""

import os
import argparse
import pickle
import json
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as prep_mammo
from tensorflow.keras.applications.resnet50 import preprocess_input as prep_us
from huggingface_hub import hf_hub_download
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print('TF:', tf.__version__)
print('GPUs:', tf.config.list_physical_devices('GPU'))

# ── Download models from HuggingFace ─────────────────────────────────────────
HF_REPO = 'hab200/breast-cancer-late-fusion'

print('Downloading models from Hugging Face...')
mammo_path  = hf_hub_download(repo_id=HF_REPO, filename='Models/mammo_ft_classification.keras')
us_path     = hf_hub_download(repo_id=HF_REPO, filename='Models/ultrasound_ft_classification.keras')
fusion_path = hf_hub_download(repo_id=HF_REPO, filename='Models/fusion_model.pkl')
config_path = hf_hub_download(repo_id=HF_REPO, filename='Models/fusion_config.json')
print('All models downloaded ✓')

# ── Load models ───────────────────────────────────────────────────────────────
print('Loading models...')
mammo_model  = load_model(mammo_path, compile=False)
us_model     = load_model(us_path,    compile=False)

with open(fusion_path, 'rb') as f:
    fusion_model = pickle.load(f)

with open(config_path) as f:
    config = json.load(f)

THRESHOLD   = config['threshold']
CLASS_NAMES = config['class_names']

print(f'Mammogram model input:  {mammo_model.input_shape}')
print(f'Ultrasound model input: {us_model.input_shape}')
print(f'Fusion threshold:       {THRESHOLD:.4f}')
print('Models loaded ✓')


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_mammogram(img_path: str) -> np.ndarray:
    """BGR → RGB → resize 456×456 → CLAHE → EfficientNetV2L preprocess."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f'Cannot read image: {img_path}')
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img  = cv2.resize(img, (456, 456), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq   = clahe.apply(gray)
    img  = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
    img  = prep_mammo(img.astype(np.float32))
    return np.expand_dims(img, axis=0)   # (1, 456, 456, 3)


def preprocess_ultrasound(img_path: str) -> np.ndarray:
    """PIL open → RGB → resize 224×224 → ResNet50 preprocess."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    arr = prep_us(np.array(img, dtype=np.float32))
    return np.expand_dims(arr, axis=0)   # (1, 224, 224, 3)


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(mammo_img_path: str, us_img_path: str,
            threshold: float = THRESHOLD) -> dict:
    """
    Late Fusion inference — Mammogram + Ultrasound.

    Returns:
      {'class': 'benign' | 'malignant', 'confidence': float (0–100)}
    """
    mammo_arr = preprocess_mammogram(mammo_img_path)
    us_arr    = preprocess_ultrasound(us_img_path)

    mammo_probs = mammo_model.predict(mammo_arr, verbose=0)[0]
    us_probs    = us_model.predict(us_arr,       verbose=0)[0]

    ps_mammo = float(mammo_probs[0]) if len(mammo_probs) == 1 else float(mammo_probs[1])
    ps_us    = float(us_probs[0])    if len(us_probs) == 1    else float(us_probs[1])

    features     = pd.DataFrame([[ps_mammo, ps_us]], columns=['PS_mammo_img', 'PS_us_img'])
    fusion_score = float(fusion_model.predict_proba(features)[0][1])

    if fusion_score >= threshold:
        prediction = 'malignant'
        confidence = 0.5 + 0.5 * ((fusion_score - threshold) / (1.0 - threshold))
    else:
        prediction = 'benign'
        confidence = 0.5 + 0.5 * ((threshold - fusion_score) / threshold)

    return {'class': prediction, 'confidence': round(confidence * 100, 2)}


# ── CLI entrypoint ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Late Fusion Breast Cancer Inference')
    parser.add_argument('--mammo', required=True, help='Path to mammogram image')
    parser.add_argument('--us',    required=True, help='Path to ultrasound image')
    args = parser.parse_args()

    result = predict(args.mammo, args.us)

    print('=' * 45)
    print('PREDICTION RESULT')
    print('=' * 45)
    for k, v in result.items():
        print(f'  {k:<20}: {v}')
    print('=' * 45)
