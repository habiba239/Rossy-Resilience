#!/usr/bin/env python
# coding: utf-8
"""
ResNet50 Fine-Tuning on BUS-BRA (Ultrasound Classification)
=============================================================
Fine-tunes a ResNet50 model on BUS-BRA ultrasound images for
binary (benign/malignant) classification.

Base model: downloaded from HuggingFace hab200/breast-cancer-late-fusion
Dataset:
  - BUS-BRA images: set BUS_BRA_PATH
  - Split CSVs: data/train_predictions.csv, validation_predictions.csv, test_predictions.csv

Output:
  - ultrasound_ft_classification.keras   (best checkpoint)
"""

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import AUC, Recall, Precision
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from huggingface_hub import hf_hub_download

SEED = 42
tf.random.set_seed(SEED); np.random.seed(SEED); random.seed(SEED)
print('TF:', tf.__version__)
print('GPUs:', tf.config.list_physical_devices('GPU'))

# ── Paths ─────────────────────────────────────────────────────────────────────
BUS_BRA_PATH = os.environ.get('BUS_BRA_PATH', 'data/BUS-BRA')
DATA_DIR     = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(DATA_DIR, 'train_predictions.csv')
VAL_CSV   = os.path.join(DATA_DIR, 'validation_predictions.csv')
TEST_CSV  = os.path.join(DATA_DIR, 'test_predictions.csv')

# Base model (ResNet50 pre-trained) — download from HF
BASE_MODEL_PATH = os.environ.get(
    'US_BASE_MODEL_PATH',
    hf_hub_download(repo_id='hab200/breast-cancer-late-fusion',
                    filename='Models/ultrasound_ft_classification.keras')
)

IMG_SIZE   = 224
BATCH_SIZE = 32
