#!/usr/bin/env python
# coding: utf-8
"""
Late Fusion — Training Script
==============================
Loads paired mammogram and ultrasound images, runs inference through
their respective fine-tuned models, and trains a Logistic Regression
fusion classifier on the combined image-level probabilities.

Models downloaded automatically from HuggingFace:
  hab200/breast-cancer-late-fusion

Dataset images:
  - Mammogram: set CBIS_DDSM_PATH
  - Ultrasound: set BUS_BRA_PATH

Input: data/all_pairs.csv  (produced by pairing_data_optimal_transport.py)
Output: fusion_model.pkl + fusion_config.json
"""

import os
import pickle
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as prep_mammo
from tensorflow.keras.applications.resnet50 import preprocess_input as prep_us
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print('TF Version:', tf.__version__)
print('GPUs Available:', len(tf.config.list_physical_devices('GPU')))

# ── Paths ─────────────────────────────────────────────────────────────────────
CBIS_DDSM_PATH = os.environ.get('CBIS_DDSM_PATH', 'data/CBIS-DDSM')
DATA_DIR       = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR     = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PAIRS_CSV_PATH = os.path.join(DATA_DIR, 'all_pairs.csv')

# ── Download models from HuggingFace ─────────────────────────────────────────
HF_REPO = 'hab200/breast-cancer-late-fusion'
print('Downloading models from HuggingFace...')
MAMMO_MODEL_PATH = hf_hub_download(repo_id=HF_REPO, filename='Models/mammo_ft_classification.keras')
US_MODEL_PATH    = hf_hub_download(repo_id=HF_REPO, filename='Models/ultrasound_ft_classification.keras')
print('Models downloaded ✓')

# ── Load models ───────────────────────────────────────────────────────────────
print('Loading models...')
mammo_model = load_model(MAMMO_MODEL_PATH, compile=False)
us_model    = load_model(US_MODEL_PATH,    compile=False)
print(f'Mammo input: {mammo_model.input_shape}')
print(f'US input:    {us_model.input_shape}')

# ── Preprocessing ─────────────────────────────────────────────────────────────
MAMMO_BASE_DIR = os.path.join(CBIS_DDSM_PATH, 'jpeg')

def prepare_mammo_image(img_path):
    raw_path   = img_path.replace('CBIS-DDSM/', '')
    full_path  = os.path.join(MAMMO_BASE_DIR, raw_path)
    img = cv2.imread(full_path)
    if img is None:
        raise ValueError(f'Could not read mammogram: {full_path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (456, 456), interpolation=cv2.INTER_CUBIC)
    return prep_mammo(img.astype(np.float32))

def prepare_us_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    return prep_us(np.array(img, dtype=np.float32))

# ── Run inference on all pairs ────────────────────────────────────────────────
from tqdm.auto import tqdm

pairs_df = pd.read_csv(PAIRS_CSV_PATH)
print(f'Total pairs: {len(pairs_df)}')

ps_mammo_img_list, ps_us_img_list = [], []

for index, row in tqdm(pairs_df.iterrows(), total=len(pairs_df)):
    try:
        m_img   = np.expand_dims(prepare_mammo_image(row['mammo_path']), 0)
        m_preds = mammo_model.predict(m_img, verbose=0)
        m_ps    = float(m_preds[0][1]) if m_preds.shape[-1] > 1 else float(m_preds[0][0])

        u_img   = np.expand_dims(prepare_us_image(row['us_path']), 0)
        u_preds = us_model.predict(u_img, verbose=0)
        u_ps    = float(u_preds[0][1]) if u_preds.shape[-1] > 1 else float(u_preds[0][0])

        ps_mammo_img_list.append(m_ps)
        ps_us_img_list.append(u_ps)
    except Exception as e:
        print(f'Error row {index}: {e}')
        ps_mammo_img_list.append(np.nan)
        ps_us_img_list.append(np.nan)

pairs_df['PS_mammo_img'] = ps_mammo_img_list
pairs_df['PS_us_img']    = ps_us_img_list
pairs_df = pairs_df.dropna(subset=['PS_mammo_img', 'PS_us_img']).reset_index(drop=True)

print(f'Successfully processed {len(pairs_df)} pairs')

inference_path = os.path.join(OUTPUT_DIR, 'synthetic_pairs_with_img_ps.csv')
pairs_df.to_csv(inference_path, index=False)

# ── Train Logistic Regression fusion model ────────────────────────────────────
train_df = pairs_df[pairs_df['split'] == 'ft_train']
val_df   = pairs_df[pairs_df['split'] == 'ft_val']
test_df  = pairs_df[pairs_df['split'] == 'final_test']

features = ['PS_mammo_img', 'PS_us_img']
X_train, y_train = train_df[features], train_df['label']
X_val,   y_val   = val_df[features],   val_df['label']
X_test,  y_test  = test_df[features],  test_df['label']

param_grid  = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

fusion_model = grid_search.best_estimator_
print(f'Best C: {grid_search.best_params_["C"]}')

y_prob_val  = fusion_model.predict_proba(X_val)[:, 1]
y_prob_test = fusion_model.predict_proba(X_test)[:, 1]

val_auc  = roc_auc_score(y_val,  y_prob_val)
test_auc = roc_auc_score(y_test, y_prob_test)
print(f'Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f}')

# ── Youden-J threshold optimization ──────────────────────────────────────────
fpr, tpr, thresholds = roc_curve(y_val, y_prob_val)
j_scores   = tpr - fpr
best_idx   = np.argmax(j_scores)
THRESHOLD  = float(thresholds[best_idx])
print(f'Optimal threshold (Youden-J): {THRESHOLD:.4f}')

# ── Save fusion model + config ────────────────────────────────────────────────
fusion_pkl_path    = os.path.join(OUTPUT_DIR, 'fusion_model.pkl')
fusion_config_path = os.path.join(OUTPUT_DIR, 'fusion_config.json')

with open(fusion_pkl_path, 'wb') as f:
    pickle.dump(fusion_model, f)

with open(fusion_config_path, 'w') as f:
    json.dump({'threshold': THRESHOLD, 'class_names': ['benign', 'malignant'],
               'test_auc': test_auc, 'val_auc': val_auc}, f, indent=2)

print(f'Saved fusion_model.pkl and fusion_config.json to {OUTPUT_DIR} ✓')
