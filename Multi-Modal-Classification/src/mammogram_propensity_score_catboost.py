#!/usr/bin/env python
# coding: utf-8
"""
Mammogram Propensity Score Estimation using CatBoost
=====================================================
Trains a tabular CatBoost classifier on CBIS-DDSM clinical metadata
to produce a malignancy Propensity Score (PS) for each mammogram sample.

Inputs:
  - CBIS-DDSM raw CSVs (mass + calcification, train + test)
  - dicom_info.csv
  - data/tabular_ft_train.csv, tabular_ft_val.csv, tabular_final_test.csv  (split keys)

Outputs:
  - data/tabular_ft_train.csv    (with PS_tabular column)
  - data/tabular_ft_val.csv
  - data/tabular_final_test.csv

Set CBIS_DDSM_PATH to the root directory of the CBIS-DDSM dataset.
Download from: https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from catboost import CatBoostClassifier

print('Libraries loaded ✓')

# ── Paths — set CBIS_DDSM_PATH to your local dataset root ────────────────────
BASE_PATH    = os.environ.get('CBIS_DDSM_PATH', 'data/CBIS-DDSM')
DATA_DIR     = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR   = DATA_DIR

MASSES_TRAIN = os.path.join(BASE_PATH, 'csv/mass_case_description_train_set.csv')
MASSES_TEST  = os.path.join(BASE_PATH, 'csv/mass_case_description_test_set.csv')
CALC_TRAIN   = os.path.join(BASE_PATH, 'csv/calc_case_description_train_set.csv')
CALC_TEST    = os.path.join(BASE_PATH, 'csv/calc_case_description_test_set.csv')
DICOM_CSV    = os.path.join(BASE_PATH, 'csv/dicom_info.csv')

mass_train = pd.read_csv(MASSES_TRAIN)
mass_test  = pd.read_csv(MASSES_TEST)
calc_train = pd.read_csv(CALC_TRAIN)
calc_test  = pd.read_csv(CALC_TEST)
dicom      = pd.read_csv(DICOM_CSV)

print('Files loaded:')
for name, df in [('mass_train', mass_train), ('mass_test', mass_test),
                 ('calc_train', calc_train), ('calc_test',  calc_test),
                 ('dicom',      dicom)]:
    print(f'  {name:<12s}  shape={df.shape}')

# ── Standardize and combine all splits ───────────────────────────────────────
def prep_mammo(df, split, abn_type):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df['split']            = split
    df['abnormality_type'] = abn_type
    for col in ['image_file_path', 'cropped_image_file_path', 'roi_mask_file_path']:
        if col in df.columns:
            df[col] = df[col].str.strip()
    df.rename(columns={'breast density': 'breast_density'}, inplace=True)
    return df

mammo = pd.concat([
    prep_mammo(mass_train, 'train', 'mass'),
    prep_mammo(mass_test,  'test',  'mass'),
    prep_mammo(calc_train, 'train', 'calcification'),
    prep_mammo(calc_test,  'test',  'calcification'),
], ignore_index=True)

print(f'Combined shape: {mammo.shape}')

# ── Merge DICOM paths ─────────────────────────────────────────────────────────
dicom_full = (dicom[dicom['SeriesDescription'] == 'full mammogram images']
              [['PatientID', 'image_path']]
              .rename(columns={'PatientID': 'key_full', 'image_path': 'full_img_path'}))
dicom_crop = (dicom[dicom['SeriesDescription'] == 'cropped images']
              [['PatientID', 'image_path']]
              .rename(columns={'PatientID': 'key_cropped', 'image_path': 'cropped_img_path'}))
dicom_roi  = (dicom[dicom['SeriesDescription'] == 'ROI mask images']
              [['PatientID', 'image_path']]
              .rename(columns={'PatientID': 'key_roi', 'image_path': 'roi_img_path'}))

mammo['key_full']    = mammo['image_file_path'].str.split('/').str[0]
mammo['key_cropped'] = mammo['cropped_image_file_path'].str.split('/').str[0]
mammo['key_roi']     = mammo['roi_mask_file_path'].str.split('/').str[0]

mammo = mammo.merge(dicom_full, on='key_full',    how='left')
mammo = mammo.merge(dicom_crop, on='key_cropped', how='left')
mammo = mammo.merge(dicom_roi,  on='key_roi',     how='left')

# ── Binary label + cleanup ────────────────────────────────────────────────────
mammo['label'] = mammo['pathology'].map({
    'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0
})
drop_cols = ['image_file_path', 'cropped_image_file_path', 'roi_mask_file_path',
             'key_full', 'key_cropped', 'key_roi']
mammo.drop(columns=drop_cols, inplace=True)

# ── Drop rows without full_img_path ──────────────────────────────────────────
df = mammo[mammo['full_img_path'].notnull()].copy()
print(f'Rows with full_img_path: {len(df)}')

# ── Drop non-feature columns ──────────────────────────────────────────────────
drop_for_model = ['patient_id', 'abnormality_id', 'pathology', 'image_view',
                  'left_or_right_breast', 'cropped_img_path', 'assessment',
                  'subtlety', 'roi_img_path']
df.drop(columns=drop_for_model, inplace=True)

# ── Fill nulls ────────────────────────────────────────────────────────────────
for col in ['calc_type', 'calc_distribution', 'mass_shape', 'mass_margins']:
    df[col] = df[col].fillna('NONE')

df_features = df.drop(columns=['label', 'split'], errors='ignore')
df_features = df_features.drop_duplicates(subset=['full_img_path'], keep='first')

# ── Load pre-computed split keys ──────────────────────────────────────────────
train_split = pd.read_csv(os.path.join(DATA_DIR, 'tabular_ft_train.csv'))
val_split   = pd.read_csv(os.path.join(DATA_DIR, 'tabular_ft_val.csv'))
test_split  = pd.read_csv(os.path.join(DATA_DIR, 'tabular_final_test.csv'))

train_tab = pd.merge(train_split[['full_img_path', 'label']], df_features, on='full_img_path', how='inner')
val_tab   = pd.merge(val_split[['full_img_path',   'label']], df_features, on='full_img_path', how='inner')
test_tab  = pd.merge(test_split[['full_img_path',  'label']], df_features, on='full_img_path', how='inner')

print(f'Train: {train_tab.shape} | Val: {val_tab.shape} | Test: {test_tab.shape}')

cat_cols = ['abnormality_type', 'breast_density', 'mass_shape',
            'mass_margins', 'calc_type', 'calc_distribution']
for col in cat_cols:
    for tab in [train_tab, val_tab, test_tab]:
        if col in tab.columns:
            tab[col] = tab[col].fillna('MISSING').astype(str)

X_train, y_train = train_tab[cat_cols], train_tab['label']
X_val,   y_val   = val_tab[cat_cols],   val_tab['label']
X_test,  y_test  = test_tab[cat_cols],  test_tab['label']

# ── Train CatBoost ────────────────────────────────────────────────────────────
print('\nTraining CatBoost...')
model = CatBoostClassifier(
    iterations=500, learning_rate=0.05, depth=6,
    cat_features=cat_cols, verbose=100, auto_class_weights='Balanced'
)
model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

train_tab['PS_tabular'] = model.predict_proba(X_train)[:, 1]
val_tab['PS_tabular']   = model.predict_proba(X_val)[:, 1]
test_tab['PS_tabular']  = model.predict_proba(X_test)[:, 1]

# ── Evaluate ──────────────────────────────────────────────────────────────────
test_preds = (test_tab['PS_tabular'] >= 0.5).astype(int)
print(f'\nTest AUC: {roc_auc_score(y_test, test_tab["PS_tabular"]):.4f}')
print(classification_report(y_test, test_preds))

# ── Save outputs ──────────────────────────────────────────────────────────────
train_tab[['full_img_path', 'label', 'PS_tabular']].to_csv(
    os.path.join(OUTPUT_DIR, 'tabular_ft_train.csv'), index=False)
val_tab[['full_img_path', 'label', 'PS_tabular']].to_csv(
    os.path.join(OUTPUT_DIR, 'tabular_ft_val.csv'), index=False)
test_tab[['full_img_path', 'label', 'PS_tabular']].to_csv(
    os.path.join(OUTPUT_DIR, 'tabular_final_test.csv'), index=False)

print('Saved 3 CSVs to data/ ✓')
