#!/usr/bin/env python
# coding: utf-8
"""
Ultrasound Propensity Score Estimation using CatBoost
======================================================
Trains a tabular CatBoost classifier on BUS-BRA clinical metadata
to produce a malignancy Propensity Score (PS) for each ultrasound sample.

Input:
  - BUS-BRA metadata CSV (bus_data.csv)
    Download: https://www.kaggle.com/datasets/orvile/bus-bra-a-breast-ultrasound-dataset
    Set BUS_BRA_PATH to the directory containing bus_data.csv

Outputs:
  - data/train_predictions.csv
  - data/validation_predictions.csv
  - data/test_predictions.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
from catboost import CatBoostClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
BUS_BRA_PATH = os.environ.get('BUS_BRA_PATH', 'data/BUS-BRA')
DATA_DIR     = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR   = DATA_DIR

CSV_PATH = os.path.join(BUS_BRA_PATH, 'bus_data.csv')

us_df = pd.read_csv(CSV_PATH)
print(f'BUS-BRA metadata loaded: {us_df.shape}')
print(us_df.head())
