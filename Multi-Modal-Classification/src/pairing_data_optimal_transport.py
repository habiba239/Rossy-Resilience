#!/usr/bin/env python
# coding: utf-8
"""
Cross-Modal Pairing via Optimal Transport
==========================================
Pairs mammogram and ultrasound samples based on their Propensity Scores
using the Hungarian algorithm (Optimal Transport).

Inputs (from data/):
  - tabular_ft_train.csv, tabular_ft_val.csv, tabular_final_test.csv   (mammogram PS)
  - train_predictions.csv, validation_predictions.csv, test_predictions.csv  (ultrasound PS)

Output:
  - data/all_pairs.csv   (1,681 synthetic mammogram–ultrasound pairs)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance
import warnings
warnings.filterwarnings('ignore')

print('Libraries loaded ✓')

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = DATA_DIR

# Mammogram propensity scores
MAMMO_TRAIN = os.path.join(DATA_DIR, 'tabular_ft_train.csv')
MAMMO_VAL   = os.path.join(DATA_DIR, 'tabular_ft_val.csv')
MAMMO_TEST  = os.path.join(DATA_DIR, 'tabular_final_test.csv')

# Ultrasound propensity scores
US_TRAIN = os.path.join(DATA_DIR, 'train_predictions.csv')
US_VAL   = os.path.join(DATA_DIR, 'validation_predictions.csv')
US_TEST  = os.path.join(DATA_DIR, 'test_predictions.csv')

# ── Load and rename columns ───────────────────────────────────────────────────
mammo = {
    'ft_train':   pd.read_csv(MAMMO_TRAIN),
    'ft_val':     pd.read_csv(MAMMO_VAL),
    'final_test': pd.read_csv(MAMMO_TEST),
}
us = {
    'ft_train':   pd.read_csv(US_TRAIN),
    'ft_val':     pd.read_csv(US_VAL),
    'final_test': pd.read_csv(US_TEST),
}

for split in mammo:
    mammo[split] = mammo[split].rename(columns={
        'full_img_path': 'mammo_path',
        'PS_tabular':    'PS_mammo'
    })

for split in us:
    us[split] = us[split].rename(columns={
        'image_path': 'us_path',
        'ps_score':   'PS_us'
    })

print('=== Mammogram ===')
for split, df in mammo.items():
    print(f'  {split}: {len(df)} rows | label dist: {df["label"].value_counts().to_dict()}')
print('\n=== Ultrasound ===')
for split, df in us.items():
    print(f'  {split}: {len(df)} rows | label dist: {df["label"].value_counts().to_dict()}')


# ── Optimal Transport Pairing ─────────────────────────────────────────────────
def ot_pair_split(mammo_df: pd.DataFrame,
                  us_df: pd.DataFrame,
                  split_name: str,
                  max_ps_diff: float = 0.15) -> pd.DataFrame:
    """
    Pairs mammogram and ultrasound samples within each class label using
    the Hungarian algorithm. Discards pairs with PS_diff > max_ps_diff.
    """
    all_pairs = []

    for label in [0, 1]:
        label_name = 'Benign' if label == 0 else 'Malignant'
        m = mammo_df[mammo_df['label'] == label].reset_index(drop=True)
        u = us_df[us_df['label'] == label].reset_index(drop=True)

        if len(m) == 0 or len(u) == 0:
            print(f'  [{split_name}] {label_name}: SKIP — empty group')
            continue

        # Build cost matrix and solve assignment
        cost = np.abs(m['PS_mammo'].values[:, np.newaxis] - u['PS_us'].values[np.newaxis, :])
        wd_before = wasserstein_distance(m['PS_mammo'].values, u['PS_us'].values)
        row_idx, col_idx = linear_sum_assignment(cost)

        paired = pd.DataFrame({
            'mammo_path': m.loc[row_idx, 'mammo_path'].values,
            'us_path':    u.loc[col_idx, 'us_path'].values,
            'PS_mammo':   m.loc[row_idx, 'PS_mammo'].values,
            'PS_us':      u.loc[col_idx, 'PS_us'].values,
            'PS_diff':    cost[row_idx, col_idx],
            'label':      label,
            'split':      split_name,
        })

        wd_after     = paired['PS_diff'].mean()
        improvement  = (wd_before - wd_after) / wd_before * 100 if wd_before > 0 else 0
        print(f'  [{split_name}] {label_name}: mammo={len(m)}, us={len(u)} → {len(paired)} pairs '
              f'| mean PS_diff={wd_after:.4f} | improvement={improvement:.1f}%')

        before_filter = len(paired)
        paired = paired[paired['PS_diff'] <= max_ps_diff].reset_index(drop=True)
        removed = before_filter - len(paired)
        if removed > 0:
            print(f'    Filtered {removed} pairs with PS_diff > {max_ps_diff}')

        all_pairs.append(paired)

    return pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()


# ── Run pairing on all 3 splits ───────────────────────────────────────────────
print('\nRunning OT Pairing on all 3 splits...\n')
paired_splits = {}

for split_name in ['ft_train', 'ft_val', 'final_test']:
    print(f'--- {split_name} ---')
    paired = ot_pair_split(
        mammo_df=mammo[split_name],
        us_df=us[split_name],
        split_name=split_name,
        max_ps_diff=0.15
    )
    paired_splits[split_name] = paired
    print(f'  Total pairs: {len(paired)}\n')

all_pairs = pd.concat(list(paired_splits.values()), ignore_index=True)

print('=' * 55)
print('SUMMARY')
print('=' * 55)
for split_name, df in paired_splits.items():
    print(f'  {split_name}: {len(df)} pairs | label dist: {df["label"].value_counts().to_dict()}')
print(f'  TOTAL: {len(all_pairs)} pairs')

# ── Save ──────────────────────────────────────────────────────────────────────
all_pairs_path = os.path.join(OUTPUT_DIR, 'all_pairs.csv')
all_pairs.to_csv(all_pairs_path, index=False)
print(f'\nSaved → {all_pairs_path} ({len(all_pairs)} total pairs) ✓')
