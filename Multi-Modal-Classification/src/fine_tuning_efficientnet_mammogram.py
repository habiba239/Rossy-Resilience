#!/usr/bin/env python
# coding: utf-8
"""
EfficientNetV2L Fine-Tuning on CBIS-DDSM
==========================================
Fine-tunes a pre-trained EfficientNetV2L mammography model on CBIS-DDSM
to produce binary (benign/malignant) predictions.

Base model (3-class EfficientNetV2L):
  Download from HuggingFace:
    hf_hub_download('hab200/breast-cancer-late-fusion', 'Models/base_mammo_3class.h5')
  Or set MAMMO_BASE_MODEL_PATH to your local .h5 / .keras file.

Dataset:
  - CBIS-DDSM images: set CBIS_DDSM_PATH
  - Split CSVs: data/tabular_ft_train.csv, tabular_ft_val.csv (produced by mammogram_propensity_score_catboost.py)

Output:
  - mammo_ft_classification.keras   (best checkpoint)
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from huggingface_hub import hf_hub_download

# ── Paths ─────────────────────────────────────────────────────────────────────
CBIS_DDSM_PATH      = os.environ.get('CBIS_DDSM_PATH', 'data/CBIS-DDSM')
DATA_DIR            = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR          = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Base model — download from HF or point to local file
MAMMO_BASE_MODEL_PATH = os.environ.get(
    'MAMMO_BASE_MODEL_PATH',
    hf_hub_download(repo_id='hab200/breast-cancer-late-fusion',
                    filename='Models/base_mammo_3class.h5')
)

IMG_SIZE   = 456
BATCH_SIZE = 16
SEED       = 42

# ── Load split CSVs ───────────────────────────────────────────────────────────
ft_train_df   = pd.read_csv(os.path.join(DATA_DIR, 'tabular_ft_train.csv'))
ft_val_df     = pd.read_csv(os.path.join(DATA_DIR, 'tabular_ft_val.csv'))
final_test_df = pd.read_csv(os.path.join(DATA_DIR, 'tabular_final_test.csv'))

MAMMO_IMG_ROOT = os.path.join(CBIS_DDSM_PATH, 'jpeg')

def build_mammo_path(rel_path):
    if rel_path.startswith('CBIS-DDSM'):
        return rel_path.replace('CBIS-DDSM', MAMMO_IMG_ROOT, 1)
    return os.path.join(MAMMO_IMG_ROOT, rel_path)

for df in [ft_train_df, ft_val_df, final_test_df]:
    df['full_img_path'] = df['full_img_path'].apply(build_mammo_path)

print(f'Train: {len(ft_train_df)} | Val: {len(ft_val_df)} | Test: {len(final_test_df)}')

# ── Preprocessing & Dataset ───────────────────────────────────────────────────
def load_and_preprocess(img_path, label, augment=False):
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32), np.float32(label)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32)
    if augment:
        if np.random.rand() > 0.5: img = np.fliplr(img)
        if np.random.rand() > 0.5: img = np.flipud(img)
        img = np.clip(img * np.random.uniform(0.85, 1.15), 0, 255)
    return preprocess_input(img), np.float32(label)

def make_dataset(df, augment=False, shuffle=False):
    paths  = df['full_img_path'].values
    labels = df['label'].values.astype(np.float32)

    def load_fn(path, label):
        img, lbl = tf.numpy_function(
            lambda p, l: load_and_preprocess(p.decode(), float(l), augment),
            [path, label], [tf.float32, tf.float32]
        )
        img.set_shape([IMG_SIZE, IMG_SIZE, 3])
        lbl.set_shape([])
        return img, lbl

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED)
    return ds.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(ft_train_df,   augment=True,  shuffle=True)
val_ds   = make_dataset(ft_val_df,     augment=False, shuffle=False)

n_b = (ft_train_df['label'] == 0).sum()
n_m = (ft_train_df['label'] == 1).sum()
n   = len(ft_train_df)
class_weight = {0: n / (2 * n_b), 1: n / (2 * n_m) * 2}
print('Class weights:', {k: round(v, 3) for k, v in class_weight.items()})

# ── Fine-tuning experiments ───────────────────────────────────────────────────
def run_experiment(freeze_until, exp_name, epochs=15):
    print(f'\n{"="*55}')
    print(f'  Experiment {exp_name}: freeze_until={freeze_until}')
    print(f'{"="*55}')

    base_model = load_model(MAMMO_BASE_MODEL_PATH, compile=False)
    inputs  = base_model.input
    outputs = base_model.output
    prob_malignant = outputs[:, 2:3]  # extract malignant class probability
    model = tf.keras.Model(inputs=inputs, outputs=prob_malignant)

    for layer in model.layers:
        layer.trainable = False
    if freeze_until == 0:
        for layer in model.layers:
            layer.trainable = True
    else:
        for layer in model.layers[freeze_until:]:
            layer.trainable = True

    lr = 1e-5 if freeze_until != 0 else 5e-6

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )

    ckpt_path = os.path.join(OUTPUT_DIR, f'mammo_ft_{exp_name}.keras')
    callbacks = [
        ModelCheckpoint(ckpt_path, monitor='val_auc', mode='max', save_best_only=True, verbose=0),
        EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=3, min_lr=1e-8, verbose=0),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        callbacks=callbacks, class_weight=class_weight, verbose=1)

    best_val_auc = max(history.history['val_auc'])
    print(f'  Best Val AUC: {best_val_auc:.4f}')
    return model, history, best_val_auc, ckpt_path


# ── Run experiments ───────────────────────────────────────────────────────────
results = {}
results['A_last10'], hist_A, auc_A, ckpt_A = run_experiment(-10, 'A_last10')
results['B_last30'], hist_B, auc_B, ckpt_B = run_experiment(-30, 'B_last30')

print('\n' + '='*55)
print('RESULTS SUMMARY')
print('='*55)
print(f'  Exp A (last 10 layers): Val AUC = {auc_A:.4f}')
print(f'  Exp B (last 30 layers): Val AUC = {auc_B:.4f}')

best_exp = max([('A', auc_A), ('B', auc_B)], key=lambda x: x[1])
print(f'\n  Best experiment: {best_exp[0]} with AUC = {best_exp[1]:.4f}')
