
# # Phase 1: Breast Mass Detection_02 — YOLO11s Evaluation
# 
# 

# ## Install & Import libraries

import os, re, shutil, cv2, math
import numpy as np
import pandas as pd
import pydicom
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
from huggingface_hub import hf_hub_download, login
from ultralytics import YOLO
import albumentations as A

print(' Libraries ready!')

# ### 1. Download Pretrained Model (CBIS-DDSM)

REPO_ID        = 'hab200/Breast-Cancer-Mammogram-Detection-Segmentation'
FILENAME       = 'models/yolo_best.pt'
old_model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
print(f' Model downloaded: {old_model_path}')

# ### 2. Prepare Inbreast dataset

# ── Paths ──────────────────────────────────────────────────────────────────
DATASET_BASE = os.environ.get('INBREAST_PATH', 'data/INbreast/INbreast Release 1.0')
DICOM_DIR    = os.path.join(DATASET_BASE, 'AllDICOMs')
XML_DIR      = os.path.join(DATASET_BASE, 'AllXML')
CSV_FILE     = os.path.join(DATASET_BASE, 'INbreast.csv')
XLS_FILE     = os.path.join(DATASET_BASE, 'INbreast.xls')

RAW_OUT   = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'inbreast_mass')
IMG_RAW   = os.path.join(RAW_OUT, 'images')
MASK_RAW  = os.path.join(RAW_OUT, 'masks')
LABEL_RAW = os.path.join(RAW_OUT, 'labels')
for d in [IMG_RAW, MASK_RAW, LABEL_RAW]:
    os.makedirs(d, exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────
def parse_point(s):
    # Extract floating point coordinates from XML string format
    nums = re.findall(r'[-+]?\d*\.?\d+', s)
    return (int(round(float(nums[0]))), int(round(float(nums[1])))) if len(nums) >= 2 else None

def read_plist_dict(d):
    # Parse Apple Plist XML format commonly used in INbreast annotations
    result, children, i = {}, list(d), 0
    while i < len(children) - 1:
        if children[i].tag == 'key':
            result[children[i].text] = children[i + 1]
            i += 2
        else:
            i += 1
    return result

def get_mass_ids(csv_file, xls_file):
    # Cross-reference CSV and XLS metadata to isolate images containing actual masses
    try:
        csv_df = pd.read_csv(csv_file, sep=';')
    except:
        csv_df = pd.read_csv(csv_file)
        
    csv_df.columns = [c.strip() for c in csv_df.columns]
    csv_df['fn'] = csv_df['File Name'].astype(str).str.strip()
    
    xls_df = pd.read_excel(xls_file, engine='xlrd')
    xls_df.columns = [c.strip() for c in xls_df.columns]
    
    mass_col = next(c for c in xls_df.columns if c.strip().lower() == 'mass')
    xls_df['fn'] = xls_df['File Name'].apply(lambda x: str(int(x)) if pd.notna(x) else None)
    
    merged = csv_df.merge(xls_df[['fn', mass_col]], on='fn', how='left')
    ids = set(merged[merged[mass_col].astype(str).str.strip() == 'X']['fn'].tolist())
    print(f'  Mass IDs identified: {len(ids)}')
    return ids

def build_dicom_index(dicom_dir):
    # Map patient IDs to their respective DICOM file paths
    index = {dcm.stem.split('_')[0]: str(dcm) for dcm in Path(dicom_dir).rglob('*.dcm')}
    print(f'  DICOM index mapping: {len(index)} files')
    return index

def dicom_to_png(dcm_path):
    # Convert raw DICOM arrays to visualizable PNG arrays with CLAHE enhancement
    ds  = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)
    
    # Standardize polarity (ensure tissue is bright and background is dark)
    if getattr(ds, 'PhotometricInterpretation', '') == 'MONOCHROME1':
        arr = arr.max() - arr
        
    # Min-Max Normalization to 8-bit scale [0-255]
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    arr = arr.astype(np.uint8)
    
    # Apply localized contrast enhancement for medical feature visibility
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(arr), ds.Rows, ds.Columns

def parse_xml_masses(xml_path, img_h, img_w):
    # Extract ROI polygons from XML and convert them to YOLO Bounding Boxes
    try:
        content = re.sub(r'<!DOCTYPE[^>]*>', '', open(xml_path, encoding='utf-8', errors='ignore').read())
        root = ET.fromstring(content)
    except ET.ParseError as e:
        print(f'  [XML ERROR] {xml_path}: {e}')
        return [], []
        
    masks, bboxes = [], []
    root_kv = read_plist_dict(root.find('dict'))
    images_array = root_kv.get('Images')
    
    if images_array is None: return [], []
    
    for image_dict in images_array.findall('dict'):
        rois_array = read_plist_dict(image_dict).get('ROIs')
        if rois_array is None: continue
            
        for roi_dict in rois_array.findall('dict'):
            roi_kv    = read_plist_dict(roi_dict)
            name_node = roi_kv.get('Name')
            if name_node is None: continue
                
            name = (name_node.text or '').lower()
            # Strict filtering: only process ROIs classified as masses or nodules
            if not any(k in name for k in ['mass', 'nódulo', 'nodulo', 'spiculated']):
                continue
                
            point_node = roi_kv.get('Point_px')
            if point_node is None: continue
                
            pts = [list(p) for s in point_node.findall('string') if (p := parse_point(s.text or ''))]
            if len(pts) < 3: continue
                
            pts_np = np.array(pts, dtype=np.int32)
            mask   = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts_np], 255)
            masks.append(mask)
            
            # Extract tight bounding box coordinates
            x1 = max(pts_np[:, 0].min(), 0);  x2 = min(pts_np[:, 0].max(), img_w - 1)
            y1 = max(pts_np[:, 1].min(), 0);  y2 = min(pts_np[:, 1].max(), img_h - 1)
            
            # Convert to YOLO format (Center X, Center Y, Width, Height)
            bboxes.append((
                (x1 + x2) / 2.0 / img_w, (y1 + y2) / 2.0 / img_h,
                (x2 - x1) / img_w,        (y2 - y1) / img_h,
            ))
            
    return masks, bboxes

# --- 3. Main Data Extraction Pipeline ---
print('=' * 60)
print('INbreast Mass Preprocessing Initialization')
print('=' * 60)

mass_ids    = get_mass_ids(CSV_FILE, XLS_FILE)
dicom_index = build_dicom_index(DICOM_DIR)
print(f'  Matched Files: {len(mass_ids & set(dicom_index.keys()))}/{len(mass_ids)}')

processed, no_xml, no_annot, rows = 0, 0, 0, []

for file_id in tqdm(sorted(mass_ids), desc='Processing INbreast Datapoints'):
    dcm_path = dicom_index.get(file_id)
    if not dcm_path: continue
        
    try:
        img_arr, h, w = dicom_to_png(dcm_path)
    except Exception as e:
        print(f'  [SKIP] {file_id}: {e}')
        continue

    # Locate corresponding XML annotation file
    xml_path = None
    for candidate in [Path(XML_DIR) / f'{file_id}.xml', Path(XML_DIR) / f'{file_id}.XML']:
        if candidate.exists():
            xml_path = str(candidate); break
            
    if not xml_path:
        hits = list(Path(XML_DIR).glob(f'{file_id}*'))
        if hits: xml_path = str(hits[0])

    # Save processed image array
    cv2.imwrite(os.path.join(IMG_RAW, f'{file_id}.png'), img_arr)

    # Parse and save annotations
    if xml_path:
        masks, bboxes = parse_xml_masses(xml_path, h, w)
        if not masks: no_annot += 1
    else:
        masks, bboxes = [], []; no_xml += 1

    has_mask = len(masks) > 0
    if has_mask:
        combined = np.zeros((h, w), dtype=np.uint8)
        for m in masks: combined = cv2.bitwise_or(combined, m)
        cv2.imwrite(os.path.join(MASK_RAW, f'{file_id}.png'), combined)

    # Export YOLO labels
    with open(os.path.join(LABEL_RAW, f'{file_id}.txt'), 'w') as f:
        for (cx, cy, bw, bh) in bboxes:
            f.write(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')

    processed += 1
    rows.append({'file_id': file_id, 'has_mask': has_mask, 'n_bboxes': len(bboxes)})

print(f'\nPipeline Complete: Processed: {processed} | Contains Mask: {sum(r["has_mask"] for r in rows)}')
print(f'Missing Data - No XML: {no_xml} | No Annotations: {no_annot}')

# ### 3. INbreast Spatial Preprocessing (Align to CBIS format)

PREP_OUT  = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'inbreast_preprocessed')
IMG_PREP  = os.path.join(PREP_OUT, 'images')
LBL_PREP  = os.path.join(PREP_OUT, 'labels')
os.makedirs(IMG_PREP, exist_ok=True)
os.makedirs(LBL_PREP, exist_ok=True)

def crop_borders(img, mask, top_bottom=0.025, sides=0.01):
    # Strip artificial borders and text artifacts from the edges
    h, w = img.shape
    tb, sd = int(h * top_bottom), int(w * sides)
    return img[tb:h-tb, sd:w-sd], mask[tb:h-tb, sd:w-sd]

def orient_to_right(img, mask):
    # Standardize orientation: flip left-facing breasts to the right
    if np.sum(img[:, :img.shape[1]//2]) < np.sum(img[:, img.shape[1]//2:]):
        img, mask = np.fliplr(img), np.fliplr(mask)
    return img, mask

def remove_black_background(img, mask=None, margin=5):
    # Isolate breast tissue by cropping out extreme black backgrounds
    _, binary = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return (img, mask) if mask is not None else img
    
    x, y, w, h = cv2.boundingRect(coords)
    x, y = max(0, x - margin), max(0, y - margin)
    w = min(img.shape[1] - x, w + 2*margin)
    h = min(img.shape[0] - y, h + 2*margin)
    
    return (img[y:y+h, x:x+w], mask[y:y+h, x:x+w]) if mask is not None else img[y:y+h, x:x+w]

def resize_with_padding(img, mask, target=640):
    # Resize while maintaining aspect ratio via zero-padding
    h, w    = img.shape
    scale   = target / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    img  = cv2.resize(img,  (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    pad_h, pad_w = target - new_h, target - new_w
    img  = np.pad(img,  ((0, pad_h), (0, pad_w)), mode='constant')
    mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant')
    return img, mask

def mask_to_yolo(mask, label_path, min_area_ratio=0.0001):
    # Convert binary segmentation components to YOLO bounding boxes
    h, w = mask.shape
    _, binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    num_labels, labels_map = cv2.connectedComponents(binary)
    boxes = []
    
    for i in range(1, num_labels):
        ys, xs = np.where(labels_map == i)
        # Filter negligible noise artifacts
        if len(xs) / (h * w) < min_area_ratio: continue
            
        cx = (xs.min() + xs.max()) / 2 / w
        cy = (ys.min() + ys.max()) / 2 / h
        bw = (xs.max() - xs.min()) / w
        bh = (ys.max() - ys.min()) / h
        boxes.append(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
        
    if not boxes: return False
    
    with open(label_path, 'w') as f: 
        f.write('\n'.join(boxes))
    return True

# --- 3. Pipeline Execution ---
valid_images = 0

for img_name in tqdm([f for f in os.listdir(IMG_RAW) if f.endswith('.png')], desc='Spatial Preprocessing'):
    img_path  = os.path.join(IMG_RAW,  img_name)
    mask_path = os.path.join(MASK_RAW, img_name)
    
    if not os.path.exists(mask_path): continue
        
    img  = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None or mask is None: continue
        
    # Apply Transformations
    img, mask = crop_borders(img, mask)
    img, mask = orient_to_right(img, mask)
    img, mask = remove_black_background(img, mask)
    img, mask = resize_with_padding(img, mask, target=640)
    
    # Save Outputs
    cv2.imwrite(os.path.join(IMG_PREP, img_name), img)
    out_lbl = os.path.join(LBL_PREP, img_name.replace('.png', '.txt'))
    
    if mask_to_yolo(mask, out_lbl): 
        valid_images += 1

print(f'\nPipeline Complete: {valid_images} images successfully preprocessed and annotated.')

# ### 4. Cross-Validation Data Augmentation Strategy

# --- 1. Define Conservative Medical Augmentations ---
mamo_transform = A.Compose([
    # Spatial transformations to introduce variance while preserving tissue integrity
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.05, 
        scale_limit=0.05, 
        rotate_limit=15, 
        border_mode=cv2.BORDER_CONSTANT, 
        value=0, 
        p=0.8
    ),
    # Simulate variations in mammographic X-ray exposure
    A.RandomBrightnessContrast(p=0.4)
], bbox_params=A.BboxParams(
    format='yolo', 
    label_fields=['class_labels'],
    min_area=10, 
    min_visibility=0.2  # Discard bounding box if less than 20% of the mass remains visible
))

# --- 2. Augmentation Execution Function ---
def augment_images(img_dir, lbl_dir, out_img_dir, out_lbl_dir, n_aug=2):
    """
    Copies original images to the target directory and generates `n_aug` 
    augmented versions per image specifically for the training fold.
    """
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    for img_name in os.listdir(img_dir):
        if not img_name.endswith('.png'): continue
            
        img_path = os.path.join(img_dir, img_name)
        txt_path = os.path.join(lbl_dir, img_name.replace('.png', '.txt'))
        if not os.path.exists(txt_path): continue

        # 1. Retain the original (unaugmented) image and label
        shutil.copy(img_path, os.path.join(out_img_dir, img_name))
        shutil.copy(txt_path, os.path.join(out_lbl_dir, img_name.replace('.png', '.txt')))

        # 2. Read image and extract YOLO bounding boxes
        image = cv2.imread(img_path)
        if image is None: continue
            
        bboxes, class_labels = [], []
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_labels.append(int(parts[0]))
                    bboxes.append([float(x) for x in parts[1:]])
                    
        if not bboxes: continue

        # 3. Generate augmented variations
        for i in range(n_aug):
            try:
                aug      = mamo_transform(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_img  = aug['image']
                aug_bbs  = aug['bboxes']
                
                # Skip saving if the augmentation pushed the mass completely out of bounds
                if not aug_bbs: continue
                    
                new_name = img_name.replace('.png', f'_aug{i}.png')
                cv2.imwrite(os.path.join(out_img_dir, new_name), aug_img)
                
                # Save augmented YOLO labels
                with open(os.path.join(out_lbl_dir, new_name.replace('.png', '.txt')), 'w') as f:
                    for bb, cls in zip(aug_bbs, class_labels):
                        f.write(f'{cls} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n')
                        
            except ValueError:
                # Catch albumentations bounding box clipping exceptions and proceed safely
                continue

print(' Cross-Validation Augmentation function initialized.')

# ### 5. 5-Fold Cross-Validation Execution

# --- 1. Dataset Aggregation ---
all_images = sorted([
    f for f in os.listdir(IMG_PREP)
    if f.endswith('.png') and 
    os.path.exists(os.path.join(LBL_PREP, f.replace('.png', '.txt')))
])
print(f'Total valid images for Cross-Validation: {len(all_images)}')

# --- 2. 5-Fold Initialization ---
kf         = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []
CV_BASE    = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'cv_folds')

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_images)):
    print(f'\n{"="*55}')
    print(f' FOLD {fold_idx + 1} / 5   |   Train Split: {len(train_idx)}   |   Val Split: {len(val_idx)}')
    print(f'{"="*55}')

    train_imgs = [all_images[i] for i in train_idx]
    val_imgs   = [all_images[i] for i in val_idx]

    # --- 3. Directory Configuration for Current Fold ---
    fold_dir      = os.path.join(CV_BASE, f'fold_{fold_idx}')
    raw_train_img = os.path.join(fold_dir, 'raw_train', 'images')
    raw_train_lbl = os.path.join(fold_dir, 'raw_train', 'labels')
    aug_train_img = os.path.join(fold_dir, 'images', 'train')
    aug_train_lbl = os.path.join(fold_dir, 'labels', 'train')
    val_img_dir   = os.path.join(fold_dir, 'images', 'val')
    val_lbl_dir   = os.path.join(fold_dir, 'labels', 'val')
    
    for d in [raw_train_img, raw_train_lbl, aug_train_img, aug_train_lbl, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    # --- 4. Populate Training Split ---
    for img in train_imgs:
        shutil.copy(os.path.join(IMG_PREP, img), os.path.join(raw_train_img, img))
        shutil.copy(os.path.join(LBL_PREP, img.replace('.png', '.txt')),
                    os.path.join(raw_train_lbl, img.replace('.png', '.txt')))

    # Apply data augmentation strictly to the training split
    augment_images(raw_train_img, raw_train_lbl, aug_train_img, aug_train_lbl, n_aug=2)
    print(f' Augmented Training Set Size: {len(os.listdir(aug_train_img))} images')

    # --- 5. Populate Validation Split (No Augmentation) ---
    for img in val_imgs:
        shutil.copy(os.path.join(IMG_PREP, img), os.path.join(val_img_dir, img))
        shutil.copy(os.path.join(LBL_PREP, img.replace('.png', '.txt')),
                    os.path.join(val_lbl_dir, img.replace('.png', '.txt')))

    # --- 6. Generate YAML Configuration ---
    yaml_path = os.path.join(fold_dir, 'fold.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f'path: {fold_dir}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ["mass"]\n')

    # --- 7. Model Training ---
    model = YOLO(old_model_path)
    model.train(
        data      = yaml_path,
        epochs    = 50,
        imgsz     = 640,
        batch     = 8,
        optimizer = 'AdamW',
        lr0       = 1e-4,
        lrf       = 0.01,
        freeze    = 5,          # Freeze backbone early layers to preserve base features
        fliplr    = 0.5,
        flipud    = 0.5,
        scale     = 0.1,
        translate = 0.1,
        mosaic    = 0.0,        # Disabled to prevent fragmentation of subtle masses
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        project   = CV_BASE,
        name      = f'fold_{fold_idx}_results',
        verbose   = False
    )

    # --- 8. Fold Evaluation ---
    best_weights = os.path.join(CV_BASE, f'fold_{fold_idx}_results', 'weights', 'best.pt')
    eval_model   = YOLO(best_weights)
    metrics      = eval_model.val(data=yaml_path, split='val', conf=0.25, verbose=False)

    fold_result = {
        'Fold':      fold_idx + 1,
        'Precision': round(metrics.box.mp,    3),
        'Recall':    round(metrics.box.mr,    3),
        'mAP@50':    round(metrics.box.map50, 3),
    }
    cv_results.append(fold_result)
    print(f' [Result] Fold {fold_idx+1} | Precision: {fold_result["Precision"]} | Recall: {fold_result["Recall"]} | mAP: {fold_result["mAP@50"]}')

    # --- 9. Memory Management ---
    # Remove temporary augmented image directories to conserve disk space
    shutil.rmtree(os.path.join(fold_dir, 'raw_train'), ignore_errors=True)
    shutil.rmtree(os.path.join(fold_dir, 'images'),   ignore_errors=True)
    shutil.rmtree(os.path.join(fold_dir, 'labels'),   ignore_errors=True)

# --- 10. Final Statistical Summary ---
df_cv = pd.DataFrame(cv_results)
mean_row = {
    'Fold': 'Mean', 
    'Precision': round(df_cv['Precision'].mean(), 3),
    'Recall': round(df_cv['Recall'].mean(), 3),
    'mAP@50': round(df_cv['mAP@50'].mean(), 3)
}
std_row  = {
    'Fold': 'Std',  
    'Precision': round(df_cv['Precision'].std(),  3),
    'Recall': round(df_cv['Recall'].std(),  3),
    'mAP@50': round(df_cv['mAP@50'].std(),  3)
}

df_cv = pd.concat([df_cv, pd.DataFrame([mean_row, std_row])], ignore_index=True)
print('\n[SUMMARY] 5-Fold Cross-Validation Results on INbreast Dataset:')
print('-' * 65)
print(df_cv.to_string(index=False))
print('-' * 65)

# ### 6. Final Production Fine-Tuning (Full INbreast Dataset)

FINAL_DIR     = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'final_finetune')
FINAL_IMG_TR  = os.path.join(FINAL_DIR, 'images', 'train')
FINAL_LBL_TR  = os.path.join(FINAL_DIR, 'labels', 'train')
FINAL_IMG_VAL = os.path.join(FINAL_DIR, 'images', 'val')
FINAL_LBL_VAL = os.path.join(FINAL_DIR, 'labels', 'val')
for d in [FINAL_IMG_TR, FINAL_LBL_TR, FINAL_IMG_VAL, FINAL_LBL_VAL]:
    os.makedirs(d, exist_ok=True)

# --- 2. Train Split: Full Dataset with Augmentation ---
# Apply the conservative medical augmentation pipeline to the entire dataset
augment_images(IMG_PREP, LBL_PREP, FINAL_IMG_TR, FINAL_LBL_TR, n_aug=2)
print(f'Final Training Set Size (with augmentation): {len(os.listdir(FINAL_IMG_TR))} images')

# --- 3. Validation Split: Unaugmented Originals ---
# Utilized strictly to monitor training convergence prior to external benchmarking
for img_name in all_images:
    shutil.copy(os.path.join(IMG_PREP, img_name), os.path.join(FINAL_IMG_VAL, img_name))
    shutil.copy(os.path.join(LBL_PREP, img_name.replace('.png', '.txt')),
                os.path.join(FINAL_LBL_VAL, img_name.replace('.png', '.txt')))

# --- 4. Configuration YAML ---
final_yaml = os.path.join(FINAL_DIR, 'inbreast_full.yaml')
with open(final_yaml, 'w') as f:
    f.write(f'path: {FINAL_DIR}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ["mass"]\n')

# --- 5. Production Model Training ---
print('\n[INFO] Initiating Final Fine-Tuning on the complete INbreast dataset...')
model = YOLO(old_model_path)

model.train(
    data         = final_yaml,
    epochs       = 60,          # Extended epochs for final convergence
    imgsz        = 640,
    batch        = 8,
    optimizer    = 'AdamW',
    lr0          = 1e-4,
    lrf          = 0.01,
    weight_decay = 0.001,       # Added L2 regularization to prevent overfitting on the full dataset
    freeze       = 5,           # Retain foundational domain-specific features
    fliplr       = 0.5,
    flipud       = 0.5,
    scale        = 0.1,
    translate    = 0.1,
    mosaic       = 0.0,         # Disabled to preserve true medical image architecture
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    project      = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'final_results'),
    name         = 'inbreast_final_model'
)

FINAL_MODEL = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'final_results/inbreast_final_model/weights/best.pt')
print(f'\n[SUCCESS] Final production model weights saved at: {FINAL_MODEL}')

# ### 7. prepare BCDR dataset for Evaluation 

BCDR_IMG_DIR = os.environ.get('BCDR_IMG_PATH', 'data/BCDR/original/BCDR-Original-Preprocessed-IMG')
BCDR_MSK_DIR = os.environ.get('BCDR_MSK_PATH', 'data/BCDR/original/BCDR-Original-Preprocessed-MSK')
CSV_D01      = os.path.join(os.environ.get('BCDR_CSV_PATH', 'data/BCDR/original/csv'), 'bcdr_d01_outlines.csv')
CSV_D02      = os.path.join(os.environ.get('BCDR_CSV_PATH', 'data/BCDR/original/csv'), 'bcdr_d02_outlines.csv')
BCDR_OUT     = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'bcdr_yolo')

# ── Mass keys from CSV ────────────────────────────────────────────────────
d01 = pd.read_csv(CSV_D01)
d02 = pd.read_csv(CSV_D02)
d02 = d02[d02['mammography_nodule'] == 1]
df  = pd.concat([d01, d02], ignore_index=True)

def csv_to_key(path):
    name = str(path).strip().replace('/', '_')
    return os.path.splitext(name)[0]

df['key']  = df['image_filename'].apply(csv_to_key)
mass_keys  = df['key'].unique().tolist()
print(f'Mass keys from CSV: {len(mass_keys)}')

# ── Index files ───────────────────────────────────────────────────────────
img_index = {f.replace('___PRE.png', ''): os.path.join(BCDR_IMG_DIR, f)
             for f in os.listdir(BCDR_IMG_DIR) if f.endswith('.png')}
msk_index = {f.replace('_MASK___PRE.png', ''): os.path.join(BCDR_MSK_DIR, f)
             for f in os.listdir(BCDR_MSK_DIR) if f.endswith('.png')}
print(f'Images: {len(img_index)} | Masks: {len(msk_index)}')

# ── Extract YOLO labels ───────────────────────────────────────────────────
def mask_to_yolo_bcdr(mask, min_area_ratio=0.0001):
    h, w = mask.shape
    _, binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    num, labels_map = cv2.connectedComponents(binary)
    boxes = []
    for i in range(1, num):
        ys, xs = np.where(labels_map == i)
        if len(xs) / (h * w) < min_area_ratio: continue
        cx = (xs.min() + xs.max()) / 2 / w
        cy = (ys.min() + ys.max()) / 2 / h
        bw = (xs.max() - xs.min()) / w
        bh = (ys.max() - ys.min()) / h
        boxes.append(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
    return boxes

tmp_img = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'bcdr_tmp/images')
tmp_lbl = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'bcdr_tmp/labels')
os.makedirs(tmp_img, exist_ok=True)
os.makedirs(tmp_lbl, exist_ok=True)

valid, not_found, no_label = [], [], []
for key in tqdm(mass_keys, desc='Processing BCDR'):
    ip = img_index.get(key)
    mp = msk_index.get(key)
    if ip is None or mp is None: not_found.append(key); continue
    mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.max() == 0: no_label.append(key); continue
    boxes = mask_to_yolo_bcdr(mask)
    if not boxes: no_label.append(key); continue
    shutil.copy(ip, os.path.join(tmp_img, key + '.png'))
    with open(os.path.join(tmp_lbl, key + '.txt'), 'w') as f:
        f.write('\n'.join(boxes))
    valid.append(key)

print(f'\n Valid: {len(valid)} | Not found: {len(not_found)} | No label: {len(no_label)}')

# ── Split 70/15/15 ────────────────────────────────────────────────────────
train_b, temp = train_test_split(valid, test_size=0.30, random_state=42)
val_b,   test_b = train_test_split(temp, test_size=0.50, random_state=42)
print(f'BCDR → Train: {len(train_b)} | Val: {len(val_b)} | Test: {len(test_b)}')

for split_name, split_files in [('train', train_b), ('val', val_b), ('test', test_b)]:
    os.makedirs(f'{BCDR_OUT}/images/{split_name}', exist_ok=True)
    os.makedirs(f'{BCDR_OUT}/labels/{split_name}', exist_ok=True)
    for key in split_files:
        shutil.copy(f'{tmp_img}/{key}.png',  f'{BCDR_OUT}/images/{split_name}/{key}.png')
        shutil.copy(f'{tmp_lbl}/{key}.txt',  f'{BCDR_OUT}/labels/{split_name}/{key}.txt')

print(' BCDR YOLO dataset ready!')

# ### 8. External Validation Tests

def evaluate_on_dataset(model, images_dir, label_dir, name):
    """
    Evaluates the model on an external dataset by sweeping across multiple 
    confidence thresholds to map the precision-recall performance and identify 
    the optimal operational threshold.
    """
    # Dynamically generate dataset YAML configuration
    yaml_content = f'train: {images_dir}\nval: {images_dir}\nnc: 1\nnames: ["mass"]\n'
    yaml_path    = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), f'eval_{name}.yaml')
    
    with open(yaml_path, 'w') as f: 
        f.write(yaml_content)

    # Define confidence threshold spectrum for rigorous trade-off analysis
    thresholds   = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    results_list = []
    
    for conf in thresholds:
        m = model.val(data=yaml_path, split='val', conf=conf, verbose=False, plots=False)
        
        results_list.append({
            'Conf': f'{conf:.2f}',
            'Precision': round(m.box.mp,    3),
            'Recall':    round(m.box.mr,    3),
            'mAP@50':    round(m.box.map50, 3)
        })

    # Compile and display evaluation metrics
    df = pd.DataFrame(results_list)
    print(f'\n[RESULTS] Performance on {name}:')
    print('-' * 60)
    print(df.to_string(index=False))
    print('-' * 60)
    
    # Extract the optimal threshold based on maximum mAP@50
    best = df.loc[df['mAP@50'].idxmax()]
    print(f' Optimal Configuration: conf={best["Conf"]} | mAP@50={best["mAP@50"]} | P={best["Precision"]} | R={best["Recall"]}')
    
    return df

# --- External Validation 1: BCDR Test Set ---
df_bcdr = evaluate_on_dataset(
    final_model,
    f'{BCDR_OUT}/images/test',
    f'{BCDR_OUT}/labels/test',
    'BCDR_test'
)

# ── Test 2: VinDr ──────────────────────────────────────────────────────────
VINDR_ROOT = os.environ.get('VINDR_PATH', 'data/VinDr/dataset_yolo')
df_vindr   = evaluate_on_dataset(
    final_model,
    f'{VINDR_ROOT}/images/val',
    f'{VINDR_ROOT}/labels/val',
    'VinDr_val'
)

# ── Test 3: CBIS-DDSM (Catastrophic Forgetting check) ─────────────────────
RAW_CBIS_DIR = os.environ.get('CBIS_TEST_PATH', 'data/CBIS-DDSM/organized/test')
CBIS_OUT     = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'cbis_ready_for_test')
CBIS_IMG_OUT = os.path.join(CBIS_OUT, 'images/test')
CBIS_LBL_OUT = os.path.join(CBIS_OUT, 'labels/test')
os.makedirs(CBIS_IMG_OUT, exist_ok=True)
os.makedirs(CBIS_LBL_OUT, exist_ok=True)

def apply_clahe(img):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)

def preprocess_cbis(img, mask):
    h, w   = img.shape
    tb, sd = int(h * 0.025), int(w * 0.01)
    img, mask = img[tb:h-tb, sd:w-sd], mask[tb:h-tb, sd:w-sd]
    _, binary = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, wc, hc = cv2.boundingRect(coords)
        img, mask = img[y:y+hc, x:x+wc], mask[y:y+hc, x:x+wc]
    img = apply_clahe(img)
    target = 640
    scale  = target / max(img.shape)
    img    = cv2.resize(img,  (int(img.shape[1]*scale), int(img.shape[0]*scale)))
    mask   = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    ph, pw = target - img.shape[0], target - img.shape[1]
    img    = np.pad(img,  ((0, ph), (0, pw)), mode='constant')
    mask   = np.pad(mask, ((0, ph), (0, pw)), mode='constant')
    return img, mask

image_paths = list(Path(RAW_CBIS_DIR).rglob('full.jpg'))
for img_path in tqdm(image_paths, desc='Processing CBIS'):
    img         = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    mask_files  = list(img_path.parent.glob('mask_*.jpg'))
    if not mask_files: continue
    combined_mask = np.zeros_like(img)
    skip = False
    for mf in mask_files:
        m = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
        if m.shape != img.shape: skip = True; break
        combined_mask = cv2.bitwise_or(combined_mask, m)
    if skip: continue
    img, mask = preprocess_cbis(img, combined_mask)
    new_name  = f'{img_path.parent.name}_{img_path.name}'
    cv2.imwrite(os.path.join(CBIS_IMG_OUT, new_name), img)
    _, binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    num, labels = cv2.connectedComponents(binary)
    with open(os.path.join(CBIS_LBL_OUT, new_name.replace('.jpg', '.txt')), 'w') as f:
        for i in range(1, num):
            ys, xs = np.where(labels == i)
            x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
            h, w = mask.shape
            f.write(f'0 {(x_min+x_max)/2/w:.6f} {(y_min+y_max)/2/h:.6f} '
                    f'{(x_max-x_min)/w:.6f} {(y_max-y_min)/h:.6f}\n')

df_cbis = evaluate_on_dataset(final_model, CBIS_IMG_OUT, CBIS_LBL_OUT, 'CBIS_test')

# ### 9. Final Summary Table

print('\n' + '='*65)
print('  FINAL RESULTS SUMMARY')
print('='*65)

# CV Results
print('\n 5-Fold CV on INbreast:')
print(df_cv.to_string(index=False))

# External Test Results (best conf per dataset)
print('\n External Validation (best confidence threshold):')
summary_rows = []
for name, df in [('BCDR test', df_bcdr), ('VinDr val', df_vindr), ('CBIS test', df_cbis)]:
    best = df.loc[df['mAP@50'].idxmax()]
    summary_rows.append({
        'Dataset':   name,
        'Best Conf': best['Conf'],
        'Precision': best['Precision'],
        'Recall':    best['Recall'],
        'mAP@50':    best['mAP@50']
    })

df_summary = pd.DataFrame(summary_rows)
print(df_summary.to_string(index=False))
print('='*65)

# ### 10. Visualization — Predictions vs Ground Truth

def visualize_predictions(model, img_dir, lbl_dir, title, n_samples=10, conf=0.25):
    images  = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))][:n_samples]
    cols    = 5
    rows    = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes    = axes.flatten()

    for idx, img_name in enumerate(images):
        img_path = os.path.join(img_dir, img_name)
        txt_path = os.path.join(lbl_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        img = cv2.imread(img_path)
        if img is None: continue
        h, w, _ = img.shape

# Ground Truth (green)
        if os.path.exists(txt_path):
            with open(txt_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cx, cy, bw, bh = map(float, parts[1:])
                        x1, y1 = int((cx-bw/2)*w), int((cy-bh/2)*h)
                        x2, y2 = int((cx+bw/2)*w), int((cy+bh/2)*h)
                        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 5) 
                        cv2.putText(img, 'GT', (x1, y1-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 3)
        # Predictions (red)
        results = model.predict(source=img_path, conf=conf, verbose=False)
        for box in results[0].boxes:
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            c = float(box.conf[0])
            cv2.rectangle(img, (bx1,by1), (bx2,by2), (0,0,255), 4) 
            cv2.putText(img, f'P:{c:.2f}', (bx1, by2+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 3) 
        axes[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(img_name[:20], fontsize=9)
        axes[idx].axis('off')

    for i in range(len(images), len(axes)): axes[i].axis('off')
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), f'viz_{title.replace(" ","_")}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f' Saved: {out_path}')

# Visualize on BCDR test
visualize_predictions(
    final_model,
    f'{BCDR_OUT}/images/test',
    f'{BCDR_OUT}/labels/test',
    title='BCDR Test — Green=GT | Red=Predicted',
    n_samples=10
)

