
# ------------------------------------------------------------
# # Breast Mass Segmentation: UNet on CBIS-DDSM → Fine-Tuned on INbreast

# ------------------------------------------------------------
# ## Part 1 — Dependencies & Imports

# ------------------------------------------------------------
# ### 1.1 Install packages

# !pip install matplotlib huggingface_hub pydicom scikit-learn -q
# ------------------------------------------------------------
# ### 1.2 Core imports

import os, re, shutil, cv2, random, gc, math
import numpy as np
import pandas as pd
import pydicom
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import KFold
from huggingface_hub import hf_hub_download, login, HfApi

print(' Libraries ready!')

# ── Config ──────────────────────────────────────────────────────
CROP_SIZE    = 128     
SEG_BATCH    = 16
SEG_EPOCHS   = 160
SEG_PATIENCE = 25
SEG_LR_INIT  = 1e-4   
SEG_CROP_ROOT = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'seg_crops_128')
UNET_BEST     = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'unet_mobilev3_best.pth')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gc.collect(); torch.cuda.empty_cache()

unet_model = smp.Unet(
    encoder_name='timm-mobilenetv3_small_075',
    encoder_weights='imagenet',
    in_channels=1,
    classes=1,
    activation=None,
)
unet_model = unet_model.to(DEVICE)

total = sum(p.numel() for p in unet_model.parameters())
print(f'UNet MobileNetV3-Small-0.75: {total/1e6:.1f}M parameters')
print(f'Paper result: Dice=90.4%, IoU=82.8% on CBIS-DDSM')

# ------------------------------------------------------------
# ### 2.2 Dataset paths

TRAIN_IMGS = os.environ.get('CBIS_TRAIN_IMGS', 'data/CBIS-DDSM/preprocessed/train_imgs')
TRAIN_MASKS = os.environ.get('CBIS_TRAIN_MASKS', 'data/CBIS-DDSM/preprocessed/train_masks')

TEST_IMGS = os.environ.get('CBIS_TEST_IMGS', 'data/CBIS-DDSM/preprocessed/test_imgs')
TEST_MASKS = os.environ.get('CBIS_TEST_MASKS', 'data/CBIS-DDSM/preprocessed/test_masks')

VAL_IMGS = os.environ.get('CBIS_VAL_IMGS', 'data/CBIS-DDSM/preprocessed/val_imgs')
VAL_MASKS = os.environ.get('CBIS_VAL_MASKS', 'data/CBIS-DDSM/preprocessed/val_masks')

# ------------------------------------------------------------
# ### 2.3 Data preparation — GT crop extraction
# 
# Extract 128×128 crops around each mass using the ground-truth bounding box from the mask.  
# A small padding (`pad_ratio=0.05`) is added around each bounding box.
# 

CROP_SIZE   = 128
SEG_BATCH   = 16
SEG_CROP_ROOT = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'seg_crops_128')

# ── Step 1: Extract GT crops directly from the masks ──────────────
def extract_gt_crops(img_dir, mask_dir, out_dir, split, pad_ratio=0.05):
    """
    Takes the GT bounding box from the mask and creates a 128×128 crop
    This exactly matches the paper methodology
    """
    out_imgs  = os.path.join(out_dir, split, 'images')
    out_masks = os.path.join(out_dir, split, 'masks')
    os.makedirs(out_imgs,  exist_ok=True)
    os.makedirs(out_masks, exist_ok=True)

    files   = sorted([f for f in os.listdir(img_dir)
                      if f.lower().endswith(('.jpg', '.png'))])
    saved = skipped = 0

    for fname in tqdm(files, desc=f'[{split}] GT crops'):
        img_path  = os.path.join(img_dir,  fname)
        mask_path = os.path.join(mask_dir, fname)
        if not os.path.exists(mask_path): skipped += 1; continue

        img  = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None: skipped += 1; continue

        H, W = img.shape
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Extract all connected components (masses)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8), connectivity=8
        )

        for comp_idx in range(1, n_labels):
            area = stats[comp_idx, cv2.CC_STAT_AREA]
            if area < 50: continue  # Skip small noise components

            # GT bounding box
            x = stats[comp_idx, cv2.CC_STAT_LEFT]
            y = stats[comp_idx, cv2.CC_STAT_TOP]
            w = stats[comp_idx, cv2.CC_STAT_WIDTH]
            h = stats[comp_idx, cv2.CC_STAT_HEIGHT]

            # Small padding around the bounding box
            pad = int(pad_ratio * max(w, h))
            x1 = max(0, x - pad);  y1 = max(0, y - pad)
            x2 = min(W, x+w+pad);  y2 = min(H, y+h+pad)

            crop_img  = img[y1:y2, x1:x2]
            crop_mask = mask[y1:y2, x1:x2]

            if crop_img.size == 0: continue

            # Resize to 128×128
            crop_img  = cv2.resize(crop_img,  (CROP_SIZE, CROP_SIZE),
                                   interpolation=cv2.INTER_LINEAR)
            crop_mask = cv2.resize(crop_mask, (CROP_SIZE, CROP_SIZE),
                                   interpolation=cv2.INTER_NEAREST)
            _, crop_mask = cv2.threshold(crop_mask, 127, 255, cv2.THRESH_BINARY)

            if crop_mask.max() == 0: continue

            stem = fname.rsplit('.', 1)[0]
            name = f'{stem}_m{comp_idx}.png'
            cv2.imwrite(os.path.join(out_imgs,  name), crop_img)
            cv2.imwrite(os.path.join(out_masks, name), crop_mask)
            saved += 1

    print(f'  [{split}] saved={saved} | skipped={skipped}')

# Run extraction
for split, idir, mdir in [
    ('train', TRAIN_IMGS, TRAIN_MASKS),
    ('val',   VAL_IMGS,   VAL_MASKS),
    ('test',  TEST_IMGS,  TEST_MASKS),
]:
    extract_gt_crops(idir, mdir, SEG_CROP_ROOT, split)

for s in ['train', 'val', 'test']:
    p = f'{SEG_CROP_ROOT}/{s}/images'
    if os.path.exists(p):
        print(f'  {s}: {len(os.listdir(p))} crops')

# ── Step 2: Dataset class ──────────────────────────────────────────
seg_train_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Affine(translate_percent={'x':(-0.05,0.05), 'y':(-0.05,0.05)},
             scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
    A.ElasticTransform(alpha=20, sigma=4, p=0.2),
    A.RandomGamma(gamma_limit=(85, 115), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.1,
                               contrast_limit=0.1, p=0.3),
    A.GaussNoise(var_limit=(1.0, 8.0), p=0.2),
    A.Normalize(mean=[0.5], std=[0.5]),
    ToTensorV2(),
])

seg_val_aug = A.Compose([
    A.Normalize(mean=[0.5], std=[0.5]),
    ToTensorV2(),
])

class SegCropDataset(Dataset):
    def __init__(self, split, transform):
        self.img_dir  = f'{SEG_CROP_ROOT}/{split}/images'
        self.msk_dir  = f'{SEG_CROP_ROOT}/{split}/masks'
        self.files    = sorted(os.listdir(self.img_dir))
        self.transform = transform

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img  = cv2.imread(os.path.join(self.img_dir, fname),
                          cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(self.msk_dir, fname),
                          cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        out  = self.transform(image=img, mask=mask)
        return out['image'].float(), out['mask'].unsqueeze(0).float()

seg_train_ds = SegCropDataset('train', seg_train_aug)
seg_val_ds   = SegCropDataset('val',   seg_val_aug)

seg_train_dl = DataLoader(seg_train_ds, batch_size=SEG_BATCH,
                          shuffle=True, num_workers=4, pin_memory=True)
seg_val_dl   = DataLoader(seg_val_ds,   batch_size=SEG_BATCH,
                          shuffle=False, num_workers=4, pin_memory=True)

print(f'Train crops: {len(seg_train_ds)}')
print(f'Val   crops: {len(seg_val_ds)}')

# ------------------------------------------------------------
# ### 2.4 Loss function, optimizer & training loop
# 
# - **DiceLoss** (smooth=1.0)   
# - **Adam** optimizer with **ReduceLROnPlateau** scheduler  
# - Model checkpoint saved whenever validation Dice improves
# 

#  1. Combined function to compute Dice and IoU efficiently 
def calculate_metrics(logits, targets, thr=0.5):
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * targets).sum(dim=(2,3))
    
    # Compute denominators
    union_dice = pred.sum(dim=(2,3)) + targets.sum(dim=(2,3))
    union_iou = union_dice - inter
    
    # Add 1 to avoid division by zero (smoothing)
    dice = ((2 * inter + 1) / (union_dice + 1)).mean().item()
    iou = ((inter + 1) / (union_iou + 1)).mean().item()
    
    return dice, iou

#  Loss: Dice only, same as the paper 
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        prob  = torch.sigmoid(logits)
        inter = (prob * targets).sum(dim=(2,3))
        union = prob.sum(dim=(2,3)) + targets.sum(dim=(2,3))
        return (1 - (2*inter+self.smooth)/(union+self.smooth)).mean()

unet_criterion = DiceLoss()

#  Optimizer + Scheduler 
unet_opt = torch.optim.Adam(unet_model.parameters(), lr=SEG_LR_INIT)
unet_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
    unet_opt, mode='max', factor=0.6, patience=5
)

best_dice  = 0.0
no_improve = 0

print('='*70)
print(' Training UNet MobileNetV3-Small-0.75 on 128×128 crops')
print('='*70)

for epoch in range(1, SEG_EPOCHS+1):
    
    #  TRAIN PHASE 
    unet_model.train()
    tl = td = tiou = 0.0
    
    # Create training progress bar
    pbar_train = tqdm(seg_train_dl, desc=f"Epoch {epoch:03d}/{SEG_EPOCHS:03d} [TRAIN]", leave=False)
    
    for imgs, masks in pbar_train:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        
        unet_opt.zero_grad()
        logits = unet_model(imgs)
        loss = unet_criterion(logits, masks)
        
        loss.backward()
        unet_opt.step()
        
        # Compute metrics (using detach to avoid affecting gradients)
        dice, iou = calculate_metrics(logits.detach(), masks)
        
        tl += loss.item()
        td += dice
        tiou += iou
        
        # Update progress bar live
        pbar_train.set_postfix({'Loss': f'{loss.item():.4f}', 'Dice': f'{dice:.4f}', 'IoU': f'{iou:.4f}'})
        
    tl /= len(seg_train_dl)
    td /= len(seg_train_dl)
    tiou /= len(seg_train_dl)

    #  VALIDATION PHASE 
    unet_model.eval()
    vl = vd = viou = 0.0
    
    # Create validation progress bar
    pbar_val = tqdm(seg_val_dl, desc=f"Epoch {epoch:03d}/{SEG_EPOCHS:03d} [VAL]  ", leave=False)
    
    with torch.no_grad():
        for imgs, masks in pbar_val:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            
            logits = unet_model(imgs)
            loss = unet_criterion(logits, masks)
            
            dice, iou = calculate_metrics(logits, masks)
            
            vl += loss.item()
            vd += dice
            viou += iou
            
            pbar_val.set_postfix({'Loss': f'{loss.item():.4f}', 'Dice': f'{dice:.4f}', 'IoU': f'{iou:.4f}'})
            
    vl /= len(seg_val_dl)
    vd /= len(seg_val_dl)
    viou /= len(seg_val_dl)
    
    # Update scheduler based on validation Dice
    unet_sch.step(vd)

    #  CHECKPOINT & EARLY STOPPING 
    flag = ''
    if vd > best_dice:
        best_dice = vd
        no_improve = 0
        flag = '  BEST'
        torch.save({
            'model_state': unet_model.state_dict(),
            'val_dice': vd, 
            'val_iou': viou,
            'epoch': epoch
        }, UNET_BEST)
    else:
        no_improve += 1

    # Print epoch summary after tqdm progress bar closes
    print(f"Ep {epoch:03d} | "
          f"TRAIN: Loss={tl:.4f}, Dice={td:.4f}, IoU={tiou:.4f} | "
          f"VAL: Loss={vl:.4f}, Dice={vd:.4f}, IoU={viou:.4f}{flag}")

    if no_improve >= SEG_PATIENCE:
        print(f'\n Early stopping triggered at epoch {epoch} (No improvement for {SEG_PATIENCE} epochs)')
        break

print(f'\n Training Completed! Best Val Dice: {best_dice:.4f}')

# ------------------------------------------------------------
# ### 2.5 Qualitative visualization — training predictions
# 
# Overlay view: Original ROI | Ground Truth | Prediction

def visualize_predictions(model, image_dir, mask_dir, dataset_name="Test", num_samples=4, device='cuda'):
    model.eval()
    if not os.path.exists(image_dir):
        print(f"Directory not found: {image_dir}")
        return

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))][:num_samples]
    if not image_files:
        print(f"No images found in {image_dir}")
        return

    fig, axes = plt.subplots(len(image_files), 3, figsize=(12, 4 * len(image_files)))
    if len(image_files) == 1:
        axes = [axes]

    print(f"\n Visualizing Predictions for {dataset_name}...")
    
    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or gt_mask is None:
            continue

        img_resized = cv2.resize(img, (128, 128))
        gt_resized = cv2.resize(gt_mask, (128, 128))

        img_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            pred = model(img_tensor)
            pred = torch.sigmoid(pred)
            pred_mask = (pred > 0.5).float().cpu().numpy()[0, 0] * 255
            pred_mask = pred_mask.astype(np.uint8)

        axes[idx][0].imshow(img_resized, cmap='gray')
        axes[idx][0].set_title(f"ROI ({img_name[:10]}...)")
        axes[idx][0].axis('off')

        axes[idx][1].imshow(gt_resized, cmap='gray')
        axes[idx][1].set_title("Ground Truth Mask", color='green')
        axes[idx][1].axis('off')

        axes[idx][2].imshow(pred_mask, cmap='gray')
        axes[idx][2].set_title("U-Net Prediction", color='blue')
        axes[idx][2].axis('off')

    plt.tight_layout()
    plt.show()

TEST_IMGS_DIR  = os.environ.get('CBIS_TEST_IMGS', 'data/CBIS-DDSM/preprocessed/test_imgs')
TEST_MASKS_DIR = os.environ.get('CBIS_TEST_MASKS', 'data/CBIS-DDSM/preprocessed/test_masks')

visualize_predictions(
    model=unet_model, 
    image_dir=TEST_IMGS_DIR, 
    mask_dir=TEST_MASKS_DIR, 
    dataset_name="CBIS-DDSM Test Set", 
    num_samples=3, 
    device=DEVICE
)
# ------------------------------------------------------------
# 
# ## Part 3 — INbreast Fine-Tuning

# ------------------------------------------------------------
# ### 3.1 Configuration & paths

# ── Pipeline Config ─────────────────────────────────────────────────
CROP_SIZE     = 128       
SEG_BATCH     = 16
CV_EPOCHS     = 80        
FINAL_EPOCHS  = 100       # Final fine-tune after cross-validation
SEG_PATIENCE  = 20
SEG_LR_INIT   = 5e-5     # Smaller LR than original (1e-4) for fine-tuning
FREEZE_EPOCHS = 10        
N_AUG         = 2         # Same augmentation factor as the detection notebook

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ── Paths ───────────────────────────────────────────────────────────
DATASET_BASE  = os.environ.get('INBREAST_PATH', 'data/INbreast/INbreast Release 1.0')
DICOM_DIR     = os.path.join(DATASET_BASE, 'AllDICOMs')
XML_DIR       = os.path.join(DATASET_BASE, 'AllXML')
CSV_FILE      = os.path.join(DATASET_BASE, 'INbreast.csv')
XLS_FILE      = os.path.join(DATASET_BASE, 'INbreast.xls')

RAW_OUT       = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'inbreast_raw_seg')
IMG_RAW       = os.path.join(RAW_OUT, 'images')
MASK_RAW      = os.path.join(RAW_OUT, 'masks')

PREP_OUT      = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'inbreast_prep_seg')
IMG_PREP      = os.path.join(PREP_OUT, 'images')
MASK_PREP     = os.path.join(PREP_OUT, 'masks')

SEG_CROP_ROOT = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'inbreast_crops_128')
CV_BASE       = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'cv_seg_folds')
FINAL_DIR     = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'final_seg_finetune')
UNET_BEST_CV  = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'unet_inbreast_cv_best.pth')
UNET_FINAL    = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'unet_inbreast_final.pth')

for d in [IMG_RAW, MASK_RAW, IMG_PREP, MASK_PREP]:
    os.makedirs(d, exist_ok=True)

print(' Config ready!')

# ------------------------------------------------------------
# ### 3.2 Download pre-trained weights from HuggingFace

REPO_ID        = 'hab200/Breast-Cancer-Mammogram-Detection-Segmentation'
UNET_FILENAME  = 'models/unet_mobilev3_best.pth'

pretrained_ckpt = hf_hub_download(repo_id=REPO_ID, filename=UNET_FILENAME)
print(f' UNet downloaded: {pretrained_ckpt}')

# ------------------------------------------------------------
# ### 3.3 INbreast preprocessing helpers

# ------------------------------------------------------------
# Parse XML annotations, read DICOM files, apply CLAHE, and extract mass polygons.
# 

# ── Helper Functions (same as the detection notebook) ───────────────

def parse_point(s):
    nums = re.findall(r'[-+]?\d*\.?\d+', s)
    return (int(round(float(nums[0]))), int(round(float(nums[1])))) if len(nums) >= 2 else None

def read_plist_dict(d):
    result, children, i = {}, list(d), 0
    while i < len(children) - 1:
        if children[i].tag == 'key':
            result[children[i].text] = children[i + 1]; i += 2
        else:
            i += 1
    return result

def get_mass_ids(csv_file, xls_file):
    try:
        csv_df = pd.read_csv(csv_file, sep=';')
    except:
        csv_df = pd.read_csv(csv_file)
    csv_df.columns = [c.strip() for c in csv_df.columns]
    csv_df['fn'] = csv_df['File Name'].astype(str).str.strip()
    xls_df = pd.read_excel(xls_file, engine='xlrd')
    xls_df.columns = [c.strip() for c in xls_df.columns]
    mass_col = next(c for c in xls_df.columns if c.strip().lower() == 'mass')
    xls_df['fn'] = xls_df['File Name'].apply(lambda x: str(int(x)) if pd.notna(x) else '')
    merged = csv_df.merge(xls_df[['fn', mass_col]], on='fn', how='left')
    ids = set(merged[merged[mass_col].astype(str).str.strip() == 'X']['fn'].tolist())
    print(f'  Mass IDs: {len(ids)}')
    return ids

def build_dicom_index(dicom_dir):
    index = {dcm.stem.split('_')[0]: str(dcm) for dcm in Path(dicom_dir).rglob('*.dcm')}
    print(f'  DICOM index: {len(index)} files')
    return index

def dicom_to_png(dcm_path):
    ds  = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)
    if getattr(ds, 'PhotometricInterpretation', '') == 'MONOCHROME1':
        arr = arr.max() - arr
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    arr = arr.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(arr), ds.Rows, ds.Columns

def parse_xml_masks(xml_path, img_h, img_w):
    """
    Extracts polygon annotations from the XML and converts them to binary masks.
    Same as parse_xml_masses in the detection notebook but returns masks only (not bboxes).
    """
    try:
        content = re.sub(r'<!DOCTYPE[^>]*>', '', open(xml_path, encoding='utf-8', errors='ignore').read())
        root = ET.fromstring(content)
    except ET.ParseError as e:
        print(f'  [XML ERROR] {xml_path}: {e}')
        return []

    masks = []
    root_kv = read_plist_dict(root.find('dict'))
    images_array = root_kv.get('Images')
    if images_array is None:
        return []

    for image_dict in images_array.findall('dict'):
        rois_array = read_plist_dict(image_dict).get('ROIs')
        if rois_array is None:
            continue
        for roi_dict in rois_array.findall('dict'):
            roi_kv    = read_plist_dict(roi_dict)
            name_node = roi_kv.get('Name')
            if name_node is None:
                continue
            name = (name_node.text or '').lower()
            if not any(k in name for k in ['mass', 'nódulo', 'nodulo', 'spiculated']):
                continue
            point_node = roi_kv.get('Point_px')
            if point_node is None:
                continue
            pts = [list(p) for s in point_node.findall('string')
                   if (p := parse_point(s.text or ''))]
            if len(pts) < 3:
                continue
            pts_np = np.array(pts, dtype=np.int32)
            mask   = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts_np], 255)
            masks.append(mask)

    return masks

print(' Helper functions ready!')

# ------------------------------------------------------------
# ### 3.4 DICOM + XML → PNG + mask conversion

# ── Main Preprocessing: DICOM + XML → PNG + Mask ───────────────────
print('=' * 60)
print('INbreast Segmentation Preprocessing')
print('=' * 60)

mass_ids    = get_mass_ids(CSV_FILE, XLS_FILE)
dicom_index = build_dicom_index(DICOM_DIR)
print(f'  Matched: {len(mass_ids & set(dicom_index.keys()))}/{len(mass_ids)}')

processed, no_xml, no_annot = 0, 0, 0

for file_id in tqdm(sorted(mass_ids), desc='Processing INbreast'):
    dcm_path = dicom_index.get(file_id)
    if not dcm_path:
        continue
    try:
        img_arr, h, w = dicom_to_png(dcm_path)
    except Exception as e:
        print(f'  [SKIP] {file_id}: {e}'); continue

    # Search for the XML file
    xml_path = None
    for candidate in [Path(XML_DIR) / f'{file_id}.xml', Path(XML_DIR) / f'{file_id}_1.xml']:
        if candidate.exists():
            xml_path = str(candidate); break
    if not xml_path:
        hits = list(Path(XML_DIR).glob(f'{file_id}*'))
        if hits: xml_path = str(hits[0])

    cv2.imwrite(os.path.join(IMG_RAW, f'{file_id}.png'), img_arr)

    if xml_path:
        masks = parse_xml_masks(xml_path, h, w)
        if not masks:
            no_annot += 1
        else:
            # Merge all mass masks into a single combined mask
            combined = np.zeros((h, w), dtype=np.uint8)
            for m in masks:
                combined = cv2.bitwise_or(combined, m)
            cv2.imwrite(os.path.join(MASK_RAW, f'{file_id}.png'), combined)
    else:
        no_xml += 1

    processed += 1

print(f'\n Processed: {processed} | With mask: {len(os.listdir(MASK_RAW))}')
print(f'  No XML: {no_xml} | No annotation: {no_annot}')

# ------------------------------------------------------------
# ### 3.5 Spatial preprocessing
# 
# 

def crop_borders(img, mask, top_bottom=0.025, sides=0.01):
    h, w = img.shape
    tb, sd = int(h * top_bottom), int(w * sides)
    return img[tb:h-tb, sd:w-sd], mask[tb:h-tb, sd:w-sd]

def orient_to_right(img, mask):
    """Always orient the breast to the right for consistency."""
    if np.sum(img[:, :img.shape[1]//2]) < np.sum(img[:, img.shape[1]//2:]):
        img, mask = np.fliplr(img), np.fliplr(mask)
    return img, mask

def remove_black_background(img, mask, margin=5):
    _, binary = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return img, mask
    x, y, w, h = cv2.boundingRect(coords)
    x, y = max(0, x - margin), max(0, y - margin)
    w = min(img.shape[1] - x, w + 2*margin)
    h = min(img.shape[0] - y, h + 2*margin)
    return img[y:y+h, x:x+w], mask[y:y+h, x:x+w]

def resize_with_padding(img, mask, target=640):
    h, w    = img.shape
    scale   = target / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img  = cv2.resize(img,  (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    pad_h, pad_w = target - new_h, target - new_w
    img  = np.pad(img,  ((0, pad_h), (0, pad_w)), mode='constant')
    mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant')
    return img, mask

valid_count = 0
for img_name in tqdm([f for f in os.listdir(IMG_RAW) if f.endswith('.png')],
                     desc='Spatial Preprocessing'):
    img_path  = os.path.join(IMG_RAW,  img_name)
    mask_path = os.path.join(MASK_RAW, img_name)
    if not os.path.exists(mask_path):
        continue  # Only process images that have corresponding masks

    img  = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None: continue

    img, mask = crop_borders(img, mask)
    img, mask = orient_to_right(img, mask)
    img, mask = remove_black_background(img, mask)
    img, mask = resize_with_padding(img, mask, target=640)
    _, mask   = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    cv2.imwrite(os.path.join(IMG_PREP,  img_name), img)
    cv2.imwrite(os.path.join(MASK_PREP, img_name), mask)
    valid_count += 1

print(f'\n {valid_count} images preprocessed and ready.')

# ------------------------------------------------------------
# ### 3.6 GT crop extraction function

def extract_gt_crops(img_dir, mask_dir, out_dir, split, file_list=None, pad_ratio=0.05):
    """
    Same as extract_gt_crops in the pre-training notebook.
    Takes the GT bounding box from the mask and creates a 128×128 crop.
    file_list: optional list of specific filenames (used for 5-Fold CV splits).
    """
    out_imgs  = os.path.join(out_dir, split, 'images')
    out_masks = os.path.join(out_dir, split, 'masks')
    os.makedirs(out_imgs,  exist_ok=True)
    os.makedirs(out_masks, exist_ok=True)

    if file_list is None:
        files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))])
    else:
        files = file_list

    saved = skipped = 0

    for fname in tqdm(files, desc=f'[{split}] Extracting crops', leave=False):
        img_path  = os.path.join(img_dir,  fname)
        mask_path = os.path.join(mask_dir, fname)
        if not os.path.exists(mask_path) or not os.path.exists(img_path):
            skipped += 1; continue

        img  = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None: skipped += 1; continue

        H, W = img.shape
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8), connectivity=8
        )

        for comp_idx in range(1, n_labels):
            area = stats[comp_idx, cv2.CC_STAT_AREA]
            if area < 50: continue

            x = stats[comp_idx, cv2.CC_STAT_LEFT]
            y = stats[comp_idx, cv2.CC_STAT_TOP]
            w = stats[comp_idx, cv2.CC_STAT_WIDTH]
            h = stats[comp_idx, cv2.CC_STAT_HEIGHT]

            pad = int(pad_ratio * max(w, h))
            x1 = max(0, x - pad);  y1 = max(0, y - pad)
            x2 = min(W, x+w+pad);  y2 = min(H, y+h+pad)

            crop_img  = img[y1:y2, x1:x2]
            crop_mask = mask[y1:y2, x1:x2]
            if crop_img.size == 0: continue

            crop_img  = cv2.resize(crop_img,  (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
            crop_mask = cv2.resize(crop_mask, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_NEAREST)
            _, crop_mask = cv2.threshold(crop_mask, 127, 255, cv2.THRESH_BINARY)
            if crop_mask.max() == 0: continue

            stem = fname.rsplit('.', 1)[0]
            name = f'{stem}_m{comp_idx}.png'
            cv2.imwrite(os.path.join(out_imgs,  name), crop_img)
            cv2.imwrite(os.path.join(out_masks, name), crop_mask)
            saved += 1

    print(f'  [{split}] saved={saved} | skipped={skipped}')
    return saved

print(' extract_gt_crops ready!')

# ------------------------------------------------------------
# ### 3.7 Augmentation pipeline

# Augmentation applied jointly to image and mask to preserve alignment
seg_aug_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Affine(
        translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
        scale=(0.9, 1.1),
        rotate=(-15, 15),
        p=0.7
    ),
    A.ElasticTransform(alpha=20, sigma=4, p=0.2),
    A.RandomGamma(gamma_limit=(85, 115), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    A.GaussNoise(var_limit=(1.0, 8.0), p=0.2),
])

def augment_crops(img_dir, mask_dir, out_img_dir, out_mask_dir, n_aug=2):
    """Copy originals + generate n_aug augmented versions per crop (same as detection notebook)."""
    os.makedirs(out_img_dir,  exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    for fname in os.listdir(img_dir):
        if not fname.endswith('.png'): continue
        img_path  = os.path.join(img_dir,  fname)
        mask_path = os.path.join(mask_dir, fname)
        if not os.path.exists(mask_path): continue

        img  = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None: continue

        # Copy original
        shutil.copy(img_path,  os.path.join(out_img_dir,  fname))
        shutil.copy(mask_path, os.path.join(out_mask_dir, fname))

        # Augmented versions
        for i in range(n_aug):
            try:
                aug = seg_aug_transform(image=img, mask=mask)
                aug_img, aug_mask = aug['image'], aug['mask']
                if aug_mask.max() == 0: continue  # Skip if mask was wiped out by augmentation
                new_name = fname.replace('.png', f'_aug{i}.png')
                cv2.imwrite(os.path.join(out_img_dir,  new_name), aug_img)
                cv2.imwrite(os.path.join(out_mask_dir, new_name), aug_mask)
            except Exception:
                continue

print(' Augmentation function ready!')

# ------------------------------------------------------------
# ### 3.8 Training utilities

#  Transforms 
train_tf = A.Compose([
    A.Normalize(mean=[0.5], std=[0.5]),
    ToTensorV2(),
])

val_tf = A.Compose([
    A.Normalize(mean=[0.5], std=[0.5]),
    ToTensorV2(),
])

#  Dataset 
class SegCropDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform):
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.files     = sorted(os.listdir(img_dir))
        self.transform = transform

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img  = cv2.imread(os.path.join(self.img_dir,  fname), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(self.mask_dir, fname), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        out  = self.transform(image=img, mask=mask)
        return out['image'].float(), out['mask'].unsqueeze(0).float()

#  Model Builder 
def build_unet(pretrained_ckpt=None):
    model = smp.Unet(
        encoder_name='timm-mobilenetv3_small_075',
        encoder_weights='imagenet' if pretrained_ckpt is None else None,
        in_channels=1,
        classes=1,
        activation=None,
    )
    if pretrained_ckpt:
        ckpt = torch.load(pretrained_ckpt, map_location='cpu', weights_only=False)
        
        # >>> Fix to handle both full-model and checkpoint-dict formats <<<
        if isinstance(ckpt, torch.nn.Module):
            # If the loaded file is a complete model object
            state = ckpt.state_dict()
        else:
            # If the loaded file is a checkpoint dictionary
            state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
            
        model.load_state_dict(state, strict=True)
        print(f'   Loaded weights from {Path(pretrained_ckpt).name}')
    return model.to(DEVICE)

def freeze_encoder(model, freeze=True):
    """Freeze or unfreeze the encoder, similar to freeze=5 in YOLO."""
    for param in model.encoder.parameters():
        param.requires_grad = not freeze
    status = 'FROZEN' if freeze else 'UNFROZEN'
    print(f'  Encoder: {status}')
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        prob  = torch.sigmoid(logits)
        inter = (prob * targets).sum(dim=(2,3))
        union = prob.sum(dim=(2,3)) + targets.sum(dim=(2,3))
        return (1 - (2*inter+self.smooth)/(union+self.smooth)).mean()

def calculate_metrics(logits, targets, thr=0.5):
    pred  = (torch.sigmoid(logits) > thr).float()
    inter = (pred * targets).sum(dim=(2,3))
    union_dice = pred.sum(dim=(2,3)) + targets.sum(dim=(2,3))
    union_iou  = union_dice - inter
    dice = ((2 * inter + 1) / (union_dice + 1)).mean().item()
    iou  = ((inter + 1) / (union_iou  + 1)).mean().item()
    return dice, iou

#  Training Loop 
def train_model(model, train_dl, val_dl, save_path, epochs, patience, lr):
    """
    Training loop with:
    - Encoder freezing for the first FREEZE_EPOCHS epochs
    - Early stopping
    - Checkpoint saving
    """
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.6, patience=5
    )

    best_dice  = 0.0
    no_improve = 0

    #  Freeze the encoder initially 
    freeze_encoder(model, freeze=True)

    for epoch in range(1, epochs + 1):

        #  Unfreeze encoder after FREEZE_EPOCHS 
        if epoch == FREEZE_EPOCHS + 1:
            freeze_encoder(model, freeze=False)
            # Re-initialize optimizer to include newly unfrozen parameters
            optimizer = torch.optim.Adam(model.parameters(), lr=lr * 0.5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.6, patience=5
            )

        #  TRAIN 
        model.train()
        tl = td = tiou = 0.0
        pbar = tqdm(train_dl, desc=f'Ep {epoch:03d}/{epochs} [TRAIN]', leave=False)
        for imgs, masks in pbar:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            dice, iou = calculate_metrics(logits.detach(), masks)
            tl += loss.item(); td += dice; tiou += iou
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Dice': f'{dice:.4f}'})
        tl /= len(train_dl); td /= len(train_dl); tiou /= len(train_dl)

        #  VALIDATION 
        model.eval()
        vl = vd = viou = 0.0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                logits = model(imgs)
                loss   = criterion(logits, masks)
                dice, iou = calculate_metrics(logits, masks)
                vl += loss.item(); vd += dice; viou += iou
        vl /= len(val_dl); vd /= len(val_dl); viou /= len(val_dl)

        scheduler.step(vd)

        flag = ''
        if vd > best_dice:
            best_dice  = vd
            no_improve = 0
            flag       = '  BEST'
            torch.save({'model_state': model.state_dict(), 'val_dice': vd,
                        'val_iou': viou, 'epoch': epoch}, save_path)
        else:
            no_improve += 1

        print(f'Ep {epoch:03d} | TRAIN: Loss={tl:.4f} Dice={td:.4f} IoU={tiou:.4f} | '
              f'VAL: Loss={vl:.4f} Dice={vd:.4f} IoU={viou:.4f}{flag}')

        if no_improve >= patience:
            print(f'\nEarly stopping at epoch {epoch}')
            break

    print(f'\n Best Val Dice: {best_dice:.4f}')
    return best_dice

print(' Training utilities ready!')

# ------------------------------------------------------------
# ### 3.9 5-Fold Cross-Validation on INbreast

#  Collect all images that have corresponding masks 
all_images = sorted([
    f for f in os.listdir(IMG_PREP)
    if f.endswith('.png') and os.path.exists(os.path.join(MASK_PREP, f))
])
print(f'Total valid images for CV: {len(all_images)}')

kf         = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []
os.makedirs(CV_BASE, exist_ok=True)

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_images)):
    print(f'\n{"="*60}')
    print(f'  FOLD {fold_idx+1} / 5   |  Train: {len(train_idx)}  |  Val: {len(val_idx)}')
    print(f'{"="*60}')

    train_imgs = [all_images[i] for i in train_idx]
    val_imgs   = [all_images[i] for i in val_idx]

    #  Fold directories 
    fold_dir         = os.path.join(CV_BASE, f'fold_{fold_idx}')
    raw_train_imgs   = os.path.join(fold_dir, 'raw_train', 'images')
    raw_train_masks  = os.path.join(fold_dir, 'raw_train', 'masks')
    aug_train_imgs   = os.path.join(fold_dir, 'train', 'images')
    aug_train_masks  = os.path.join(fold_dir, 'train', 'masks')
    val_imgs_dir     = os.path.join(fold_dir, 'val', 'images')
    val_masks_dir    = os.path.join(fold_dir, 'val', 'masks')

    for d in [raw_train_imgs, raw_train_masks, aug_train_imgs, aug_train_masks,
              val_imgs_dir, val_masks_dir]:
        os.makedirs(d, exist_ok=True)

    #  Step 1: Extract crops for the training fold 
    print('  Extracting train crops...')
    extract_gt_crops(IMG_PREP, MASK_PREP, fold_dir + '/raw_train_crops',
                     'imgs', file_list=train_imgs)

    #  Step 2: Apply augmentation to the training crops 
    raw_crop_imgs  = os.path.join(fold_dir, 'raw_train_crops', 'imgs', 'images')
    raw_crop_masks = os.path.join(fold_dir, 'raw_train_crops', 'imgs', 'masks')
    augment_crops(raw_crop_imgs, raw_crop_masks, aug_train_imgs, aug_train_masks, n_aug=N_AUG)
    print(f'  Train crops (with aug): {len(os.listdir(aug_train_imgs))}')

    #  Step 3: Extract crops for the validation fold (no augmentation) 
    print('  Extracting val crops...')
    extract_gt_crops(IMG_PREP, MASK_PREP, fold_dir + '/val_crops',
                     'imgs', file_list=val_imgs)
    raw_val_imgs  = os.path.join(fold_dir, 'val_crops', 'imgs', 'images')
    raw_val_masks = os.path.join(fold_dir, 'val_crops', 'imgs', 'masks')
    # Copy val crops without augmentation
    for f in os.listdir(raw_val_imgs):
        shutil.copy(os.path.join(raw_val_imgs,  f), os.path.join(val_imgs_dir,  f))
        shutil.copy(os.path.join(raw_val_masks, f), os.path.join(val_masks_dir, f))
    print(f'  Val crops: {len(os.listdir(val_imgs_dir))}')

    #  Step 4: DataLoaders 
    train_ds = SegCropDataset(aug_train_imgs, aug_train_masks, train_tf)
    val_ds   = SegCropDataset(val_imgs_dir,   val_masks_dir,   val_tf)
    train_dl = DataLoader(train_ds, batch_size=SEG_BATCH, shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=SEG_BATCH, shuffle=False, num_workers=4, pin_memory=True)

    #  Step 5: Build model and fine-tune 
    gc.collect(); torch.cuda.empty_cache()
    model = build_unet(pretrained_ckpt=pretrained_ckpt)

    fold_save = os.path.join(CV_BASE, f'fold_{fold_idx}_best.pth')
    best_dice = train_model(
        model, train_dl, val_dl,
        save_path = fold_save,
        epochs    = CV_EPOCHS,
        patience  = SEG_PATIENCE,
        lr        = SEG_LR_INIT
    )

    #  Step 6: Evaluate on validation fold 
    ckpt = torch.load(fold_save, map_location=DEVICE, weights_only=False)
    
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    total_dice = total_iou = 0.0
    with torch.no_grad():
        for imgs, masks in val_dl:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits = model(imgs)
            d, iou = calculate_metrics(logits, masks)
            total_dice += d; total_iou += iou
    final_dice = total_dice / len(val_dl)
    final_iou  = total_iou  / len(val_dl)

    cv_results.append({'Fold': fold_idx+1, 'Dice': round(final_dice, 4), 'IoU': round(final_iou, 4)})
    print(f'   Fold {fold_idx+1} → Dice={final_dice:.4f} | IoU={final_iou:.4f}')

    #  Clean up temporary fold directories 
    shutil.rmtree(os.path.join(fold_dir, 'raw_train_crops'), ignore_errors=True)
    shutil.rmtree(os.path.join(fold_dir, 'val_crops'),       ignore_errors=True)
    shutil.rmtree(os.path.join(fold_dir, 'train'),           ignore_errors=True)
    shutil.rmtree(os.path.join(fold_dir, 'val'),             ignore_errors=True)

#  Summary 
df_cv = pd.DataFrame(cv_results)
mean_row = {'Fold': 'Mean', 'Dice': round(df_cv['Dice'].mean(), 4), 'IoU': round(df_cv['IoU'].mean(), 4)}
std_row  = {'Fold': 'Std',  'Dice': round(df_cv['Dice'].std(),  4), 'IoU': round(df_cv['IoU'].std(),  4)}
df_cv = pd.concat([df_cv, pd.DataFrame([mean_row, std_row])], ignore_index=True)

print('\n 5-Fold CV Results on INbreast:')
print('-' * 40)
print(df_cv.to_string(index=False))
print('-' * 40)

# ------------------------------------------------------------
# ### 3.10 Final fine-tuning on full INbreast dataset
# 
# Train on ALL 106 INbreast images (with augmentation), using the best  
# hyperparameters identified during cross-validation.
# 

os.makedirs(FINAL_DIR, exist_ok=True)

#  Step 1: Extract crops from all images 
print('Extracting crops from ALL INbreast images...')
extract_gt_crops(IMG_PREP, MASK_PREP, FINAL_DIR + '/raw_crops', 'all')

raw_all_imgs  = os.path.join(FINAL_DIR, 'raw_crops', 'all', 'images')
raw_all_masks = os.path.join(FINAL_DIR, 'raw_crops', 'all', 'masks')

#  Step 2: Apply augmentation to training crops 
final_train_imgs  = os.path.join(FINAL_DIR, 'train', 'images')
final_train_masks = os.path.join(FINAL_DIR, 'train', 'masks')
final_val_imgs    = os.path.join(FINAL_DIR, 'val',   'images')  # No augmentation — used to monitor training
final_val_masks   = os.path.join(FINAL_DIR, 'val',   'masks')

augment_crops(raw_all_imgs, raw_all_masks, final_train_imgs, final_train_masks, n_aug=N_AUG)
print(f'Train crops (with aug): {len(os.listdir(final_train_imgs))}')

# Val = originals without augmentation (to monitor training)
os.makedirs(final_val_imgs,  exist_ok=True)
os.makedirs(final_val_masks, exist_ok=True)
for f in os.listdir(raw_all_imgs):
    shutil.copy(os.path.join(raw_all_imgs,  f), os.path.join(final_val_imgs,  f))
    shutil.copy(os.path.join(raw_all_masks, f), os.path.join(final_val_masks, f))
print(f'Val crops (originals): {len(os.listdir(final_val_imgs))}')

#  Step 3: DataLoaders 
final_train_ds = SegCropDataset(final_train_imgs, final_train_masks, train_tf)
final_val_ds   = SegCropDataset(final_val_imgs,   final_val_masks,   val_tf)
final_train_dl = DataLoader(final_train_ds, batch_size=SEG_BATCH, shuffle=True,  num_workers=4, pin_memory=True)
final_val_dl   = DataLoader(final_val_ds,   batch_size=SEG_BATCH, shuffle=False, num_workers=4, pin_memory=True)

#  Step 4: Fine-Tune 
print('\n Starting Final Fine-Tuning on full INbreast...')
gc.collect(); torch.cuda.empty_cache()

final_model = build_unet(pretrained_ckpt=pretrained_ckpt)
train_model(
    final_model, final_train_dl, final_val_dl,
    save_path = UNET_FINAL,
    epochs    = FINAL_EPOCHS,
    patience  = SEG_PATIENCE,
    lr        = SEG_LR_INIT
)

print(f'\n Final model saved at: {UNET_FINAL}')

# ------------------------------------------------------------
# 
# ## Part 4 — Final Results & External Validation

# ------------------------------------------------------------
# ### 4.1 Load final model & evaluation function

# ── Load the final model ────────────────────────────────────────────
ckpt = torch.load(UNET_FINAL, map_location=DEVICE)
final_model = build_unet()
final_model.load_state_dict(ckpt['model_state'])
final_model.eval()
print(f' Final model loaded (best val Dice: {ckpt["val_dice"]:.4f})')

def evaluate_segmentation(model, img_dir, mask_dir, dataset_name):
    """Evaluate the segmentation model on an external dataset."""
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))])
    files = [f for f in files if os.path.exists(os.path.join(mask_dir, f))]

    if not files:
        print(f'   No matching files for {dataset_name}')
        return None

    ds = SegCropDataset(img_dir, mask_dir, val_tf)
    dl = DataLoader(ds, batch_size=SEG_BATCH, shuffle=False, num_workers=4, pin_memory=True)

    total_dice = total_iou = 0.0
    model.eval()
    with torch.no_grad():
        for imgs, masks in tqdm(dl, desc=f'Evaluating {dataset_name}'):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits = model(imgs)
            d, iou = calculate_metrics(logits, masks)
            total_dice += d; total_iou += iou

    final_dice = total_dice / len(dl)
    final_iou  = total_iou  / len(dl)
    print(f'  {dataset_name}: Dice={final_dice:.4f} | IoU={final_iou:.4f}')
    return {'Dataset': dataset_name, 'Dice': round(final_dice, 4), 'IoU': round(final_iou, 4)}

# ------------------------------------------------------------
# ### 4.2 CBIS-DDSM test evaluation (catastrophic forgetting check)

# ── Test 1: CBIS-DDSM (Catastrophic Forgetting check) ──────────────
# These images were converted to 128×128 crops in the pre-training notebook
CBIS_TEST_IMGS  = os.environ.get('CBIS_TEST_IMGS', 'data/CBIS-DDSM/preprocessed/test_imgs')
CBIS_TEST_MASKS = os.environ.get('CBIS_TEST_MASKS', 'data/CBIS-DDSM/preprocessed/test_masks')

# ── Extract crops from CBIS-DDSM test set ───────────────────────────
CBIS_CROPS_OUT = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'cbis_test_crops')
extract_gt_crops(CBIS_TEST_IMGS, CBIS_TEST_MASKS, CBIS_CROPS_OUT, 'test')

cbis_result = evaluate_segmentation(
    final_model,
    os.path.join(CBIS_CROPS_OUT, 'test', 'images'),
    os.path.join(CBIS_CROPS_OUT, 'test', 'masks'),
    'CBIS-DDSM test'
)

# ------------------------------------------------------------
# ### 4.3 BCDR mask preparation

# ------------------------------------------------------------
# ### 4.4 BCDR test evaluation (generalization)

# ── Test 2: BCDR (Generalization — Pixel Perfect Masks) ─────────────

BCDR_IMGS  = os.environ.get('BCDR_YOLO_TEST_IMGS', 'data/BCDR/bcdr_yolo/images/test')
# Use the pre-prepared clean mask folder
BCDR_MASKS = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'bcdr_test_masks_clean') 

BCDR_CROPS_OUT = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'bcdr_test_crops')

# Extract crops for testing
print('Extracting BCDR test crops...')
extract_gt_crops(BCDR_IMGS, BCDR_MASKS, BCDR_CROPS_OUT, 'test')

# Run evaluation and compute Dice Score
bcdr_result = evaluate_segmentation(
    final_model,
    os.path.join(BCDR_CROPS_OUT, 'test', 'images'),
    os.path.join(BCDR_CROPS_OUT, 'test', 'masks'),
    'BCDR test'
)

# ------------------------------------------------------------
# ### 4.6 Full results print

print('\n' + '='*60)
print('  FINAL RESULTS SUMMARY')
print('='*60)

print('\n 5-Fold CV on INbreast (Dice / IoU):')
print(df_cv.to_string(index=False))

print('\n External Validation:')
ext_rows = [r for r in [cbis_result, bcdr_result] if r is not None]
if ext_rows:
    df_ext = pd.DataFrame(ext_rows)
    print(df_ext.to_string(index=False))

print('='*60)
