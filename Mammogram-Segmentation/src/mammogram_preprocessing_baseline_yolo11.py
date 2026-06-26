# # Phase 1: Data Preparation and Baseline YOLO11 Training
# This notebook contains the foundational pipeline for breast mass detection. It covers medical image preprocessing, converting segmentation masks into YOLO-format bounding boxes, data augmentation, and executing the initial baseline training of the YOLO11s model on CBIS_DDSM Data.

# ### 1. Medical Image Preprocessing Pipeline

import os
import cv2
import shutil
import pandas as pd
import numpy as np
import urllib.request
import albumentations as A
import matplotlib.pyplot as plt

from ultralytics import YOLO
from tqdm import tqdm


def crop_borders(img, mask, top_bottom=0.025, sides=0.01):
    # Remove artificial margins, text labels, and scanner artifacts from edges
    h, w = img.shape
    tb = int(h * top_bottom)
    sd = int(w * sides)

    img  = img[tb:h-tb, sd:w-sd]
    mask = mask[tb:h-tb, sd:w-sd]
    return img, mask

def orient_to_right(img, mask):
    # Standardize breast orientation to the right side for architectural consistency
    if np.sum(img[:, :img.shape[1]//2]) < np.sum(img[:, img.shape[1]//2:]):
        img  = np.fliplr(img)
        mask = np.fliplr(mask)
    return img, mask

def remove_black_background(img, mask=None, margin=5):
    # Isolate the breast tissue region and crop out excessive uninformative black background
    _, binary = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)

    if coords is None:
        return (img, mask) if mask is not None else img

    x, y, w, h = cv2.boundingRect(coords)

    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(img.shape[1] - x, w + 2*margin)
    h = min(img.shape[0] - y, h + 2*margin)

    img_cropped = img[y:y+h, x:x+w]
    if mask is not None:
        return img_cropped, mask[y:y+h, x:x+w]
    return img_cropped

def resize_with_padding(img, mask, target=640):
    # Resize the image to the target dimensions while preserving the original aspect ratio using padding
    h, w = img.shape
    scale = target / max(h, w)

    new_h, new_w = int(h * scale), int(w * scale)

    img  = cv2.resize(img,  (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    pad_h = target - new_h
    pad_w = target - new_w

    img  = np.pad(img,  ((0, pad_h), (0, pad_w)), mode='constant')
    mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant')

    return img, mask

def apply_clahe(img):
    # Enhance local tissue contrast using Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

# Mask cleaning 
def clean_mask(mask):
    # Mask cleaning 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask

# Main preprocessing :
def preprocess_dataset(input_dir, out_img_dir, out_mask_dir):
    os.makedirs(out_img_dir,  exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    patients = sorted(os.listdir(input_dir))
    skipped = 0

    for patient in tqdm(patients, desc=f'Processing {os.path.basename(input_dir)}'):
        patient_path = os.path.join(input_dir, patient)

        # find image + mask 
        files = os.listdir(patient_path)

        img_path = None
        mask_path = None

        for f in files:
            if 'full' in f.lower():
                img_path = os.path.join(patient_path, f)
            if 'mask' in f.lower():
                mask_path = os.path.join(patient_path, f)

        if img_path is None or mask_path is None:
            skipped += 1
            continue

        img  = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            skipped += 1
            continue

        #preprocessing pipeline 
        img, mask = crop_borders(img, mask)
        img, mask = orient_to_right(img, mask)
        img, mask = remove_black_background(img, mask)
        mask = clean_mask(mask)
        img  = apply_clahe(img)
        img, mask = resize_with_padding(img, mask, target=640)

        #save
        name = patient + '.jpg'
        cv2.imwrite(os.path.join(out_img_dir,  name), img)
        cv2.imwrite(os.path.join(out_mask_dir, name), mask)

    print(f'Done. Skipped: {skipped}')

DATA_ROOT = os.environ.get('CBIS_ORGANIZED_PATH', 'data/CBIS-DDSM/organized')

OUT_ROOT = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'preprocessed')

preprocess_dataset(
    os.path.join(DATA_ROOT, 'train'),
    os.path.join(OUT_ROOT, 'train_imgs'),
    os.path.join(OUT_ROOT, 'train_masks')
)

preprocess_dataset(
    os.path.join(DATA_ROOT, 'val'),
    os.path.join(OUT_ROOT, 'val_imgs'),
    os.path.join(OUT_ROOT, 'val_masks')
)

preprocess_dataset(
    os.path.join(DATA_ROOT, 'test'),
    os.path.join(OUT_ROOT, 'test_imgs'),
    os.path.join(OUT_ROOT, 'test_masks')
)

# ### 2. Annotation Conversion (Segmentation Masks to YOLO Bounding Boxes)

def mask_to_yolo_labels(mask_path, label_path, min_area_ratio=0.0001):
    # Load the mask in grayscale mode
    mask = cv2.imread(mask_path, 0)
    if mask is None: return False

    h, w = mask.shape
    img_area = h * w
    
    # Binarize the mask to ensure clear foreground/background separation
    _, binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    
    # Identify distinct tissue masses using connected components
    num_labels, labels_map = cv2.connectedComponents(binary)
    boxes = []

    for i in range(1, num_labels):
        ys, xs = np.where(labels_map == i)
        comp_area = len(xs)
        
        # Filter out negligible noise artifacts based on area threshold
        if comp_area / img_area < min_area_ratio: continue

        # Extract precise bounding box coordinates (tight fit around the mass)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        # Normalize coordinates to [0, 1] for YOLO format (cx, cy, w, h)
        cx = (x_min + x_max) / 2 / w
        cy = (y_min + y_max) / 2 / h
        bw = (x_max - x_min) / w
        bh = (y_max - y_min) / h

        boxes.append(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')

    # If no valid masses are found, skip creating a label file
    if not boxes: return False
    
    # Save the normalized annotations
    with open(label_path, 'w') as f: 
        f.write('\n'.join(boxes))
        
    return True

# ## Spliting data for training 


def convert_split(img_dir, mask_dir, split_name, dataset_root):
    img_out = f'{dataset_root}/images/{split_name}'
    lbl_out = f'{dataset_root}/labels/{split_name}'

    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    kept, skipped = 0, 0

    for img_name in tqdm(os.listdir(img_dir), desc=split_name):
        if not img_name.endswith('.jpg'):
            continue

        img_path  = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)

        if not os.path.exists(mask_path):
            skipped += 1
            continue

        label_path = os.path.join(lbl_out, img_name.replace('.jpg', '.txt'))

        ok = mask_to_yolo_labels(mask_path, label_path)

        if ok:
            shutil.copy(img_path, os.path.join(img_out, img_name))
            kept += 1
        else:
            skipped += 1

    print(f'{split_name}: kept={kept}, skipped={skipped}')

convert_split(
    img_dir  = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'preprocessed/train_imgs'),
    mask_dir = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'preprocessed/train_masks'),
    split_name = 'train',
    dataset_root = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'dataset_yolo')
)

convert_split(
    img_dir  = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'preprocessed/val_imgs'),
    mask_dir = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'preprocessed/val_masks'),
    split_name = 'val',
    dataset_root = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'dataset_yolo')
)

convert_split(
    img_dir  = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'preprocessed/test_imgs'),
    mask_dir = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'preprocessed/test_masks'),
    split_name = 'test',
    dataset_root = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'dataset_yolo')
)

# ### 3. Mammogram Data Augmentation
# We apply simple augmentations to increase data variety without distorting the real shape of breast masses. Transformations are kept minimal because medical anomalies lose their physical meaning if heavily deformed.


mamo_aug = A.Compose(
    [
        # Flip image left/right (since we have both left and right breasts)
        A.HorizontalFlip(p=0.5),
        
        # Very small rotation and scale to keep the breast mass shape natural
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.08,
            rotate_limit=5,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.4
        ),
        
        # Change lighting slightly to simulate different machines
        A.RandomGamma(p=0.3),
        
        # Add slight noise similar to real mammogram scans
        A.GaussNoise(p=0.2),
    ],
    
    # Bounding Box settings for YOLO
    bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['category_ids'],
        clip=True,               # Keep boxes inside the image limits
        min_visibility=0.5       # Ignore the box if more than 50% of the mass is cut off
    )
)

def augment_and_save(img_dir, lbl_dir, out_img_dir, out_lbl_dir, n_aug=2):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    for img_name in tqdm(os.listdir(img_dir), desc="Augmenting Dataset"):
        if not img_name.endswith('.jpg'):
            continue

        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, img_name.replace('.jpg', '.txt'))

        if not os.path.exists(lbl_path):
            continue

        # Read image in grayscale, then convert to RGB because Albumentations expects 3 channels
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        bboxes, cat_ids = [], []

        # Read original YOLO bounding boxes
        with open(lbl_path) as f:
            for line in f:
                cls, cx, cy, w, h = map(float, line.split())
                bboxes.append([cx, cy, w, h])
                cat_ids.append(int(cls))

        # Always keep the original (unaugmented) image and label in the new dataset
        shutil.copy(img_path, os.path.join(out_img_dir, img_name))
        shutil.copy(lbl_path, os.path.join(out_lbl_dir, img_name.replace('.jpg', '.txt')))

        # Generate new augmented versions
        for i in range(n_aug):
            result = mamo_aug(image=img3, bboxes=bboxes, category_ids=cat_ids)

            # Skip saving if the mass was completely cropped out during augmentation
            if not result['bboxes']:
                continue

            # Convert back to grayscale for consistent medical imaging format
            aug_img = cv2.cvtColor(result['image'], cv2.COLOR_RGB2GRAY)

            new_name = img_name.replace('.jpg', f'_aug{i}.jpg')
            new_lbl  = new_name.replace('.jpg', '.txt')

            # Save the new augmented image and its corresponding labels
            cv2.imwrite(os.path.join(out_img_dir, new_name), aug_img)

            with open(os.path.join(out_lbl_dir, new_lbl), 'w') as f:
                for cid, box in zip(result['category_ids'], result['bboxes']):
                    f.write(f'{cid} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n')

TRAIN_IMG = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'dataset_yolo/images/train')
TRAIN_LBL = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'dataset_yolo/labels/train')

AUG_IMG = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'dataset_yolo_aug/images/train')
AUG_LBL = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'dataset_yolo_aug/labels/train')

augment_and_save(
    TRAIN_IMG,
    TRAIN_LBL,
    AUG_IMG,
    AUG_LBL,
    n_aug=2
)

import shutil

shutil.copytree(os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'dataset_yolo/images/val'),
                os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'dataset_yolo_aug/images/val'), dirs_exist_ok=True)

shutil.copytree(os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'dataset_yolo/labels/val'),
                os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'dataset_yolo_aug/labels/val'), dirs_exist_ok=True)

shutil.copytree(os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'dataset_yolo/images/test'),
                os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'dataset_yolo_aug/images/test'), dirs_exist_ok=True)

shutil.copytree(os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'dataset_yolo/labels/test'),
                os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'dataset_yolo_aug/labels/test'), dirs_exist_ok=True)

# # 4-YAML File 

yaml_path = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'mammo.yaml')

with open(yaml_path, 'w') as f:
    f.write(f"""
path: {output_dir}/dataset_yolo_aug  # set OUTPUT_DIR env variable

train: images/train
val: images/val
test: images/test

nc: 1
names: ['mass']
""")

# ### 5. Initializing Domain-Specific Pre-trained Model (Transfer Learning)
# Instead of starting from a model trained on general everyday objects (like the COCO dataset), we initialize our YOLO11s architecture using weights pre-trained on a mammography benchmark. This domain-specific transfer learning significantly accelerates convergence and enhances the model's ability to extract relevant medical features, such as tissue densities and mass borders.


os.makedirs(os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'models'), exist_ok=True)

model_url  = 'https://github.com/cbddobvyz/digitaleye-mammography/releases/download/shared-models.v2/yolo11_s.pt'
model_path = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'models/yolo11_s.pt')

# Download the model weights if they are not already downloaded
if not os.path.exists(model_path):
    print('Downloading YOLO11s pretrained weights...')
    urllib.request.urlretrieve(model_url, model_path)
    print(f'Downloaded: {os.path.getsize(model_path)/1e6:.1f} MB')
else:
    print(f'Model already exists: {model_path}')

# ### 6. Model Training with Medical-Specific Hyperparameters


model = YOLO(os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'models/yolo11_s.pt'))

model.train(
    data=yaml_path,
    imgsz=640,
    batch=8,

    optimizer='AdamW',
    lr0=1e-4,

    kobj=2.5,
    box=7.5,
    cls=1.2,

    fliplr=0.5,
    scale=0.1,
    translate=0.03,
    mosaic=0.1,
)


model = YOLO(os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'runs/detect/train/weights/best.pt'))

# 1. Evaluation on test set
metrics = model.val(data=os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'mammo.yaml'), split='test', plots=True, imgsz=640)

print(f"mAP@50       : {metrics.box.map50:.3f}")
print(f"mAP@50-95    : {metrics.box.map:.3f}")
print(f"Precision    : {metrics.box.mp:.3f}")
print(f"Recall (TPR) : {metrics.box.mr:.3f}")  


model = YOLO(os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'runs/detect/train/weights/best.pt'))

metrics = model.val(
    data=os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'mammo.yaml'), 
    split='test', 
    imgsz=640, 
    augment=True,  
    verbose=False
)
print(f"mAP@50 (with TTA) : {metrics.box.map50:.3f}")
print(f"Recall (with TTA) : {metrics.box.mr:.3f}")

# ### 7. Qualitative Evaluation and Visual Inspection
# To complement the quantitative metrics (like mAP and Precision), we perform a qualitative visual inspection on random samples from the test set. This step is critical in medical imaging to manually verify that the predicted bounding boxes align accurately with the anatomical location of the masses defined by the ground truth masks.


# Load the best trained YOLO model weights
yolo_model = YOLO(os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'runs/detect/train/weights/best.pt'))

def test_yolo_thresholds(img_path, mask_path, thresholds=[0.15, 0.25, 0.40, 0.60]):
    """
    Evaluates the model on a specific image across different confidence thresholds 
    and visualizes the predicted bounding boxes against the ground truth.
    """
    orig_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if orig_img is None or gt_mask is None:
        print(f"Error loading: {img_path}")
        return
            
    # Initialize plotting canvas
    fig = plt.figure(figsize=(24, 6))
    
    # Plot 1: Original Preprocessed Mammogram
    plt.subplot(1, len(thresholds) + 2, 1)
    plt.title("Original Image")
    plt.imshow(orig_img, cmap='gray')
    plt.axis('off')
    
    # Plot 2: Ground Truth Segmentation Mask
    plt.subplot(1, len(thresholds) + 2, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(gt_mask, cmap='gray')
    plt.axis('off')
    
    # Plot 3+: YOLO Predictions at specified confidence thresholds
    for i, conf in enumerate(thresholds):
        # Convert grayscale to RGB to draw colored bounding boxes clearly
        vis_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2RGB)
        
        # Run inference with the current confidence threshold
        results = yolo_model(img_path, imgsz=640, conf=conf, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy() 

        # Draw predicted bounding boxes in Red
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 3) 
        
        # Display the prediction plot
        plt.subplot(1, len(thresholds) + 2, i + 3)
        plt.title(f"YOLO Conf = {conf}\nDetected Boxes: {len(boxes)}")
        plt.imshow(vis_img)
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

# ── Randomly sample images from the Test Set for visual evaluation ──
print("\nExecuting Qualitative Visual Evaluation...")

TEST_IMGS_DIR = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'preprocessed/test_imgs')
TEST_MASKS_DIR = os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'preprocessed/test_masks')

test_files = [f for f in os.listdir(TEST_IMGS_DIR) if f.endswith(('.jpg', '.png'))]
sample_files = random.sample(test_files, min(25, len(test_files)))

for img_name in sample_files:
    img_path = os.path.join(TEST_IMGS_DIR, img_name)
    mask_path = os.path.join(TEST_MASKS_DIR, img_name)
    print(f"-"*40)
    print(f"Processing: {img_name}")
    
    # Using conf=0.15 as the baseline threshold for visual inspection
    test_yolo_thresholds(img_path, mask_path, thresholds=[0.15])

# ### 8. Confidence Threshold Optimization & Trade-off Analysis


# Load the optimized model weights
yolo_model = YOLO(os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'runs/detect/train/weights/best.pt'))

# Define a range of confidence thresholds to evaluate the Precision-Recall trade-off
thresholds_to_test = [0.01, 0.05, 0.15, 0.25, 0.40, 0.60]

results_list = []

print("⏳ Evaluating metrics for different Confidence Thresholds on the Test Set...\n")

for conf in thresholds_to_test:
    # Evaluate the model on the test split for the current threshold
    metrics = yolo_model.val(
        data=os.path.join(os.environ.get('OUTPUT_DIR', 'outputs'), 'mammo.yaml'), 
        split='test', 
        conf=conf, 
        imgsz=640, 
        plots=False,   # Disable automatic plot generation to speed up the loop
        verbose=False  # Suppress excessive YOLO logging output
    )
    
    # Extract core detection metrics
    p = metrics.box.mp
    r = metrics.box.mr
    map50 = metrics.box.map50
    
    # Store the results for tabular presentation
    results_list.append({
        "Confidence Thresh": conf,
        "Precision": round(p, 3),
        "Recall (TPR)": round(r, 3),
        "mAP@50": round(map50, 3)
    })

# Compile and display the final metrics in a structured dataframe
df_results = pd.DataFrame(results_list)
print(" Final Results Table:")
print("-" * 40)
print(df_results.to_string(index=False))

