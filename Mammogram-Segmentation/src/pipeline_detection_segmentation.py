import os
import cv2
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from huggingface_hub import hf_hub_download, login
from ultralytics import YOLO

# # 1- Preprocessing

def crop_borders(img, top_bottom=0.025, sides=0.01):
    h, w = img.shape
    tb = int(h * top_bottom)
    sd = int(w * sides)
    return img[tb:h-tb, sd:w-sd]

def orient_to_right(img):
    if np.sum(img[:, :img.shape[1]//2]) < np.sum(img[:, img.shape[1]//2:]):
        img = np.fliplr(img)
    return img

def remove_black_background(img, margin=5):
    _, binary = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(img.shape[1] - x, w + 2*margin)
    h = min(img.shape[0] - y, h + 2*margin)
    return img[y:y+h, x:x+w]

def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def resize_with_padding(img, target=640):
    h, w = img.shape
    scale = target / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h = target - new_h
    pad_w = target - new_w
    img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant')
    return img

def preprocess_single_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    img = crop_borders(img)
    img = orient_to_right(img)
    img = remove_black_background(img)
    img = apply_clahe(img)
    img = resize_with_padding(img, target=640)
    return img  # (640, 640) grayscale

# # 2- Load Models

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_models(repo_id="hab200/Breast-Cancer-Mammogram-Detection-Segmentation_new"):
    print("Loading models from Hugging Face...")
    
    # Load YOLO
    yolo_path = hf_hub_download(repo_id=repo_id, filename="yolo_best.pt")
    yolo_model = YOLO(yolo_path)

    # Load UNet
    unet_path = hf_hub_download(repo_id=repo_id, filename="unet_inbreast_final.pth")
    torch.serialization.add_safe_globals([smp.Unet])
    unet_model = torch.load(unet_path, map_location=DEVICE, weights_only=False)
    unet_model.eval()
    print(f"Models loaded successfully!")
    return yolo_model, unet_model

# # 3- Yolo Detection 

def run_yolo(preprocessed_img, yolo_model, conf=0.15, iou=0.25):
    img_bgr = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2BGR)
    
    results = yolo_model(img_bgr, imgsz=640, conf=conf, iou=iou, verbose=False)[0]

    raw_boxes   = results.boxes.xyxy.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy().tolist()

    final_boxes = [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in raw_boxes]

    return final_boxes, confidences

# # 4- Unet Model

unet_transform = A.Compose([
    A.Normalize(mean=[0.5], std=[0.5]),
    ToTensorV2(),
])

def run_unet_on_box(preprocessed_img, box, unet_model):
    H, W = preprocessed_img.shape
    x1, y1, x2, y2 = map(int, box)

    # Clip box to image boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)

    crop = preprocessed_img[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return None

    # Resize to 128x128 - same as training input size
    crop_resized = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_LINEAR)

    # Apply same normalization as training
    tensor = unet_transform(image=crop_resized)['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = unet_model(tensor)
        pred_mask = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()[0, 0]

    # Resize mask back to original box size
    pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
    mask_resized = cv2.resize(pred_mask_uint8, (x2-x1, y2-y1),
                              interpolation=cv2.INTER_NEAREST)
    return mask_resized, (x1, y1, x2, y2)

# # 5 - Full Pipline 

def full_pipeline(img_path, yolo_model, unet_model, yolo_conf=0.15,yolo_iou = 0.25):
    """
    Input:  path to a raw mammogram image
    Output: dict containing:
        - 'final_mask'   : binary mask (same size as preprocessed image)
        - 'confidence'   : highest YOLO confidence score
        - 'preprocessed' : image after preprocessing
        - 'boxes'        : merged bounding boxes from YOLO
    """

    # Step 1: Preprocessing
    preprocessed = preprocess_single_image(img_path)
    H, W = preprocessed.shape

    # Step 2: YOLO detection
    merged_boxes, box_confidences = run_yolo(preprocessed, yolo_model, conf=yolo_conf, iou =yolo_iou)

    # Step 3 & 4: Run UNet on each box and merge all masks
    final_mask = np.zeros((H, W), dtype=np.uint8)

    for box in merged_boxes:
        result = run_unet_on_box(preprocessed, box, unet_model)
        if result is None:
            continue
        mask_resized, (x1, y1, x2, y2) = result
        # Use bitwise_or to correctly merge multiple lesion masks
        final_mask[y1:y2, x1:x2] = cv2.bitwise_or(
            final_mask[y1:y2, x1:x2], mask_resized
        )

    best_confidence = max(box_confidences) if box_confidences else 0.0

    return {
        'final_mask':   final_mask,
        'confidence':   best_confidence,
        'preprocessed': preprocessed,
        'boxes':        merged_boxes
    }

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Mammogram Detection + Segmentation Pipeline')
    parser.add_argument('--img', required=True, help='Path to input mammogram image')
    parser.add_argument('--conf', type=float, default=0.15, help='YOLO confidence threshold')
    args = parser.parse_args()

    yolo_model, unet_model = load_models()

    result = full_pipeline(
        img_path=args.img,
        yolo_model=yolo_model,
        unet_model=unet_model,
        yolo_conf=args.conf
    )

    print(f"Confidence: {result['confidence']:.2%}")

    plt.figure(figsize=(5, 5))
    plt.imshow(result['final_mask'], cmap='gray')
    plt.axis('off')
    plt.show()

