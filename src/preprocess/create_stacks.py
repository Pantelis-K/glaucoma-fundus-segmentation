"""
BUILD STACKS: cropped images -> multi-channel stacks (RGB + CLAHE + Sobel)

Input:
  data/images/<id>.png      (cropped fundus images from jitter_crop_ROI.py)

Output:
  data/stacks/<id>_stack.npy
    shape: (H, W, 5)
    channels: [R, G, B, CLAHE_gray, Sobel_mag]
    dtype: uint8 (0..255)

These stacks are later loaded in trainer.py and normalised to [-1, 1].
"""
# imports
import argparse
from pathlib import Path
import numpy as np
import cv2

#configs 
CLIP_LIMIT = 2.0
TILE_GRID = 8
SOBEL_KSIZE = 3

# Filepaths 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
img_dir = PROJECT_ROOT / "data" / "images"
#output directory
out_stack_dir = PROJECT_ROOT / "data" / "stacks"
out_stack_dir.mkdir(parents=True, exist_ok=True)

img_files = sorted(img_dir.glob("*.png"))

# Create CLAHE object
clahe = cv2.createCLAHE(
    clipLimit=CLIP_LIMIT,
    tileGridSize=(TILE_GRID, TILE_GRID),
)

# Process each image, create stack, and save
for img_path in img_files:
    base_name = img_path.stem

    # Read image (OpenCV loads BGR)
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"[SKIP] Could not read image: {img_path.name}")
        continue

    # Convert to RGB and grayscale
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)          # (H,W,3), uint8
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)        # (H,W), uint8

    # CLAHE on grayscale
    clahe_gray = clahe.apply(gray) # (H,W), uint8                      

    # Sobel magnitude on grayscale (then normalise to 0..255)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=SOBEL_KSIZE) # (H,W), float32
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=SOBEL_KSIZE) # (H,W), float32
    mag = np.sqrt(gx * gx + gy * gy) # (H,W), float32

    # Normalize Sobel magnitude to 0..255
    maxv = float(mag.max()) if mag.size else 0.0
    if maxv > 0:
        mag = (mag / maxv) * 255.0
    sobel_mag = np.clip(mag, 0, 255).astype(np.uint8)   # (H,W), uint8

    # Stack channels: RGB + CLAHE + Sobel => (H,W,5)
    stack = np.dstack([rgb, clahe_gray[..., None], sobel_mag[..., None]]).astype(np.uint8)

    # Save as .npy
    out_path = out_stack_dir / f"{base_name}_stack.npy"
    np.save(out_path, stack)

    print(f"Saved stack: {out_path.name}  shape={stack.shape} dtype={stack.dtype}")

print(f"\n[DONE] Wrote stacks to: {out_stack_dir}")



