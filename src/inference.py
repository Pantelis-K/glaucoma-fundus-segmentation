"""
inference.py

Run the disc and cup U-Net models on a single cropped fundus image and produce
an annotated output showing predicted (red) and ground-truth (green) contours,
along with predicted and true vertical CDR.

Usage:
    python src/inference.py --id 001
    python src/inference.py          # prompts interactively
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Suppress TensorFlow C++ logs (INFO, WARNING) before the TF import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# Suppress Python-level TF and absl logs
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# src/ imports (no sys.path manipulation needed — we are already in src/)
# ---------------------------------------------------------------------------
import metrics
from config import (
    IMAGE_SIZE, N_CHANNELS, CLIP_LIMIT, TILE_GRID, SOBEL_KSIZE,
    DISC_MODEL_PATH, CUP_MODEL_PATH,
    IMAGES_DIR, MASKS_DIR, ORIGA_INFO_CSV,
)
from mask_utils import load_origa_disc_cup_masks, fit_ellipse_params, ellipse_cdr_metrics


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def _parse_id() -> str:
    """Return sample ID from --id argument, or prompt interactively."""
    parser = argparse.ArgumentParser(
        description="Run U-Net inference on a single cropped fundus image."
    )
    parser.add_argument("--id", type=str, help="Sample ID, e.g. 586 or 001.")
    args, _ = parser.parse_known_args()
    if args.id:
        return args.id.strip()
    while True:
        val = input("Enter sample ID (e.g. 001): ").strip()
        if val.isdigit():
            return val
        print("  Please enter a numeric ID.")


def _prompt_id() -> str:
    """Prompt interactively for a sample ID. Returns 'q' if the user wants to quit."""
    while True:
        val = input("\nEnter sample ID (or 'q' to quit): ").strip()
        if val.lower() == "q" or val.isdigit():
            return val
        print("  Please enter a numeric ID.")


def _resolve_image(sample_id: str) -> tuple[Path | None, str]:
    """Zero-pad the ID and return (image_path, padded_id), or None if not found."""
    padded   = f"{int(sample_id):03d}"
    img_path = IMAGES_DIR / f"{padded}.png"
    return (img_path if img_path.exists() else None), padded


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _build_stack(img_path: Path) -> np.ndarray:
    """
    Build a normalised 5-channel stack (RGB + CLAHE + Sobel) from a cropped
    fundus image. Replicates the logic in src/preprocess/create_stacks.py.

    Returns float32 array of shape (1, 256, 256, 5) normalised to [-1, 1].
    """
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    rgb        = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray       = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe      = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=(TILE_GRID, TILE_GRID))
    clahe_gray = clahe.apply(gray)

    gx  = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=SOBEL_KSIZE)
    gy  = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=SOBEL_KSIZE)
    mag = np.sqrt(gx * gx + gy * gy)
    maxv = float(mag.max())
    sobel_mag = (mag / maxv * 255.0 if maxv > 0 else mag).clip(0, 255).astype(np.uint8)

    stack = np.dstack([rgb, clahe_gray[..., None], sobel_mag[..., None]]).astype(np.float32)

    resized = np.stack(
        [cv2.resize(stack[..., c], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
         for c in range(N_CHANNELS)],
        axis=-1,
    )
    return (resized / 127.5 - 1.0)[np.newaxis]  # (1, 256, 256, 5)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_CUSTOM_OBJECTS = {
    "log_dice_loss": metrics.log_dice_loss,
    "dice_coef":     metrics.dice_coef,
    "boundary_loss": metrics.boundary_loss,
    "iou":           metrics.iou,
}


def _load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {path}\n"
            "Train first:  python src/trainer.py --target disc\n"
            "              python src/trainer.py --target cup"
        )
    return tf.keras.models.load_model(str(path), custom_objects=_CUSTOM_OBJECTS)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _predict_mask(model, stack: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Return a binary (H, W) uint8 mask from a (1, H, W, 5) stack."""
    prob = model.predict(stack, verbose=0)   # (1, 256, 256, 1)
    return (prob[0, ..., 0] > threshold).astype(np.uint8)


# ---------------------------------------------------------------------------
# CDR lookup
# ---------------------------------------------------------------------------

def _lookup_true_cdr(padded_id: str) -> float | None:
    """Return the CDR value for this image from origa_info.csv, or None."""
    if not ORIGA_INFO_CSV.exists():
        return None
    info = pd.read_csv(ORIGA_INFO_CSV)
    info.columns = info.columns.str.lower()
    row = info[info["image"] == int(padded_id)]
    return float(row["cdr"].values[0]) if not row.empty else None


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _draw_ellipse(img_bgr: np.ndarray, mask: np.ndarray, color: tuple, thickness: int = 2):
    """Fit an ellipse to the largest contour in mask and draw it on img_bgr in-place."""
    params = fit_ellipse_params(mask)
    if params is None:
        return
    cx, cy, a, b, ang_rad = params
    cv2.ellipse(img_bgr, (int(cx), int(cy)), (int(a), int(b)),
                np.degrees(ang_rad), 0, 360, color, thickness)


def _render_output(
    img_path: Path,
    disc_pred: np.ndarray,
    cup_pred: np.ndarray,
    disc_gt: np.ndarray,
    cup_gt: np.ndarray,
    pred_vcdr: float,
    true_cdr: float | None,
) -> np.ndarray:
    """
    Compose the annotated output image:
      - Green ellipses : ground-truth disc and cup boundaries
      - Red ellipses   : predicted disc and cup boundaries
      - Text overlay   : predicted vCDR and true CDR
    """
    bgr    = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    h, w   = bgr.shape[:2]
    GREEN  = (0, 255, 0)
    RED    = (0, 0, 255)

    # Resize masks to match the display image before fitting ellipses
    def _resize(m): return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

    _draw_ellipse(bgr, _resize(disc_gt),   GREEN, thickness=2)
    _draw_ellipse(bgr, _resize(cup_gt),    GREEN, thickness=2)
    _draw_ellipse(bgr, _resize(disc_pred), RED,   thickness=2)
    _draw_ellipse(bgr, _resize(cup_pred),  RED,   thickness=2)

    vcdr_str   = f"Predicted vCDR : {pred_vcdr:.3f}" if not np.isnan(pred_vcdr) else "Predicted vCDR : N/A"
    true_str   = f"True CDR       : {true_cdr:.3f}"  if true_cdr is not None    else "True CDR       : not available"
    legend_str = "Green = Ground truth    Red = Prediction"

    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    pad, line_h = 8, 22
    for i, text in enumerate([vcdr_str, true_str, legend_str]):
        y = h - pad - (2 - i) * line_h
        # dark outline for legibility, then white fill
        cv2.putText(bgr, text, (pad, y), font, scale, (0, 0, 0),       thick + 2, cv2.LINE_AA)
        cv2.putText(bgr, text, (pad, y), font, scale, (255, 255, 255),  thick,     cv2.LINE_AA)

    return bgr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Disc model : {DISC_MODEL_PATH}")
    print(f"Cup model  : {CUP_MODEL_PATH}\n")

    # Load models once — reused across all iterations
    print("Loading models...")
    disc_model = _load_model(DISC_MODEL_PATH)
    cup_model  = _load_model(CUP_MODEL_PATH)
    print("Models loaded. Enter 'q' at any prompt to quit.\n")

    # Use --id arg for the first iteration only; prompt interactively after that
    sample_id = _parse_id()

    while True:
        # --- Resolve image, re-prompting if not found ---
        while True:
            img_path, padded_id = _resolve_image(sample_id)
            if img_path is not None:
                break
            print(f"  No cropped image found for ID '{padded_id}' in {IMAGES_DIR}")
            sample_id = _prompt_id()
            if sample_id.lower() == "q":
                return

        print(f"Running inference on sample {padded_id}...")

        # --- Preprocess ---
        stack = _build_stack(img_path)

        # --- Predict ---
        disc_pred = _predict_mask(disc_model, stack)
        cup_pred  = _predict_mask(cup_model,  stack)

        # --- Predicted CDR (ellipse-based) ---
        pred_vcdr = ellipse_cdr_metrics(disc_pred, cup_pred)["vertical_cdr"]

        # --- Ground truth (cropped mask, spatially aligned with the display image) ---
        gt_mask_path = MASKS_DIR / f"{padded_id}.png"
        if gt_mask_path.exists():
            disc_gt, cup_gt = load_origa_disc_cup_masks(gt_mask_path)
        else:
            print(f"  Warning: cropped mask not found at {gt_mask_path}; GT contours skipped.")
            disc_gt = np.zeros_like(disc_pred)
            cup_gt  = np.zeros_like(cup_pred)

        # --- True CDR ---
        true_cdr = _lookup_true_cdr(padded_id)

        # --- Print results ---
        print(f"\n  Predicted vertical CDR : {pred_vcdr:.3f}" if not np.isnan(pred_vcdr) else "\n  Predicted vertical CDR : N/A")
        print(f"  True CDR (origa_info)  : {true_cdr:.3f}" if true_cdr is not None else "  True CDR               : not found in origa_info.csv")

        # --- Render and display ---
        out_img = _render_output(img_path, disc_pred, cup_pred, disc_gt, cup_gt, pred_vcdr, true_cdr)
        cv2.imshow(f"Inference - Sample {padded_id}  (press any key to close)", out_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # --- Prompt for next sample ---
        sample_id = _prompt_id()
        if sample_id.lower() == "q":
            break


if __name__ == "__main__":
    main()
