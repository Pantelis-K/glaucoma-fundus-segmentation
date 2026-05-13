"""
mask_utils.py
Core mask I/O, ellipse geometry, and CDR metric computation.

Used by both inference (src/) and analysis (analysis/) pipelines.
No matplotlib, pandas, or sklearn dependencies.
"""

import cv2
import numpy as np
from pathlib import Path

from config import CUP_VALUE  # noqa: E402


# ---------------------------------------------------------------------------
# Mask I/O
# ---------------------------------------------------------------------------

def mask_name_from_csv(image_id) -> str:
    """ORIGA mask filename for an integer image ID (e.g. 1 -> '001.png')."""
    return f"{int(image_id):03d}.png"


def load_origa_disc_cup_masks(path: Path):
    """
    Load an ORIGA annotation mask and return (disc, cup) as uint8 binary arrays
    at native image resolution.

    Returns both channels in a single call because every geometric feature
    (CDR, RDR, cup_offset, ISNT) requires disc and cup simultaneously. Native
    resolution is preserved so that ellipse fitting and pixel-count ratios reflect
    true spatial proportions.

    Use this when computing geometric features from ORIGA ground-truth annotations.
    Do NOT use it for model evaluation: it does not resize to IMAGE_SIZE and does
    not return float32.
    """
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    return (m > 0).astype(np.uint8), (m == CUP_VALUE).astype(np.uint8)


# ---------------------------------------------------------------------------
# Ellipse geometry
# ---------------------------------------------------------------------------

def fit_ellipse_params(mask: np.ndarray):
    """
    Fit an ellipse to the largest contour in a binary mask.
    Returns (cx, cy, a, b, ang_rad) or None if fitting fails.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5:
        return None
    (cx, cy), (MA, ma), ang = cv2.fitEllipse(cnt)
    a, b = MA * 0.5, ma * 0.5
    return cx, cy, a, b, np.deg2rad(ang)


def radius_along_dir(cx, cy, a, b, ang_rad, dir_xy: np.ndarray) -> float:
    """Radius of an ellipse along a direction vector in global coordinates."""
    cos_t, sin_t = np.cos(ang_rad), np.sin(ang_rad)
    u_x =  dir_xy[0] * cos_t + dir_xy[1] * sin_t
    u_y = -dir_xy[0] * sin_t + dir_xy[1] * cos_t
    denom = (u_x / a) ** 2 + (u_y / b) ** 2
    return 1.0 / np.sqrt(denom) if denom > 0 else np.nan


def _smallest_positive_root(A, B, C):
    """Smallest positive root of A t^2 + B t + C = 0."""
    disc = B * B - 4 * A * C
    if disc <= 0:
        return np.nan
    r1 = (-B - np.sqrt(disc)) / (2 * A)
    r2 = (-B + np.sqrt(disc)) / (2 * A)
    roots = [r for r in (r1, r2) if r > 0]
    return min(roots) if roots else np.nan


def distance_to_offset_ellipse(src_c, dir_xy, tgt_params) -> float:
    """Distance from src_c along dir_xy until intersecting the target ellipse."""
    cx, cy, a, b, ang = tgt_params
    cos_t, sin_t = np.cos(ang), np.sin(ang)
    vx, vy = src_c[0] - cx, src_c[1] - cy
    vx_, vy_ = vx * cos_t + vy * sin_t, -vx * sin_t + vy * cos_t
    dx_, dy_ = dir_xy[0] * cos_t + dir_xy[1] * sin_t, -dir_xy[0] * sin_t + dir_xy[1] * cos_t
    A = (dx_ / a) ** 2 + (dy_ / b) ** 2
    B = 2 * ((vx_ * dx_) / (a * a) + (vy_ * dy_) / (b * b))
    C = (vx_ / a) ** 2 + (vy_ / b) ** 2 - 1
    return _smallest_positive_root(A, B, C)


# ---------------------------------------------------------------------------
# CDR and rim metrics
# ---------------------------------------------------------------------------

def cdr_metrics(disc: np.ndarray, cup: np.ndarray) -> dict:
    """Pixel-based CDR metrics: area, vertical, and horizontal cup-to-disc ratio."""
    disc_pts = np.column_stack(np.where(disc))
    cup_pts  = np.column_stack(np.where(cup))
    dy_d = np.ptp(disc_pts[:, 0]) if disc_pts.size else 0
    dx_d = np.ptp(disc_pts[:, 1]) if disc_pts.size else 0
    dy_c = np.ptp(cup_pts[:, 0])  if cup_pts.size  else 0
    dx_c = np.ptp(cup_pts[:, 1])  if cup_pts.size  else 0
    return {
        "area_cdr":       cup.sum() / disc.sum() if disc.sum() else np.nan,
        "vertical_cdr":   dy_c / dy_d if dy_d else np.nan,
        "horizontal_cdr": dx_c / dx_d if dx_d else np.nan,
    }


def isnt_violations(disc: np.ndarray, cup: np.ndarray) -> int:
    """Count violations of a simple ISNT rim-area ordering proxy."""
    rim = np.logical_and(disc, ~cup)
    h, w = rim.shape
    cy, cx = h // 2, w // 2
    quad = {
        "I": rim[:cy, cx:],
        "S": rim[:cy, :cx],
        "N": rim[cy:, :cx],
        "T": rim[cy:, cx:],
    }
    order = ["I", "S", "N", "T"]
    cnt = [quad[o].sum() for o in order]
    return sum(cnt[i] < cnt[i + 1] for i in range(3))


def rim_to_disc_ratio(disc: np.ndarray, cup: np.ndarray) -> float:
    """Rim-to-disc ratio along the cup-disc offset direction."""
    d_p, c_p = fit_ellipse_params(disc), fit_ellipse_params(cup)
    cx_d, cy_d, a_d, b_d, ang_d = d_p
    cx_c, cy_c, a_c, b_c, ang_c = c_p
    v = np.array([cx_c - cx_d, cy_c - cy_d])
    if np.allclose(v, 0):
        v = np.array([1.0, 0.0])
    n = v / np.linalg.norm(v)
    disc_r = radius_along_dir(cx_d, cy_d, a_d, b_d, ang_d, n)
    cup_r  = distance_to_offset_ellipse((cx_d, cy_d), n, c_p)
    return (disc_r - cup_r) / disc_r if disc_r > cup_r else np.nan


def cup_offset(disc: np.ndarray, cup: np.ndarray) -> float:
    """Normalised offset between disc and cup ellipse centres."""
    d_p, c_p = fit_ellipse_params(disc), fit_ellipse_params(cup)
    cx_d, cy_d, a_d, b_d, _ = d_p
    cx_c, cy_c, a_c, b_c, _ = c_p
    return np.hypot(cx_c - cx_d, cy_c - cy_d) / np.sqrt(a_d * b_d)


def cup_eccentricity(cup: np.ndarray) -> float:
    """Ellipse eccentricity of the cup contour."""
    c_p = fit_ellipse_params(cup)
    _, _, a, b, _ = c_p
    return 1 - min(a, b) / max(a, b)


def ellipse_cdr_metrics(disc: np.ndarray, cup: np.ndarray) -> dict:
    """
    Compute CDR metrics from ellipses fitted to the disc and cup masks.

    More robust than pixel-based CDR for noisy or irregular prediction masks,
    because the ellipse acts as a smooth regulariser over the raw segmentation.

    Returns area_cdr, vertical_cdr, horizontal_cdr (NaN if ellipse fitting fails).
    """
    d_p = fit_ellipse_params(disc)
    c_p = fit_ellipse_params(cup)

    if d_p is None or c_p is None:
        return {"area_cdr": np.nan, "vertical_cdr": np.nan, "horizontal_cdr": np.nan}

    _, _, a_d, b_d, ang_d = d_p
    _, _, a_c, b_c, ang_c = c_p

    # Area CDR: ratio of ellipse areas (pi cancels)
    area_cdr = float((a_c * b_c) / (a_d * b_d)) if (a_d * b_d) > 0 else np.nan

    # Vertical CDR: vertical half-extent of each ellipse's bounding box
    disc_half_h = np.sqrt((a_d * np.sin(ang_d)) ** 2 + (b_d * np.cos(ang_d)) ** 2)
    cup_half_h  = np.sqrt((a_c * np.sin(ang_c)) ** 2 + (b_c * np.cos(ang_c)) ** 2)
    vertical_cdr = float(cup_half_h / disc_half_h) if disc_half_h > 0 else np.nan

    # Horizontal CDR: horizontal half-extent of each ellipse's bounding box
    disc_half_w = np.sqrt((a_d * np.cos(ang_d)) ** 2 + (b_d * np.sin(ang_d)) ** 2)
    cup_half_w  = np.sqrt((a_c * np.cos(ang_c)) ** 2 + (b_c * np.sin(ang_c)) ** 2)
    horizontal_cdr = float(cup_half_w / disc_half_w) if disc_half_w > 0 else np.nan

    return {
        "area_cdr":       area_cdr,
        "vertical_cdr":   vertical_cdr,
        "horizontal_cdr": horizontal_cdr,
    }
