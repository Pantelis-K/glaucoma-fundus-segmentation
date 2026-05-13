"""
Tests for the custom TensorFlow loss and metric functions in src/metrics.py.
Uses small constant tensors — no GPU or real data required.
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import dice_coef, iou


# ---------------------------------------------------------------------------
# dice_coef
# ---------------------------------------------------------------------------

def test_dice_coef_perfect():
    # Identical tensors: every pixel agrees -> Dice should be ~1.0
    y = tf.ones((1, 32, 32, 1))

    score = dice_coef(y, y).numpy()

    assert np.isclose(score, 1.0, atol=1e-4)


def test_dice_coef_zero_overlap():
    # Prediction is all zeros, ground truth is all ones -> Dice ~0.0
    y_true = tf.ones((1, 32, 32, 1))
    y_pred = tf.zeros((1, 32, 32, 1))

    score = dice_coef(y_true, y_pred).numpy()

    assert score < 0.01


# ---------------------------------------------------------------------------
# iou
# ---------------------------------------------------------------------------

def test_iou_perfect():
    # Identical binary tensors: IoU should be 1.0
    y = tf.ones((1, 32, 32, 1))

    score = iou(y, y).numpy()

    assert np.isclose(score, 1.0, atol=1e-4)


def test_iou_zero_overlap():
    # Prediction is all zeros after thresholding -> IoU = 0.0
    y_true = tf.ones((1, 32, 32, 1))
    y_pred = tf.zeros((1, 32, 32, 1))

    score = iou(y_true, y_pred).numpy()

    assert score == 0.0


def test_iou_partial_overlap():
    # Left half predicted, right half is ground truth -> partial overlap
    mask = np.zeros((1, 4, 4, 1), dtype=np.float32)
    mask[0, :, :2, 0] = 1.0   # left half
    pred = np.zeros((1, 4, 4, 1), dtype=np.float32)
    pred[0, :, 2:, 0] = 1.0   # right half (no overlap)

    score = iou(tf.constant(mask), tf.constant(pred)).numpy()

    assert score == 0.0
