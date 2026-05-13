"""
Tests for the pure-numpy evaluation helpers in analysis/analysis_utils.py.
No data files or GPU required — these run in milliseconds.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from analysis_utils import per_sample_dice, summary_stats, bootstrap_ci, bmw_indices


# ---------------------------------------------------------------------------
# per_sample_dice
# ---------------------------------------------------------------------------

def test_per_sample_dice_perfect():
    # Identical masks: every pixel overlaps -> Dice should be ~1.0
    masks = np.ones((3, 32, 32), dtype=np.float32)
    preds = np.ones((3, 32, 32), dtype=np.float32)

    scores = per_sample_dice(masks, preds)

    assert scores.shape == (3,)
    assert np.allclose(scores, 1.0, atol=1e-4)


def test_per_sample_dice_no_overlap():
    # Ground truth all ones, prediction all zeros: no overlap -> Dice ~0.0
    masks = np.ones((3, 32, 32), dtype=np.float32)
    preds = np.zeros((3, 32, 32), dtype=np.float32)

    scores = per_sample_dice(masks, preds)

    assert scores.shape == (3,)
    assert np.all(scores < 0.01)


def test_per_sample_dice_accepts_4d_input():
    # Function should squeeze a trailing channel dimension
    masks = np.ones((2, 32, 32, 1), dtype=np.float32)
    preds = np.ones((2, 32, 32, 1), dtype=np.float32)

    scores = per_sample_dice(masks, preds)

    assert scores.shape == (2,)
    assert np.allclose(scores, 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# summary_stats
# ---------------------------------------------------------------------------

def test_summary_stats_keys():
    scores = np.array([0.7, 0.8, 0.9, 0.85, 0.75], dtype=np.float32)

    stats = summary_stats(scores)

    assert set(stats.keys()) == {"mean", "std", "median", "q25", "q75"}


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

def test_bootstrap_ci_deterministic():
    # Same seed must produce identical bounds across runs
    scores = np.linspace(0.7, 0.95, 50).astype(np.float32)

    mean1, lo1, hi1 = bootstrap_ci(scores, seed=42)
    mean2, lo2, hi2 = bootstrap_ci(scores, seed=42)

    assert mean1 == mean2
    assert lo1 == lo2
    assert hi1 == hi2


def test_bootstrap_ci_bounds_ordered():
    scores = np.linspace(0.7, 0.95, 50).astype(np.float32)

    mean, lo, hi = bootstrap_ci(scores, seed=0)

    assert lo < mean < hi


# ---------------------------------------------------------------------------
# bmw_indices
# ---------------------------------------------------------------------------

def test_bmw_indices_order():
    scores = np.array([0.5, 0.9, 0.7, 0.3, 0.8], dtype=np.float32)

    best_i, med_i, worst_i = bmw_indices(scores)

    assert scores[best_i] >= scores[med_i] >= scores[worst_i]


def test_bmw_indices_returns_valid_indices():
    scores = np.array([0.6, 0.8, 0.4, 0.9, 0.7], dtype=np.float32)

    best_i, med_i, worst_i = bmw_indices(scores)

    for idx in (best_i, med_i, worst_i):
        assert 0 <= idx < len(scores)
