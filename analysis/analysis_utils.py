"""
analysis_utils.py
Shared utilities for evaluate.py and feature_importance.py.
Contains data loading, model inference helpers, evaluation metrics,
plotting, feature extraction, and statistical/multivariate analysis helpers.

Core mask I/O, geometry, and CDR metrics live in src/mask_utils.py and are
re-exported here so existing callers need no changes.
"""

import sys
import warnings
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# Bootstrap sys.path and imports from src/
# ---------------------------------------------------------------------------
# Go to project root (analysis/ -> root), then enter src/
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from config import (
    IMAGE_SIZE, N_CHANNELS, CUP_VALUE,
    DATA_DIR, RUNS_DIR, SRC_DIR,
)
from mask_utils import (
    mask_name_from_csv,
    load_origa_disc_cup_masks,
    fit_ellipse_params,
    radius_along_dir,
    distance_to_offset_ellipse,
    cdr_metrics,
    isnt_violations,
    rim_to_disc_ratio,
    cup_offset,
    cup_eccentricity,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_test_paths(cup_or_disc: str):
    """
    Return (test_stack_paths, test_mask_paths) using the same deterministic
    80/10/10 split as trainer.py: files are sorted lexicographically and split
    with take/skip - no random seed involved.
    """
    stack_paths = sorted((DATA_DIR / "stacks").glob("*_stack.npy"))
    mask_paths = sorted((DATA_DIR / "masks").glob("*.png"))
    assert len(stack_paths) == len(mask_paths), (
        f"Stack/mask count mismatch: {len(stack_paths)} stacks, {len(mask_paths)} masks"
    )
    n_total = len(stack_paths)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    return stack_paths[n_train + n_val:], mask_paths[n_train + n_val:]


def _load_stack(path) -> np.ndarray:
    """Load one 5-channel stack, resize to IMAGE_SIZE, normalise to [-1, 1]."""
    st = np.load(str(path)).astype(np.float32)[..., :N_CHANNELS]
    # Resize each channel independently (robust for C > 3)
    resized = np.stack(
        [cv2.resize(st[..., c], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
         for c in range(N_CHANNELS)],
        axis=-1,
    )
    return resized / 127.5 - 1.0


def _load_resized_eval_mask(path, cup_or_disc: str) -> np.ndarray:
    """
    Load and binarise a single mask channel for model evaluation.

    Resizes to IMAGE_SIZE (256 × 256) to match the resolution the model was trained
    on, and returns float32 for direct use as a TensorFlow input or comparison target.
    Returns one channel selected by cup_or_disc.

    Use this (via load_test_arrays) when comparing ground-truth masks against model
    predictions.  Do NOT use it for geometric feature extraction: the resize alters
    pixel proportions and invalidates ellipse-based measurements.

    Contrast with load_origa_disc_cup_masks, which preserves native resolution,
    returns uint8, and yields both disc and cup in a single call.
    """
    mask = np.array(Image.open(str(path)).convert("L"), dtype=np.uint8)
    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    if cup_or_disc == "cup":
        return (mask == CUP_VALUE).astype(np.float32)
    return (mask > 0).astype(np.float32)


def load_test_arrays(cup_or_disc: str):
    """
    Load the 65-sample test set as numpy arrays.

    Returns
    -------
    stacks : ndarray (65, 256, 256, 5)   float32, normalised to [-1, 1]
    masks  : ndarray (65, 256, 256)      float32, binary 0/1
    ids    : list[str]                   sample IDs, e.g. ["586", "587", ...]
    """
    stack_paths, mask_paths = get_test_paths(cup_or_disc)
    stacks = np.stack([_load_stack(p) for p in stack_paths])
    masks = np.stack([_load_resized_eval_mask(p, cup_or_disc) for p in mask_paths])
    ids = [p.stem.replace("_stack", "") for p in stack_paths]
    return stacks, masks, ids


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------

def load_model(cup_or_disc: str):
    """Load the best checkpoint with all custom metric objects registered."""
    import tensorflow as tf
    import metrics  # src/metrics.py

    model_path = RUNS_DIR / f"{cup_or_disc}_256_unet" / "best.h5"
    return tf.keras.models.load_model(
        str(model_path),
        custom_objects={
            "log_dice_loss": metrics.log_dice_loss,
            "dice_coef": metrics.dice_coef,
            "boundary_loss": metrics.boundary_loss,
            "iou": metrics.iou,
        },
    )


def predict_test(model, stacks: np.ndarray, batch_size: int = 8) -> np.ndarray:
    """Run model.predict on stacks; returns probability maps (N, 256, 256, 1)."""
    return model.predict(stacks, batch_size=batch_size, verbose=1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def per_sample_dice(
    masks: np.ndarray,
    preds: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Compute Dice coefficient for each sample.

    Parameters
    ----------
    masks  : (N, H, W) or (N, H, W, 1)  binary ground-truth
    preds  : (N, H, W) or (N, H, W, 1)  raw probabilities in [0, 1]

    Returns
    -------
    scores : (N,) float32
    """
    masks = masks.squeeze(-1) if masks.ndim == 4 else masks
    preds = preds.squeeze(-1) if preds.ndim == 4 else preds
    preds_bin = (preds > threshold).astype(np.float32)
    scores = []
    for gt, pr in zip(masks, preds_bin):
        inter = np.sum(gt * pr)
        scores.append((2.0 * inter + 1e-6) / (np.sum(gt) + np.sum(pr) + 1e-6))
    return np.array(scores, dtype=np.float32)


def bootstrap_ci(
    scores: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
):
    """
    Percentile bootstrap 95 % CI on the mean.

    Returns
    -------
    mean, ci_low, ci_high
    """
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(scores, size=len(scores), replace=True).mean()
        for _ in range(n_boot)
    ])
    return float(scores.mean()), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def summary_stats(scores: np.ndarray) -> dict:
    """Return dict with mean, std, median, q25, q75."""
    return {
        "mean":   float(np.mean(scores)),
        "std":    float(np.std(scores, ddof=1)),
        "median": float(np.median(scores)),
        "q25":    float(np.percentile(scores, 25)),
        "q75":    float(np.percentile(scores, 75)),
    }


def bmw_indices(scores: np.ndarray):
    """Return (best_idx, median_idx, worst_idx) indices by ascending Dice order."""
    sorted_idx = np.argsort(scores)
    return int(sorted_idx[-1]), int(sorted_idx[len(scores) // 2]), int(sorted_idx[0])


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_FONT = {"fontsize": 11}
_GRID_KW = {"alpha": 0.35}


def plot_training_curves(
    cup_df: pd.DataFrame,
    disc_df: pd.DataFrame,
    cup_best: int,
    disc_best: int,
) -> plt.Figure:
    """2 × 2 figure: (cup loss | cup Dice) over (disc loss | disc Dice)."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    _draw_pair(axs[0], cup_df,  cup_best,  "Cup Model")
    _draw_pair(axs[1], disc_df, disc_best, "Disc Model")
    fig.tight_layout()
    return fig


def _draw_pair(ax_row, df: pd.DataFrame, best_epoch: int, label: str):
    epochs = df["epoch"]
    specs = [
        (ax_row[0], "loss",      "val_loss",      "Loss (−log Dice)", f"{label} - Loss"),
        (ax_row[1], "dice_coef", "val_dice_coef", "Dice Coefficient", f"{label} - Dice"),
    ]
    for ax, train_col, val_col, ylabel, title in specs:
        ax.plot(epochs, df[train_col], label="Train")
        ax.plot(epochs, df[val_col],   label="Val")
        ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.7,
                   label=f"Best epoch ({best_epoch})")
        ax.set_xlabel("Epoch", **_FONT)
        ax.set_ylabel(ylabel, **_FONT)
        ax.set_title(title, **_FONT)
        ax.legend(fontsize=9)
        ax.grid(True, **_GRID_KW)


def plot_lr_schedule(
    cup_df: pd.DataFrame,
    disc_df: pd.DataFrame,
) -> plt.Figure:
    """Learning-rate schedule for both models (log scale)."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax, df, title in [
        (axs[0], cup_df,  "Cup Model - Learning Rate"),
        (axs[1], disc_df, "Disc Model - Learning Rate"),
    ]:
        ax.semilogy(df["epoch"], df["learning_rate"], marker=".", markersize=3)
        ax.set_xlabel("Epoch", **_FONT)
        ax.set_ylabel("Learning Rate (log scale)", **_FONT)
        ax.set_title(title, **_FONT)
        ax.grid(True, **_GRID_KW)
    fig.tight_layout()
    return fig


def plot_violin(cup_scores: np.ndarray, disc_scores: np.ndarray) -> plt.Figure:
    """Side-by-side violin plots of per-sample test Dice."""
    fig, ax = plt.subplots(figsize=(6, 5))
    parts = ax.violinplot(
        [cup_scores, disc_scores],
        positions=[1, 2],
        showmedians=True,
        showextrema=True,
    )
    colors = ["#4c9be8", "#e87c4c"]
    for body, col in zip(parts["bodies"], colors):
        body.set_facecolor(col)
        body.set_alpha(0.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Cup Model", "Disc Model"], **_FONT)
    ax.set_ylabel("Dice Coefficient", **_FONT)
    ax.set_title("Test Set Dice Distribution", **_FONT)
    ax.grid(True, axis="y", **_GRID_KW)
    fig.tight_layout()
    return fig


def plot_overlay_figure(
    stacks: np.ndarray,
    masks: np.ndarray,
    preds: np.ndarray,
    indices,
    title: str,
    sample_labels=("Best", "Median", "Worst"),
    dice_scores: np.ndarray = None,
) -> plt.Figure:
    """
    3-subplot figure.  Each subplot shows the RGB image with:
      - green contour: ground-truth boundary
      - red contour:   predicted boundary
    """
    gt_patch = mpatches.Patch(color="lime", label="Ground truth")
    pr_patch = mpatches.Patch(color="red",  label="Prediction")

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, idx, slabel in zip(axs, indices, sample_labels):
        rgb = ((stacks[idx, ..., :3] + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        gt = masks[idx] if masks.ndim == 3 else masks[idx, ..., 0]
        pr = preds[idx, ..., 0] if preds.ndim == 4 else preds[idx]
        ax.imshow(rgb)
        ax.contour(gt, levels=[0.5], colors=["lime"], linewidths=1.5)
        ax.contour(pr, levels=[0.5], colors=["red"],  linewidths=1.5)
        ax.legend(handles=[gt_patch, pr_patch], fontsize=8, loc="upper right")
        subtitle = slabel
        if dice_scores is not None:
            subtitle += f"\nDice = {dice_scores[idx]:.3f}"
        ax.set_title(subtitle, fontsize=10)
        ax.axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# ORIGA feature extraction
# ---------------------------------------------------------------------------

UNI_METRICS = [
    "area_cdr",
    "vertical_cdr",
    "horizontal_cdr",
    "rdr",
    "cup_offset",
    "cup_eccentricity",
    "isnt_violations",
]


def extract_origa_features(mask_dir: Path, label_csv: Path) -> pd.DataFrame:
    """
    Load ORIGA masks and compute geometric features for every sample in label_csv.

    Returns a DataFrame with columns: area_cdr, vertical_cdr, horizontal_cdr,
    rdr, rdr_raw, cup_offset, cup_eccentricity, isnt_violations, label, image.
    """
    from tqdm.auto import tqdm as _tqdm

    info = pd.read_csv(label_csv).rename(columns=str.lower)
    if not {"image", "label"}.issubset(info.columns):
        raise ValueError("CSV must contain columns named Image and Label.")
    print(f"{len(info)} samples found in CSV")
    rows = []
    for _, row in _tqdm(info.iterrows(), total=len(info), desc="Processing masks"):
        fname = mask_name_from_csv(row["image"])
        try:
            disc, cup = load_origa_disc_cup_masks(mask_dir / fname)
        except FileNotFoundError:
            warnings.warn(f"Skipping {fname}")
            continue
        m = cdr_metrics(disc, cup)
        m.update({
            "rdr":              rim_to_disc_ratio(disc, cup),
            "cup_offset":       cup_offset(disc, cup),
            "cup_eccentricity": cup_eccentricity(cup),
            "isnt_violations":  isnt_violations(disc, cup),
            "label":            row["label"],
            "image":            row["image"],
        })
        m["rdr_raw"] = m["rdr"]
        m["rdr"] = -m["rdr"]
        rows.append(m)
    df = pd.DataFrame(rows)
    nan_rows = df[df[UNI_METRICS].isna().any(axis=1)]
    if not nan_rows.empty:
        warnings.warn(f"NaNs detected for {len(nan_rows)} image(s) in feature computation.")
    return df


# ---------------------------------------------------------------------------
# Univariate analysis
# ---------------------------------------------------------------------------

def univariate_roc_analysis(df: pd.DataFrame, feature_cols: list):
    """
    Compute per-feature ROC-AUC, Mann-Whitney U, KS, and Welch's t-test.

    Returns (df_stats sorted by AUC descending, rocs list of dicts).
    """
    from scipy.stats import mannwhitneyu, ks_2samp, ttest_ind
    from sklearn.metrics import roc_auc_score, roc_curve

    stats, rocs = [], []
    for m in feature_cols:
        pos = df[df.label == 1][m].dropna()
        neg = df[df.label == 0][m].dropna()
        if pos.empty or neg.empty:
            continue
        mw_res = mannwhitneyu(pos, neg, alternative="two-sided")
        U_stat, mw_p = mw_res.statistic, mw_res.pvalue
        n1, n0 = len(pos), len(neg)
        f_cl = U_stat / (n1 * n0)
        ks_p = ks_2samp(pos, neg).pvalue
        t_p  = ttest_ind(pos, neg, equal_var=False).pvalue
        scores = df[m].replace([np.inf, -np.inf], np.nan).fillna(0).values
        auc    = roc_auc_score(df.label, scores)
        fpr, tpr, _ = roc_curve(df.label, scores)
        stats.append({"metric": m, "mw_p": mw_p, "ks_p": ks_p, "t_p": t_p, "auc": auc, "f": f_cl})
        rocs.append({"metric": m, "fpr": fpr, "tpr": tpr})
    return pd.DataFrame(stats).sort_values("auc", ascending=False), rocs


def plot_univariate_roc(df_stats: pd.DataFrame, rocs: list) -> plt.Figure:
    """Univariate ROC curves for all features."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in rocs:
        auc_val = df_stats.set_index("metric").loc[r["metric"], "auc"]
        ax.plot(r["fpr"], r["tpr"], label=f"{r['metric']} (AUC={auc_val:.2f})")
    ax.plot([0, 1], [0, 1], "--", lw=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Univariate ROC curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Multivariate modelling
# ---------------------------------------------------------------------------

def build_rf_pipeline():
    """Random Forest pipeline with median-threshold feature selection."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import SelectFromModel

    return make_pipeline(
        SimpleImputer(strategy="median"),
        SelectFromModel(RandomForestClassifier(n_estimators=500, random_state=42), threshold="median"),
        RandomForestClassifier(n_estimators=300, random_state=0),
    )


def build_lr_pipeline():
    """L1-regularised Logistic Regression pipeline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(penalty="l1", C=0.3, solver="saga", max_iter=4000),
    )


def run_cv_auc(pipe, X: np.ndarray, y: np.ndarray, skf) -> float:
    """Mean ROC-AUC across stratified k-fold splits."""
    from sklearn.metrics import roc_auc_score

    scores = []
    for tr, te in skf.split(X, y):
        pipe.fit(X[tr], y[tr])
        scores.append(roc_auc_score(y[te], pipe.predict_proba(X[te])[:, 1]))
    return float(np.mean(scores))


def fit_and_explain(lr, rf, X: np.ndarray, y: np.ndarray, features: list):
    """
    Fit LR and RF on (X, y) and return (lr_coef_series, rf_importance_series).
    LR: non-zero L1 coefficients sorted descending.
    RF: importances of SelectFromModel-retained features sorted descending.
    """
    lr.fit(X, y)
    coef = lr.named_steps["logisticregression"].coef_[0]
    coef_series = pd.Series(coef, index=features)
    lr_nonzero = coef_series[coef_series != 0].round(4).sort_values(ascending=False)

    rf.fit(X, y)
    selector = rf.named_steps["selectfrommodel"]
    kept_names = [features[i] for i in selector.get_support(indices=True)]
    imp = rf.named_steps["randomforestclassifier"].feature_importances_
    rf_importances = pd.Series(imp, index=kept_names).sort_values(ascending=False)

    return lr_nonzero, rf_importances
