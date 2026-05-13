"""
evaluate.py

Runs the full model evaluation pipeline on the test set and saves all results
to analysis/results/. Run this script before opening model_evaluation.ipynb.

Usage:
    python analysis/evaluate.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_ANALYSIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_ANALYSIS_DIR))
import analysis_utils as au

RESULTS_DIR = _ANALYSIS_DIR / "results"
RUNS_DIR = au.RUNS_DIR


def _load_training_history():
    cup_df = pd.read_csv(RUNS_DIR / "cup_256_unet" / "training_history.csv")
    disc_df = pd.read_csv(RUNS_DIR / "disc_256_unet" / "training_history.csv")
    cup_best = int(cup_df.loc[cup_df["val_dice_coef"].idxmax(), "epoch"])
    disc_best = int(disc_df.loc[disc_df["val_dice_coef"].idxmax(), "epoch"])
    return cup_df, disc_df, cup_best, disc_best


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading training history...")
    cup_df, disc_df, cup_best, disc_best = _load_training_history()

    print("Saving training diagnostic plots...")
    fig = au.plot_training_curves(cup_df, disc_df, cup_best, disc_best)
    fig.savefig(RESULTS_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = au.plot_lr_schedule(cup_df, disc_df)
    fig.savefig(RESULTS_DIR / "lr_schedule.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Loading models and running inference (this may take a few minutes on CPU)...")
    cup_model = au.load_model("cup")
    disc_model = au.load_model("disc")

    cup_stacks, cup_masks, cup_ids = au.load_test_arrays("cup")
    disc_stacks, disc_masks, disc_ids = au.load_test_arrays("disc")

    cup_preds = au.predict_test(cup_model, cup_stacks)
    disc_preds = au.predict_test(disc_model, disc_stacks)

    print("Computing metrics...")
    cup_dice = au.per_sample_dice(cup_masks, cup_preds)
    disc_dice = au.per_sample_dice(disc_masks, disc_preds)

    cup_mean, cup_ci_lo, cup_ci_hi = au.bootstrap_ci(cup_dice)
    disc_mean, disc_ci_lo, disc_ci_hi = au.bootstrap_ci(disc_dice)

    cup_best_i, cup_med_i, cup_worst_i = au.bmw_indices(cup_dice)
    disc_best_i, disc_med_i, disc_worst_i = au.bmw_indices(disc_dice)

    print("Saving result plots...")
    fig = au.plot_violin(cup_dice, disc_dice)
    fig.savefig(RESULTS_DIR / "dice_violin.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = au.plot_overlay_figure(
        cup_stacks, cup_masks, cup_preds,
        indices=[cup_best_i, cup_med_i, cup_worst_i],
        title="Cup Model: Best / Median / Worst Test Samples",
        dice_scores=cup_dice,
    )
    fig.savefig(RESULTS_DIR / "cup_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = au.plot_overlay_figure(
        disc_stacks, disc_masks, disc_preds,
        indices=[disc_best_i, disc_med_i, disc_worst_i],
        title="Disc Model: Best / Median / Worst Test Samples",
        dice_scores=disc_dice,
    )
    fig.savefig(RESULTS_DIR / "disc_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saving metrics.json...")
    cup_best_idx = cup_df["val_dice_coef"].idxmax()
    disc_best_idx = disc_df["val_dice_coef"].idxmax()
    results = {
        "cup": {
            "best_epoch": cup_best,
            "best_val_dice": float(cup_df.loc[cup_best_idx, "val_dice_coef"]),
            "train_dice_at_best": float(cup_df.loc[cup_best_idx, "dice_coef"]),
            "mean_dice": cup_mean,
            "ci_low": cup_ci_lo,
            "ci_high": cup_ci_hi,
            "stats": au.summary_stats(cup_dice),
            "bmw_indices": [cup_best_i, cup_med_i, cup_worst_i],
            "dice_scores": cup_dice.tolist(),
            "sample_ids": cup_ids,
        },
        "disc": {
            "best_epoch": disc_best,
            "best_val_dice": float(disc_df.loc[disc_best_idx, "val_dice_coef"]),
            "train_dice_at_best": float(disc_df.loc[disc_best_idx, "dice_coef"]),
            "mean_dice": disc_mean,
            "ci_low": disc_ci_lo,
            "ci_high": disc_ci_hi,
            "stats": au.summary_stats(disc_dice),
            "bmw_indices": [disc_best_i, disc_med_i, disc_worst_i],
            "dice_scores": disc_dice.tolist(),
            "sample_ids": disc_ids,
        },
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Results saved to {RESULTS_DIR}/")
    print(f"  Cup  Dice: {cup_mean:.4f}  95% CI [{cup_ci_lo:.4f}, {cup_ci_hi:.4f}]")
    print(f"  Disc Dice: {disc_mean:.4f}  95% CI [{disc_ci_lo:.4f}, {disc_ci_hi:.4f}]")


if __name__ == "__main__":
    main()
