"""
feature_importance.py

Extracts geometric features from ORIGA masks, runs univariate and multivariate
analysis, and saves results to analysis/results/. Run this script before
opening feature_importance.ipynb.

Usage:
    python analysis/feature_importance.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

_ANALYSIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_ANALYSIS_DIR))
import analysis_utils as au

RESULTS_DIR = _ANALYSIS_DIR / "results"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    mask_dir = au.DATA_DIR / "ORIGA" / "masks"
    label_csv = au.DATA_DIR / "ORIGA" / "origa_info.csv"

    print("Extracting geometric features from ORIGA masks...")
    df = au.extract_origa_features(mask_dir, label_csv)

    print("Running univariate ROC analysis...")
    df_stats, rocs = au.univariate_roc_analysis(df, au.UNI_METRICS)

    fig = au.plot_univariate_roc(df_stats, rocs)
    fig.savefig(RESULTS_DIR / "univariate_roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Running cross-validated multivariate models...")
    X = df[au.UNI_METRICS].values
    y = df.label.astype(int).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf_cv = au.build_rf_pipeline()
    lr_cv = au.build_lr_pipeline()
    rf_auc = au.run_cv_auc(rf_cv, X, y, skf)
    lr_auc = au.run_cv_auc(lr_cv, X, y, skf)

    # Refit on full data to extract feature weights
    rf_full = au.build_rf_pipeline()
    lr_full = au.build_lr_pipeline()
    lr_coefs, rf_importances = au.fit_and_explain(lr_full, rf_full, X, y, au.UNI_METRICS)

    print("Saving feature_results.json...")
    results = {
        "univariate": df_stats.to_dict(orient="records"),
        "cv_auc": {
            "random_forest": rf_auc,
            "logistic_regression": lr_auc,
        },
        "lr_coefficients": lr_coefs.to_dict(),
        "rf_importances": rf_importances.to_dict(),
    }
    with open(RESULTS_DIR / "feature_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Results saved to {RESULTS_DIR}/")
    print(f"  RF CV-AUC: {rf_auc:.4f}")
    print(f"  LR CV-AUC: {lr_auc:.4f}")
    print("\nTop features by LR coefficient:")
    print(lr_coefs.to_string())
    print("\nTop features by RF importance:")
    print(rf_importances.to_string())


if __name__ == "__main__":
    main()
