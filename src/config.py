"""
config.py
Central configuration for the glaucom-fundus-segmentation project.

All shared constants and directory paths live here. If the project is moved
or its folder structure changes, this is the only file that needs editing.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout — resolved relative to this file so it works from any cwd
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parents[1]
SRC_DIR        = PROJECT_ROOT / "src"
DATA_DIR       = PROJECT_ROOT / "data"
RUNS_DIR       = PROJECT_ROOT / "runs"
IMAGES_DIR     = DATA_DIR / "images"
MASKS_DIR      = DATA_DIR / "masks"
STACKS_DIR     = DATA_DIR / "stacks"
ORIGA_DIR      = DATA_DIR / "ORIGA"
ORIGA_MASK_DIR = ORIGA_DIR / "masks"
ORIGA_INFO_CSV = ORIGA_DIR / "origa_info.csv"

# ---------------------------------------------------------------------------
# Model checkpoints
# ---------------------------------------------------------------------------
DISC_MODEL_PATH = RUNS_DIR / "disc_256_unet" / "best.h5"
CUP_MODEL_PATH  = RUNS_DIR / "cup_256_unet"  / "best.h5"

# ---------------------------------------------------------------------------
# Model architecture and data format
# ---------------------------------------------------------------------------
IMAGE_SIZE = 256  # spatial resolution fed to the U-Net
N_CHANNELS = 5    # R, G, B, CLAHE-gray, Sobel
CUP_VALUE  = 2    # pixel value for the cup region in ORIGA annotation masks

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Preprocessing pipeline — create_stacks.py and inference.py must match
# ---------------------------------------------------------------------------
CLIP_LIMIT  = 2.0  # CLAHE clip limit
TILE_GRID   = 8    # CLAHE tile grid size
SOBEL_KSIZE = 3    # Sobel kernel size
