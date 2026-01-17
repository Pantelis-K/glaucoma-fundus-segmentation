# Glaucoma Fundus Segmentation (U-Net) — ORIGA

This repository contains code for a dissertation project on optic **cup** and **disc** segmentation from retinal fundus images using a **U-Net** architecture. The resulting segmentations are used to derive geometric features such as the **Cup-to-Disc Ratio (CDR)**, which is widely used in glaucoma screening.

The dataset used in this work is **ORIGA** (Online Retinal Fundus Image Database for Glaucoma Analysis).

The training pipeline operates on **multi-channel input stacks** constructed from:
- **RGB** fundus image (3 channels)
- **CLAHE**-enhanced grayscale image (1 channel)
- **Sobel** edge map (1 channel)

> **Note:** The ORIGA dataset is not included in this repository. Only code and the expected data layout are provided.

---

## Repository structure

```text
assets/                # figures for README / report (overlays, curves)
data/                  # local data (not committed)
  ORIGA/               # raw ORIGA dataset
    images/            # original fundus images
    masks/             # original annotation masks (single mask, multiple labels)
  images/              # jitter-cropped fundus images
  masks/               # corresponding jitter-cropped masks
  stacks/              # generated multi-channel stacks (.npy)
notebooks/             # optional analysis notebooks
runs/                  # outputs saved per experiment run
scripts/               # helper scripts (plotting, demo inference, etc.)
src/                   # main source code
  preprocess/          # offline preprocessing utilities
  metrics.py           # custom losses / metrics
  model.py             # U-Net architecture definition
  trainer.py           # training pipeline (tf.data + callbacks)

This repository is a work in progress - being cleaned and refactored for reproducibility. Data and training outputs are not included.