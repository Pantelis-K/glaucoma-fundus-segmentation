# Data

This folder contains the datasets and derived data used for training and evaluating the U-Net models for optic disc and optic cup segmentation on retinal fundus images.

## Dataset

This project uses the ORIGA dataset:

- Name: ORIGA (Online Retinal Fundus Image Database for Glaucoma Analysis)  
- Source: https://ieeexplore.ieee.org/document/5626137/  
- Access: The dataset can be requested from the original authors  

Only a small sample of the dataset is committed in this repository.

## Folder Structure
data/
├── ORIGA/ # Raw ORIGA dataset
│ ├── images/ # Original fundus images
│ └── masks/ # Original annotation masks (single mask with multiple labels)
├── images/ # Cropped fundus images
├── masks/ # Cropped segmentation masks
└── stacks/ # Multi-channel input stacks (.npy) + visualisation layers

## Data Generation

The processed data is generated in two main stages:

1. **Random Cropping (ROI Extraction)**  
   Fundus images are cropped around the optic disc using ground-truth annotations, followed by a small random spatial jitter.  
   This ensures consistent focus on the region of interest while preventing the model from learning fixed coordinate-based patterns.

   **Note on data leakage:**  
   This approach uses ground-truth information and is therefore not suitable for deployment.  
   In a production setting, this should be replaced with a ground-truth-agnostic ROI detection method.

   This is not a practical limitation of the pipeline, as the cropping used here is deliberately conservative.  
   Many existing ground-truth-agnostic methods are capable of producing tighter and better-centred crops.

2. **Stack Generation**  
   Cropped images are converted into multi-channel input stacks with 5 channels in the following order:
   - RGB (3 channels)  
   - CLAHE-enhanced grayscale (1 channel)  
   - Sobel gradient magnitude (1 channel)  

   The resulting stacks have shape **(5 × 640 × 640)** and are saved as `.npy` arrays, matching the expected input of the model architecture.

   For visualisation purposes, the individual CLAHE and Sobel channels are also saved as `.png` images within the `stacks/` directory.

The cropped masks are used as ground truth targets for model training.