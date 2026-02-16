A Comparative Analysis of Monocular Depth Estimation Models on Cityscapes

# Project Overview

This project provides a complete Python pipeline for benchmarking and evaluating various state-of-the-art monocular depth estimation (MDE) models on the Cityscapes dataset.

The primary challenge in this field is that many models (like MiDaS and MonoDepth2) output relative depth, which is not in meters and cannot be directly compared to ground truth. This project implements the full methodology to:

**Generate Metric Ground Truth:** Load Cityscapes disparity and camera calibration data to calculate the ground-truth metric depth for every pixel.

**Align Relative Models:** Implement a robust scale-and-shift alignment (Ground_Truth = s * Prediction + t) by calibrating on a sample of images and applying a global factor.

**Evaluate Metric Models:** Directly test models like ZoeDepth that output metric depth out-of-the-box.

**Compare Performance:** Run all models through a single, standardized evaluation script to compute 9 standard error and accuracy metrics (AbsRel, RMSE, d1, etc.).

# Project Structure

    /project-directory/

    │

    ├── run_midas_depth.py           # Script 1: For MiDaS & DPT models (Hugging Face)

    ├── run_zoedepth.py              # Script 2: For ZoeDepth models (Hugging Face)

    ├── run_monodepth2.py            # Script 3: For MonoDepth2 models (GitHub Repo)

    ├── depth_evaluation.py          # Script 4: The unified evaluation script

    │

    ├── README.md                             # This file

    │

    └── monodepth2/                           # Needs to be cloned - GitHub repo for MonoDepth2

        ├── models/                           # (Contains downloaded .pth weights)
    
        └── ...
    


# Setup and Installation

## Step 1: Install Python Libraries

Install all necessary Python packages.

### Core libraries for AI and image processing
pip install torch torchvision transformers numpy pillow matplotlib

### 'scipy' is needed for the least-squares alignment in the processing scripts
pip install scipy


## Step 2: Get Cityscapes Dataset

You must have the Cityscapes dataset downloaded and extracted. The scripts require the following three folders:

leftImg8bit/: Contains the standard RGB images.

disparity/: Contains the 16-bit disparity maps for ground truth.

camera/: Contains the .json calibration files (with focal length and baseline) needed to convert disparity to metric depth.

## Step 3: Set up MonoDepth2

The MonoDepth2 model is not on Hugging Face and requires special setup.

Clone the Repository: From your project directory, clone the monodepth2 repo.

git clone [https://github.com/nianticlabs/monodepth2.git](https://github.com/nianticlabs/monodepth2.git)


Download Model Weights: cd into the new directory and run their download script to get the pre-trained models. We primarily use the mono+stereo_640x192 model.

cd monodepth2
bash download_models.sh
cd ..


# Workflow: From Processing to Evaluation

## Step 1: Configure Paths

Before running, you MUST edit the configuration variables at the top of all three processing scripts (run_midas_depth.py, run_zoedepth.py, run_monodepth2.py):

CITYSCAPES_BASE_DIR: Set this to the full path of your Cityscapes dataset folder (e.g., datasets/cityscapes).

MONODEPTH2_REPO_PATH (in run_monodepth2.py only): Set this to the full path of the monodepth2 folder you just cloned.

## Step 2: Run Model Processing

Run the processing scripts one by one to generate the raw prediction files. Each script will create a new, model-specific output folder (e.g., ./cityscapes_output_dpt-beit-large-512/).

A. To run a MiDaS/DPT model:

Open run_midas_depth.py.

Uncomment the MODEL_NAME you wish to test.

(Optional) Change ALIGNMENT_SAMPLE_SIZE or TOTAL_IMAGES_TO_PROCESS.

Run the script:

python run_midas_depth.py


B. To run a ZoeDepth model:

Open run_zoedepth.py.

Uncomment the MODEL_NAME you wish to test.

(Optional) Change TOTAL_IMAGES_TO_PROCESS.

Run the script:

python run_zoedepth.py


C. To run the MonoDepth2 model:

Open run_monodepth2.py.

(Optional) Change ALIGNMENT_SAMPLE_SIZE or TOTAL_IMAGES_TO_PROCESS.

Run the script:

python run_monodepth2.py


## Step 3: Run Evaluation

After a processing script finishes, use depth_evaluation.py to get the final metrics. This script takes the output directory as a command-line argument.

### Example for the DPT-BEiT model
python depth_evaluation.py --dir ./cityscapes_output_dpt-beit-large-512

### Example for the MonoDepth2 model
python depth_evaluation.py --dir ./cityscapes_output_mono+stereo_640x192


This will print a formatted table of all 9 error and accuracy metrics.

# Script-by-Script Explanation

### _run_midas_depth.py_

Purpose: Runs all relative depth models from Hugging Face (MiDaS v2.1, DPT, MiDaS v3.1).

Key Logic:

Loads: A model from the Intel/ organization (e.g., dpt-beit-large-512).

Calibrates: Runs on the first N (ALIGNMENT_SAMPLE_SIZE) images, solving GT = s * Pred + t for each one to get N scales and N shifts.

Aligns: Calculates the median scale and shift from the calibration phase. This single global_scale and global_shift is then applied to all images processed by the script.

Saves: For each image, it saves the metric ground truth (_gt.npy) and the final, aligned metric prediction (_pred.npy) to a model-specific output folder.

### _run_zoedepth.py_

Purpose: Runs all metric depth models from Hugging Face (ZoeDepth).

Key Logic:

Loads: A ZoeDepth model (e.g., Intel/zoedepth-nyu-kitti).

No Alignment: This is the key difference. ZoeDepth is trained to output meters directly, so no alignment is performed.

Saves: For each image, it saves the metric ground truth (_gt.npy) and the model's raw metric prediction (_pred.npy).

### _run_monodepth2.py_

Purpose: Runs the relative depth monodepth2 model from the nianticlabs GitHub repository.

Key Logic:

Loads: Imports the monodepth2 library from the cloned folder and loads the .pth model weights.

Aligns: Uses the exact same calibration and alignment logic as run_midas_depth.py (median of N samples) to ensure a fair comparison.

Saves: For each image, it saves the metric ground truth (_gt.npy) and the final, aligned metric prediction (_pred.npy).

### _depth_evaluation.py_

Purpose: The single, unified tool for calculating results for any model.

Key Logic:

Loads: Takes a directory path (e.g., ./cityscapes_output_...) from the command line (--dir).

Gathers Data: Scans the directory, loads all _gt.npy and _pred.npy file pairs, and collects all valid pixels.

Masks: Applies the standard evaluation mask (e.g., 0.1m to 80m) to both the ground truth and prediction data.

Calculates: Computes and prints the final aggregate table for all 9 metrics (AbsRel, SqRel, RMSE, RMSElog, log10, SILog, d1, d2, d3).

