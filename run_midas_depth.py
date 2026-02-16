import torch
import os
import glob
import json
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
# !!! PLEASE UPDATE THESE PATHS !!!
# Use the full path for your WSL environment
CITYSCAPES_BASE_DIR = "/path/to/cityscapes/dataset" 
SPLIT = "val"  # 'train' or 'val'

# --- NEW MODEL SELECTION ---
# To use a different model from Hugging Face that outputs relative depth,
# you can simply UNCOMMENT one of the lines below.

# === MiDaS v3.1 Models (BEiT Backbone) ===
MODEL_NAME = "Intel/dpt-beit-large-512"  # (Highest quality)
# MODEL_NAME = "Intel/dpt-beit-large-384"
# MODEL_NAME = "Intel/dpt-beit-base-384"

# === MiDaS v3.1 Models (SwinV2 Backbone) ===
# MODEL_NAME = "Intel/dpt-swinv2-large-384"
# MODEL_NAME = "Intel/dpt-swinv2-base-384"
# MODEL_NAME = "Intel/dpt-swinv2-tiny-256" # (Fastest v3.1 DPT model)

# === MiDaS v3.0 Models (Original DPT) ===
# MODEL_NAME = "Intel/dpt-large"

# === MiDaS v2.1 Models (Hybrid and Convolutional) ===
# MODEL_NAME = "Intel/dpt-hybrid-midas"
# MODEL_NAME = "Intel/midas-v21-384"
# MODEL_NAME = "Intel/midas-v21-small-256"

# --- END NEW MODEL SELECTION ---

# --- DYNAMIC OUTPUT DIR ---
# Results will be saved in a folder named after the model,
# e.g., ./cityscapes_output_dpt-beit-large-512
OUTPUT_DIR = f"./cityscapes_output_{MODEL_NAME.split('/')[-1]}"
# --- END DYNAMIC OUTPUT DIR ---

CITYSCAPES_IMG_SIZE = (1024, 2048)  # h, w
DEPTH_MIN_METERS = 0.1  # Min distance for valid ground truth
DEPTH_MAX_METERS = 80.0  # Max distance for valid ground truth

# --- NEW CONFIGURATION ---
# Number of images to use from the *start* of the dataset to calculate 
# the global alignment factors.
ALIGNMENT_SAMPLE_SIZE = 20 

# --- NEW CONFIGURATION ---
# Total number of images to process from the dataset.
# Set to -1 to process all available images.
TOTAL_IMAGES_TO_PROCESS = 100

# --- 2. HELPER FUNCTIONS ---

def load_model():
    """Loads the MiDaS model and image processor."""
    print(f"Loading model: {MODEL_NAME}...")
    # Use a DPT model (MiDaS v3)
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME, use_safetensors=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded onto device: {device}")
    return device, model, image_processor

def get_cityscapes_paths(base_dir, split):
    """
    Finds all matching image, disparity, and camera files for a given split.
    """
    print(f"Scanning for Cityscapes files in: {base_dir}")
    
    # Define file patterns
    img_pattern = os.path.join(base_dir, "leftImg8bit", split, "*", "*_leftImg8bit.png")
    disp_pattern_template = os.path.join(base_dir, "disparity", split, "{city}", "{id}_disparity.png")
    cam_pattern_template = os.path.join(base_dir, "camera", split, "{city}", "{id}_camera.json")

    # --- [DEBUG] ---
    debug_split_folder = os.path.join(base_dir, "leftImg8bit", split)
    print(f"  [DEBUG] Checking for split folder at: {debug_split_folder}")
    print(f"  [DEBUG] Does it exist? {os.path.exists(debug_split_folder)}")
    
    debug_city_pattern = os.path.join(base_dir, "leftImg8bit", split, "*")
    print(f"  [DEBUG] Checking for city folders with pattern: {debug_city_pattern}")
    city_folders_found = glob.glob(debug_city_pattern)
    print(f"  [DEBUG] Found {len(city_folders_found)} potential city folders.")
    # --- [END DEBUG] ---
    
    print(f"  [DEBUG] Using full glob pattern: {img_pattern}")
    
    image_paths = sorted(glob.glob(img_pattern))
    
    print(f"Found {len(image_paths)} images. Matching files...")
    
    file_paths = []
    for img_path in image_paths:
        # Extract city and id from the image path
        city = os.path.basename(os.path.dirname(img_path))
        id = os.path.basename(img_path).replace("_leftImg8bit.png", "")
        
        # Construct the corresponding disparity and camera paths
        disp_path = disp_pattern_template.format(city=city, id=id)
        cam_path = cam_pattern_template.format(city=city, id=id)
        
        # Check if all three files exist
        if os.path.exists(disp_path) and os.path.exists(cam_path):
            file_paths.append((img_path, disp_path, cam_path))
        else:
            print(f"  > Warning: Missing disparity/camera file for {id}. Skipping.")

    print(f"Found {len(file_paths)} matching file sets for split '{split}'.")
    if len(file_paths) == 0:
        print("Error: No files found. Please check 'CITYSCAPES_BASE_DIR' and dataset structure.")
        
    return file_paths

def compute_gt_depth_cityscapes(disparity_path, camera_path):
    """
    Calculates the metric ground truth depth map from Cityscapes
    disparity and camera files.
    """
    # Load the 16-bit disparity map
    disparity_img = Image.open(disparity_path)
    disparity_data = np.asarray(disparity_img, dtype=np.float32)
    
    # Convert disparity data to actual disparity values
    # (Values are 0 or > 0, 0 is invalid)
    valid_mask = disparity_data > 0
    disparity_values = np.zeros_like(disparity_data)
    disparity_values[valid_mask] = (disparity_data[valid_mask] - 1.0) / 256.0
    
    # Load camera intrinsics and extrinsics
    with open(camera_path, 'r') as f:
        camera_data = json.load(f)
        
    baseline = camera_data['extrinsic']['baseline']  # in meters
    
    # Handle different keys for focal length ('fx', 'focalLength', or from 'K')
    try:
        if 'fx' in camera_data['intrinsic']:
            focal_length = camera_data['intrinsic']['fx'] # Most common
        elif 'focalLength' in camera_data['intrinsic']:
            focal_length = camera_data['intrinsic']['focalLength']
        elif 'K' in camera_data['intrinsic']:
            focal_length = camera_data['intrinsic']['K'][0][0] # From matrix
        else:
            raise KeyError("No focal length key (fx, focalLength, or K) found in camera JSON.")
    except KeyError as e:
        print(f"Error reading {camera_path}: {e}")
        raise
        
    # Depth = (Baseline * Focal Length) / Disparity
    # We must avoid division by zero for invalid pixels
    metric_depth_values = np.zeros_like(disparity_values)
    metric_depth_values[valid_mask] = (baseline * focal_length) / disparity_values[valid_mask]
    
    return metric_depth_values

def get_scale_and_shift_alignment(pred_depth_relative, gt_depth_metric):
    """
    Calculates the optimal scale (s) and shift (t) to align the
    predicted depth with the ground truth, using least-squares.
    
    Finds (s, t) to minimize: || gt - (s * pred + t) ||^2
    """
    
    # Create the evaluation mask (only use valid GT pixels)
    valid_mask = (gt_depth_metric > DEPTH_MIN_METERS) & (gt_depth_metric < DEPTH_MAX_METERS)
    
    # Get the 1D arrays of valid pixels
    gt_pixels_valid = gt_depth_metric[valid_mask]
    pred_pixels_valid = pred_depth_relative[valid_mask]

    if pred_pixels_valid.size == 0 or gt_pixels_valid.size == 0:
        print("  > Warning: No valid pixels for alignment, using default s=1, t=0.")
        return 1.0, 0.0

    # We want to solve a linear system.
    # We can write it as:
    # [ P_1  1 ] [ s ] = [ G_1 ]
    # [ P_2  1 ] [ t ]   [ G_2 ]
    # [ ...  ... ]       [ ... ]
    #
    # A * x = B
    # A = [pred_pixels_valid, 1]
    # x = [s, t]
    # B = [gt_pixels_valid]

    # Create the A matrix (N x 2)
    A = np.vstack([pred_pixels_valid, np.ones(len(pred_pixels_valid))]).T
    
    # Create the B matrix (N x 1)
    B = gt_pixels_valid
    
    # Solve for x = [s, t] using least-squares
    # This finds the 'x' that minimizes the squared Euclidean 2-norm ||B - A*x||^2
    solution, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    
    scale = solution[0]
    shift = solution[1]
    
    return scale, shift

def save_depth_visualization(depth_map, filepath, cmap='magma'):
    """
    Saves a normalized, colorized depth map visualization.
    Clips to max depth for consistent coloring.
    """
    # Clip to max depth and normalize for visualization
    # We clip *before* normalizing
    normalized_depth = np.clip(depth_map, DEPTH_MIN_METERS, DEPTH_MAX_METERS)
    
    # Normalize 0-1
    normalized_depth = normalized_depth / DEPTH_MAX_METERS
    
    # Use matplotlib to create a colormap
    plt.imsave(filepath, normalized_depth, cmap=cmap)

# --- 3. MAIN PROCESSING SCRIPT ---

def main():
    device, model, image_processor = load_model()
    file_paths = get_cityscapes_paths(CITYSCAPES_BASE_DIR, SPLIT)
    
    if not file_paths:
        return
    
    # --- NEW: Slice file_paths list based on TOTAL_IMAGES_TO_PROCESS ---
    if TOTAL_IMAGES_TO_PROCESS > 0 and TOTAL_IMAGES_TO_PROCESS < len(file_paths):
        print(f"\n--- Slicing dataset: Processing only first {TOTAL_IMAGES_TO_PROCESS} images ---")
        file_paths = file_paths[:TOTAL_IMAGES_TO_PROCESS]
    elif TOTAL_IMAGES_TO_PROCESS == -1:
         print(f"\n--- Processing all {len(file_paths)} found images ---")
    else:
        print(f"\n--- Processing all {len(file_paths)} found images (TOTAL_IMAGES_TO_PROCESS set >= total) ---")
    # --- END NEW ---
        
    # Ensure all output directories exist
    os.makedirs(os.path.join(OUTPUT_DIR, "gt_metric_viz"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "pred_metric_viz"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "pred_relative_viz"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "gt_metric_npy"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "pred_metric_npy"), exist_ok=True)

    # --- NEW STEP 1: CALIBRATION ---
    print(f"\n--- Starting calibration on first {ALIGNMENT_SAMPLE_SIZE} images (or total, if fewer) ---")
    scale_samples = []
    shift_samples = []
    
    # Determine the number of images to use for calibration
    # This correctly takes from the *already sliced* file_paths
    num_calibration_images = min(ALIGNMENT_SAMPLE_SIZE, len(file_paths)) 
    if num_calibration_images == 0:
        print("Error: No images found to calibrate on.")
        return

    for i in range(num_calibration_images):
        img_path, disp_path, cam_path = file_paths[i]
        print(f"  Calibrating with: {os.path.basename(img_path)}")
        try:
            # Load image
            image = Image.open(img_path)
            
            # Compute ground truth depth
            gt_depth_metric = compute_gt_depth_cityscapes(disp_path, cam_path)
            
            # Run model
            with torch.no_grad():
                inputs = image_processor(images=image, return_tensors="pt").to(device)
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth
                
            # Interpolate to original size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=CITYSCAPES_IMG_SIZE,
                mode="bicubic",
                align_corners=False,
            )
            
            # Get relative depth map
            relative_prediction = prediction.squeeze().cpu().numpy()
            
            # Get scale and shift for *this image*
            scale, shift = get_scale_and_shift_alignment(
                relative_prediction,
                gt_depth_metric
            )
            
            scale_samples.append(scale)
            shift_samples.append(shift)
            
        except Exception as e:
            print(f"  ! ERROR during calibration on {img_path}: {e}")

    if not scale_samples:
        print("Error: Calibration failed for all sample images. Cannot proceed.")
        return

    # --- NEW STEP 2: Calculate Global Alignment Factors ---
    # Using median is more robust to outliers than mean
    global_scale = np.median(scale_samples)
    global_shift = np.median(shift_samples)
    
    print(f"\n--- Calibration Complete ---")
    print(f"  Calculated from {len(scale_samples)} valid samples.")
    print(f"  Global Scale (s): {global_scale:.4f}")
    print(f"  Global Shift (t): {global_shift:.4f}")

    # --- NEW STEP 3: MAIN PROCESSING LOOP ---
    print(f"\n--- Starting main processing on all {len(file_paths)} images ---")
    
    for i, (img_path, disp_path, cam_path) in enumerate(file_paths):
        # Use \r to print on the same line for a clean log
        print(f"Processing [{i+1}/{len(file_paths)}]: {os.path.basename(img_path)}", end='\r')
        try:
            # Load image
            image = Image.open(img_path)
            
            # Compute ground truth depth
            # We still need this to save the GT .npy file for evaluation
            gt_depth_metric = compute_gt_depth_cityscapes(disp_path, cam_path)
            
            # Run model
            with torch.no_grad():
                inputs = image_processor(images=image, return_tensors="pt").to(device)
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth
                
            # Interpolate
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=CITYSCAPES_IMG_SIZE,
                mode="bicubic",
                align_corners=False,
            )
            
            # Get relative depth map
            relative_prediction = prediction.squeeze().cpu().numpy()
            
            # --- APPLY GLOBAL ALIGNMENT ---
            # Instead of calculating per-image, we apply the global factors
            pred_depth_metric = (relative_prediction * global_scale) + global_shift
            
            # (Optional) Clip the metric depth to the valid range
            pred_depth_metric = np.clip(pred_depth_metric, DEPTH_MIN_METERS, DEPTH_MAX_METERS)
            
            # --- Save Results ---
            base_name = os.path.basename(img_path).replace("_leftImg8bit.png", "")
            
            # --- Save Visualizations ---
            # Save Ground Truth visualization
            save_depth_visualization(
                gt_depth_metric,
                os.path.join(OUTPUT_DIR, "gt_metric_viz", f"{base_name}_gt_metric.png")
            )
            
            # Save *Un-aligned* Relative visualization
            save_depth_visualization(
                relative_prediction,
                os.path.join(OUTPUT_DIR, "pred_relative_viz", f"{base_name}_pred_relative.png")
            )
            
            # Save *Globally-Aligned* Metric visualization
            save_depth_visualization(
                pred_depth_metric,
                os.path.join(OUTPUT_DIR, "pred_metric_viz", f"{base_name}_pred_metric.png")
            )
            
            # --- Save Raw .npy Files for Evaluation ---
            np.save(
                os.path.join(OUTPUT_DIR, "gt_metric_npy", f"{base_name}_gt_metric.npy"), 
                gt_depth_metric
            )
            np.save(
                os.path.join(OUTPUT_DIR, "pred_metric_npy", f"{base_name}_pred_metric.npy"), 
                pred_depth_metric
            )

        except Exception as e:
            # print in full to avoid \r issues
            print(f"\n  ! ERROR processing {img_path}: {e}") 
    
    # Add a newline to clear the \r
    print(f"\n--- Processing Complete ---")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

