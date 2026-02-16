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
CITYSCAPES_BASE_DIR = "/path/to/cityscapes/dataset" 
SPLIT = "val"  # 'train' or 'val'

# --- MODEL SELECTION ---
# This script is for METRIC depth models like ZoeDepth.
# Uncomment the model you wish to run.

# --- Intel-hosted models (Recommended) ---
MODEL_NAME = "Intel/zoedepth-nyu-kitti"
# MODEL_NAME = "Intel/zoedepth-nyu"
# MODEL_NAME = "Intel/zoedepth-kitti"
# --- END MODEL SELECTION ---

# --- DYNAMIC OUTPUT DIR ---
OUTPUT_DIR = f"./cityscapes_output_{MODEL_NAME.split('/')[-1]}"
# --- END DYNAMIC OUTPUT DIR ---


CITYSCAPES_IMG_SIZE = (1024, 2048)  # h, w
DEPTH_MIN_METERS = 0.1  # Min distance for valid ground truth
DEPTH_MAX_METERS = 80.0  # Max distance for valid ground truth

# --- CONFIGURATION ---
# Total number of images to process from the dataset.
# Set to -1 to process all available images.
TOTAL_IMAGES_TO_PROCESS = 100 # Set to 50 for a quick test, -1 for all

# --- 2. HELPER FUNCTIONS ---

def load_model():
    """Loads the ZoeDepth model and image processor."""
    print(f"Loading model: {MODEL_NAME}...")
    # ZoeDepth is compatible with AutoModelForDepthEstimation
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded onto device: {device}")
    return device, model, image_processor

def get_cityscapes_paths(base_dir, split):
    """
    Finds all matching image, disparity, and camera files for a given split.
    (This function is identical to the one in your other script)
    """
    print(f"Scanning for Cityscapes files in: {base_dir}")
    
    img_pattern = os.path.join(base_dir, "leftImg8bit", split, "*", "*_leftImg8bit.png")
    disp_pattern_template = os.path.join(base_dir, "disparity", split, "{city}", "{id}_disparity.png")
    cam_pattern_template = os.path.join(base_dir, "camera", split, "{city}", "{id}_camera.json")

    debug_split_folder = os.path.join(base_dir, "leftImg8bit", split)
    print(f"  [DEBUG] Checking for split folder at: {debug_split_folder}")
    print(f"  [DEBUG] Does it exist? {os.path.exists(debug_split_folder)}")
    
    debug_city_pattern = os.path.join(base_dir, "leftImg8bit", split, "*")
    city_folders_found = glob.glob(debug_city_pattern)
    print(f"  [DEBUG] Found {len(city_folders_found)} potential city folders.")
    print(f"  [DEBUG] Using full glob pattern: {img_pattern}")
    
    image_paths = sorted(glob.glob(img_pattern))
    
    print(f"Found {len(image_paths)} images. Matching files...")
    
    file_paths = []
    for img_path in image_paths:
        city = os.path.basename(os.path.dirname(img_path))
        id = os.path.basename(img_path).replace("_leftImg8bit.png", "")
        disp_path = disp_pattern_template.format(city=city, id=id)
        cam_path = cam_pattern_template.format(city=city, id=id)
        
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
    (This function is identical to the one in your other script)
    """
    disparity_img = Image.open(disparity_path)
    disparity_data = np.asarray(disparity_img, dtype=np.float32)
    
    valid_mask = disparity_data > 0
    disparity_values = np.zeros_like(disparity_data)
    disparity_values[valid_mask] = (disparity_data[valid_mask] - 1.0) / 256.0
    
    with open(camera_path, 'r') as f:
        camera_data = json.load(f)
        
    baseline = camera_data['extrinsic']['baseline']
    
    try:
        if 'fx' in camera_data['intrinsic']:
            focal_length = camera_data['intrinsic']['fx']
        elif 'focalLength' in camera_data['intrinsic']:
            focal_length = camera_data['intrinsic']['focalLength']
        elif 'K' in camera_data['intrinsic']:
            focal_length = camera_data['intrinsic']['K'][0][0]
        else:
            raise KeyError("No focal length key (fx, focalLength, or K) found in camera JSON.")
    except KeyError as e:
        print(f"Error reading {camera_path}: {e}")
        raise
        
    metric_depth_values = np.zeros_like(disparity_values)
    metric_depth_values[valid_mask] = (baseline * focal_length) / disparity_values[valid_mask]
    
    return metric_depth_values

def save_depth_visualization(depth_map, filepath, cmap='magma'):
    """
    Saves a normalized, colorized depth map visualization.
    Clips to max depth for consistent coloring.
    (This function is identical to the one in your other script)
    """
    normalized_depth = np.clip(depth_map, DEPTH_MIN_METERS, DEPTH_MAX_METERS)
    normalized_depth = normalized_depth / DEPTH_MAX_METERS
    plt.imsave(filepath, normalized_depth, cmap=cmap)

# --- 3. MAIN PROCESSING SCRIPT ---

def main():
    device, model, image_processor = load_model()
    
    print(f"Using output directory: {OUTPUT_DIR}")
    
    file_paths = get_cityscapes_paths(CITYSCAPES_BASE_DIR, SPLIT)
    
    if not file_paths:
        return
    
    if TOTAL_IMAGES_TO_PROCESS > 0 and TOTAL_IMAGES_TO_PROCESS < len(file_paths):
        print(f"\n--- Slicing dataset: Processing only first {TOTAL_IMAGES_TO_PROCESS} images ---")
        file_paths = file_paths[:TOTAL_IMAGES_TO_PROCESS]
    elif TOTAL_IMAGES_TO_PROCESS == -1:
         print(f"\n--- Processing all {len(file_paths)} found images ---")
    else:
        print(f"\n--- Processing all {len(file_paths)} found images (TOTAL_IMAGES_TO_PROCESS set >= total) ---")
        
    os.makedirs(os.path.join(OUTPUT_DIR, "gt_metric_viz"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "pred_metric_viz"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "gt_metric_npy"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "pred_metric_npy"), exist_ok=True)

    # --- NO CALIBRATION STEP ---
    # ZoeDepth outputs metric depth, so we do not need to
    # calculate a global scale and shift.
    
    print(f"\n--- Starting main processing on all {len(file_paths)} images ---")
    
    for i, (img_path, disp_path, cam_path) in enumerate(file_paths):
        print(f"Processing [{i+1}/{len(file_paths)}]: {os.path.basename(img_path)}", end='\r')
        try:
            image = Image.open(img_path)
            gt_depth_metric = compute_gt_depth_cityscapes(disp_path, cam_path)
            
            with torch.no_grad():
                inputs = image_processor(images=image, return_tensors="pt").to(device)
                outputs = model(**inputs)
                
                # outputs.predicted_depth is the raw metric depth
                # We just need to interpolate it to the original size
                prediction = torch.nn.functional.interpolate(
                    outputs.predicted_depth.unsqueeze(1),
                    size=CITYSCAPES_IMG_SIZE,
                    mode="bicubic",
                    align_corners=False,
                )
            
            # --- NO ALIGNMENT ---
            # The model's output is already in meters.
            pred_depth_metric = prediction.squeeze().cpu().numpy()
            
            base_name = os.path.basename(img_path).replace("_leftImg8bit.png", "")
            
            # --- Save Visualizations ---
            save_depth_visualization(
                gt_depth_metric,
                os.path.join(OUTPUT_DIR, "gt_metric_viz", f"{base_name}_gt_metric.png")
            )
            
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
            print(f"\n  ! ERROR processing {img_path}: {e}") 
    
    print(f"\n--- Processing Complete ---")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
