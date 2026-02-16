import torch
import os
import glob
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys

# --- Monodepth2 Imports ---
# These imports require the 'monodepth2' repo to be cloned.
from torchvision import transforms 
# We will add the repo to the path in main() before importing networks
# --- End Monodepth2 Imports ---


# --- 1. CONFIGURATION ---
# !!! PLEASE UPDATE THESE PATHS !!!
CITYSCAPES_BASE_DIR = "/path/to/cityscapes/dataset" 
SPLIT = "val"  # 'train' or 'val'

# --- MODEL SELECTION ---
# This must be the path to the monodepth2 repo you cloned
MONODEPTH2_REPO_PATH = "./monodepth2" 

# Select the monodepth2 model to use (downloaded by their script)
# Options: "mono_640x192", "stereo_640x192", "mono+stereo_640x192"
MODEL_NAME = "mono+stereo_640x192" 
# --- END MODEL SELECTION ---


# --- DYNAMIC OUTPUT DIR ---
OUTPUT_DIR = f"./cityscapes_output_monodepth2_{MODEL_NAME.replace('/', '_')}"
# --- END DYNAMIC OUTPUT DIR ---


CITYSCAPES_IMG_SIZE = (1024, 2048)  # h, w
DEPTH_MIN_METERS = 0.1  # Min distance for valid ground truth
DEPTH_MAX_METERS = 80.0  # Max distance for valid ground truth

# --- CONFIGURATION ---
# Total number of images to process. Set to -1 for all.
TOTAL_IMAGES_TO_PROCESS = 100 # Set to 50 for a quick test, -1 for all

# Number of images to use for calibration.
ALIGNMENT_SAMPLE_SIZE = 20
# --- END CONFIGURATION ---


# --- 2. HELPER FUNCTIONS ---

def load_monodepth_model(model_name, repo_path):
    """Loads a monodepth2 model checkpoint."""
    print(f"Loading model: {model_name}...")
    
    # Add monodepth2 repo to Python path
    sys.path.insert(0, repo_path)
    try:
        from monodepth2 import networks
    except ImportError:
        print(f"Error: Could not import from '{repo_path}'.")
        print("Please ensure you have cloned 'monodepth2' into that directory.")
        raise

    model_path = os.path.join(repo_path, "models", model_name)
    encoder_path = os.path.join(model_path, "encoder.pth")
    decoder_path = os.path.join(model_path, "depth.pth")

    if not (os.path.exists(encoder_path) and os.path.exists(decoder_path)):
        print(f"Error: Model files not found in {model_path}")
        print("Please run 'download_models.sh' inside the 'monodepth2' directory.")
        raise FileNotFoundError
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Load Encoder ---
    encoder = networks.ResnetEncoder(18, False)
    
    # Load the state dict, which contains metadata
    # We don't use weights_only=True here because we *need* to read the metadata.
    # We trust this file since we downloaded it from the official repo.
    encoder_dict = torch.load(encoder_path, map_location=device)
    
    # Extract metadata
    feed_height = encoder_dict.get('height', 192) # Default to 192 if key missing
    feed_width = encoder_dict.get('width', 640)   # Default to 640 if key missing
    
    # Pop metadata keys so we can load the state_dict (this fixes the error)
    encoder_dict.pop("height", None)
    encoder_dict.pop("width", None)
    encoder_dict.pop("use_stereo", None)
    
    # Load the cleaned state dict
    encoder.load_state_dict(encoder_dict)
    encoder.to(device)
    encoder.eval()

    # --- Load Decoder ---
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    
    # The decoder file is just a state_dict, so we can use weights_only=True
    # to address the security warning.
    decoder_state_dict = torch.load(decoder_path, map_location=device, weights_only=True)
    depth_decoder.load_state_dict(decoder_state_dict)
    depth_decoder.to(device)
    depth_decoder.eval()
    
    print(f"Model loaded onto device: {device}")
    # Return the feed height and width
    return device, encoder, depth_decoder, feed_height, feed_width

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

def get_scale_and_shift_alignment(prediction, gt):
    """
    Finds the optimal scale and shift (s, t) to align
    prediction to gt (gt = s * prediction + t) using least squares.
    """
    # Create mask for valid ground truth pixels
    mask = (gt > DEPTH_MIN_METERS) & (gt < DEPTH_MAX_METERS)
    if not np.any(mask):
        return 1.0, 0.0 # Return default values if no valid gt

    # Flatten and apply mask
    gt_valid = gt[mask]
    pred_valid = prediction[mask]

    # Create the A matrix for least squares (A * [s, t]' = gt)
    A = np.vstack([pred_valid, np.ones_like(pred_valid)]).T

    # Solve for [s, t]
    try:
        scale, shift = np.linalg.lstsq(A, gt_valid, rcond=None)[0]
    except np.linalg.LinAlgError:
        # Fallback in case of singular matrix
        scale, shift = 1.0, 0.0

    return scale, shift

def save_depth_visualization(depth_map, filepath, cmap='magma'):
    """
    Saves a normalized, colorized depth map visualization.
    Clips to max depth for consistent coloring.
    """
    normalized_depth = np.clip(depth_map, DEPTH_MIN_METERS, DEPTH_MAX_METERS)
    normalized_depth = normalized_depth / DEPTH_MAX_METERS
    plt.imsave(filepath, normalized_depth, cmap=cmap)

# --- 3. MAIN PROCESSING SCRIPT ---

def main():
    # Update how the model is loaded to get feed_height and feed_width
    device, encoder, depth_decoder, feed_height, feed_width = load_monodepth_model(
        MODEL_NAME, MONODEPTH2_REPO_PATH
    )
    
    print(f"Using output directory: {OUTPUT_DIR}")
    
    file_paths = get_cityscapes_paths(CITYSCAPES_BASE_DIR, SPLIT)
    
    if not file_paths:
        return
        
    # --- Prepare image pre-processing ---
    # The feed_height and feed_width are now returned from the load function
    print(f"Model requires input size: {feed_height}h x {feed_width}w")
    
    # We will resize to feed_width, feed_height, and then use ToTensor
    preprocess = transforms.Compose([
        transforms.Resize((feed_height, feed_width), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor()
    ])
    
    
    # --- Handle slicing for total images and calibration ---
    total_images = len(file_paths)
    if TOTAL_IMAGES_TO_PROCESS > 0 and TOTAL_IMAGES_TO_PROCESS < total_images:
        total_images = TOTAL_IMAGES_TO_PROCESS
    
    # Ensure calibration size isn't larger than total images
    num_for_calibration = min(ALIGNMENT_SAMPLE_SIZE, total_images)
    
    if num_for_calibration <= 0:
        print("Error: ALIGNMENT_SAMPLE_SIZE or TOTAL_IMAGES_TO_PROCESS must be greater than 0.")
        return
        
    calibration_paths = file_paths[:num_for_calibration]
    processing_paths = file_paths[:total_images]

    print(f"\n--- Slicing dataset ---")
    print(f"  Total images to process: {total_images}")
    print(f"  Images for calibration: {num_for_calibration}")

    os.makedirs(os.path.join(OUTPUT_DIR, "gt_metric_viz"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "pred_relative_viz"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "pred_metric_viz"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "gt_metric_npy"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "pred_metric_npy"), exist_ok=True)
    
    
    # --- 1. CALIBRATION LOOP ---
    print(f"\n--- Starting calibration loop on first {num_for_calibration} images ---")
    scales = []
    shifts = []
    
    for i, (img_path, disp_path, cam_path) in enumerate(calibration_paths):
        print(f"Calibrating [{i+1}/{num_for_calibration}]: {os.path.basename(img_path)}", end='\r')
        try:
            image = Image.open(img_path).convert('RGB')
            gt_depth_metric = compute_gt_depth_cityscapes(disp_path, cam_path)
            
            # Pre-process image
            input_image = preprocess(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = encoder(input_image)
                outputs = depth_decoder(features)
                
                # Get disparity (scale 0 is full res)
                disp = outputs[("disp", 0)]
                
                # Upsample disparity to original image size
                disp_resized = torch.nn.functional.interpolate(
                    disp,
                    (CITYSCAPES_IMG_SIZE[0], CITYSCAPES_IMG_SIZE[1]),
                    mode="bilinear",
                    align_corners=False
                )
            
            # --- Convert Disparity to Depth ---
            # monodepth2 outputs disparity, which is inversely proportional to depth
            pred_disp = disp_resized.squeeze().cpu().numpy()
            
            # Avoid division by zero
            pred_disp[pred_disp <= 1e-7] = 1e-7
            pred_depth_relative = 1.0 / pred_disp 
            
            # Find alignment
            scale, shift = get_scale_and_shift_alignment(pred_depth_relative, gt_depth_metric)
            scales.append(scale)
            shifts.append(shift)
            
        except Exception as e:
            print(f"\n  ! ERROR during calibration on {img_path}: {e}")
    
    # Calculate global alignment factor (use median for robustness)
    global_scale = np.median(scales)
    global_shift = np.median(shifts)
    
    print(f"\n--- Calibration Complete ---")
    print(f"  Global Scale (median): {global_scale:.4f}")
    print(f"  Global Shift (median): {global_shift:.4f}")
    
    
    # --- 2. MAIN PROCESSING LOOP ---
    print(f"\n--- Starting main processing on all {total_images} images ---")
    
    for i, (img_path, disp_path, cam_path) in enumerate(processing_paths):
        print(f"Processing [{i+1}/{total_images}]: {os.path.basename(img_path)}", end='\r')
        try:
            image = Image.open(img_path).convert('RGB')
            gt_depth_metric = compute_gt_depth_cityscapes(disp_path, cam_path)
            
            # Pre-process image
            input_image = preprocess(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = encoder(input_image)
                outputs = depth_decoder(features)
                disp = outputs[("disp", 0)]
                disp_resized = torch.nn.functional.interpolate(
                    disp,
                    (CITYSCAPES_IMG_SIZE[0], CITYSCAPES_IMG_SIZE[1]),
                    mode="bilinear",
                    align_corners=False
                )
            
            pred_disp = disp_resized.squeeze().cpu().numpy()
            pred_disp[pred_disp <= 1e-7] = 1e-7
            pred_depth_relative = 1.0 / pred_disp
            
            # --- Apply Global Alignment ---
            pred_depth_metric = (global_scale * pred_depth_relative) + global_shift
            
            base_name = os.path.basename(img_path).replace("_leftImg8bit.png", "")
            
            # --- Save Visualizations ---
            save_depth_visualization(
                gt_depth_metric,
                os.path.join(OUTPUT_DIR, "gt_metric_viz", f"{base_name}_gt_metric.png")
            )
            # Save the *unaligned* relative depth for comparison
            save_depth_visualization(
                pred_depth_relative,
                os.path.join(OUTPUT_DIR, "pred_relative_viz", f"{base_name}_pred_relative.png")
            )
            # Save the final *aligned* metric depth
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

