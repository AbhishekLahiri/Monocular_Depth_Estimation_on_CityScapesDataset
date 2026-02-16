import numpy as np
import os
import glob
import argparse  # Module for command-line arguments
import sys

# --- 1. CONFIGURATION ---

# Max number of images to evaluate.
# Set to 50 for a quick test, -1 for all.
IMAGES_TO_EVALUATE = 100

# --- 2. EVALUATION METRICS IMPLEMENTATION ---
# These functions assume input data is 1D and pre-masked (all > 0)

def compute_silog_v2(gt, pred):
    """Computes SILog error (pre-masked data)"""
    # Ensure no zero or negative values remain (as a safeguard)
    safe_mask = (gt > 1e-6) & (pred > 1e-6)
    if not np.any(safe_mask):
        return 0.0
    
    d = np.log(pred[safe_mask]) - np.log(gt[safe_mask])
    return 100 * (np.mean(d ** 2) - (np.mean(d) ** 2))

def get_accuracy_metrics_v2(gt, pred):
    """Computes d1, d2, d3 accuracy (pre-masked data)"""
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()
    return d1, d2, d3

def get_error_metrics_v2(gt, pred):
    """Computes standard error metrics (pre-masked data)"""
    # RMSE
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    # RMSElog
    # Create a safe mask for log operations
    safe_mask = (gt > 1e-6) & (pred > 1e-6)
    if not np.any(safe_mask):
        rmse_log = 0.0
        log10 = 0.0
    else:
        rmse_log = (np.log(gt[safe_mask]) - np.log(pred[safe_mask])) ** 2
        rmse_log = np.sqrt(rmse_log.mean())
        log10 = np.mean(np.abs(np.log10(gt[safe_mask]) - np.log10(pred[safe_mask])))

    # AbsRel and SqRel
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    
    # SILog
    silog = compute_silog_v2(gt, pred)
    
    return abs_rel, sq_rel, rmse, rmse_log, log10, silog

# --- 3. MAIN EVALUATION FUNCTION ---

def main(args):
    # Set the base directory from the command-line argument
    BASE_DIR = args.dir
    
    if not os.path.exists(BASE_DIR):
        print(f"Error: Directory not found: {BASE_DIR}", file=sys.stderr)
        print("Please provide a valid directory using the --dir argument.", file=sys.stderr)
        sys.exit(1)
        
    gt_dir = os.path.join(BASE_DIR, "gt_metric_npy")
    pred_dir = os.path.join(BASE_DIR, "pred_metric_npy")
    
    if not os.path.exists(gt_dir) or not os.path.exists(pred_dir):
        print(f"Error: Could not find 'gt_metric_npy' or 'pred_metric_npy' folders in {BASE_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading .npy files from: {BASE_DIR}")

    pred_paths = sorted(glob.glob(os.path.join(pred_dir, "*.npy")))
    gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
    
    # Match files by basename
    file_pairs = []
    gt_map = {os.path.basename(p).replace("_gt_metric.npy", ""): p for p in gt_paths}
    
    for pred_path in pred_paths:
        base_name = os.path.basename(pred_path).replace("_pred_metric.npy", "")
        if base_name in gt_map:
            file_pairs.append((gt_map[base_name], pred_path))

    if not file_pairs:
        print("Error: Found no matching gt/pred .npy files.", file=sys.stderr)
        return
        
    print(f"Found {len(file_pairs)} .npy file pairs to evaluate.")
    
    # Slice list if needed
    if IMAGES_TO_EVALUATE > 0 and IMAGES_TO_EVALUATE < len(file_pairs):
        print(f"--- Evaluating on first {IMAGES_TO_EVALUATE} images ---")
        file_pairs = file_pairs[:IMAGES_TO_EVALUATE]
    else:
        print(f"--- Evaluating on all {len(file_pairs)} images ---")
        
    all_gt_pixels = []
    all_pred_pixels = []
    
    for i, (gt_path, pred_path) in enumerate(file_pairs):
        if (i+1) % 50 == 0:
            print(f"  ... loaded {i+1}/{len(file_pairs)} files")
            
        gt_depth = np.load(gt_path)
        pred_depth = np.load(pred_path)
        
        # Create evaluation mask
        mask = (gt_depth > 0.1) & (gt_depth < 80.0)
        
        # --- Memory Optimization ---
        # Only append the valid pixels to the list.
        # This drastically reduces memory usage.
        all_gt_pixels.append(gt_depth[mask])
        all_pred_pixels.append(pred_depth[mask])
        
    print("All files loaded. Concatenating arrays...")
    
    # Concatenate all valid pixels into two giant 1D arrays
    try:
        gt_all_valid = np.concatenate(all_gt_pixels)
        pred_all_valid = np.concatenate(all_pred_pixels)
    except ValueError:
        print("Error: No valid pixels found in any images. Cannot evaluate.", file=sys.stderr)
        return
        
    print(f"Evaluating on {gt_all_valid.size} valid pixels.")
    
    # Get metrics
    d1, d2, d3 = get_accuracy_metrics_v2(gt_all_valid, pred_all_valid)
    abs_rel, sq_rel, rmse, rmse_log, log10, silog = get_error_metrics_v2(gt_all_valid, pred_all_valid)

    # --- Print Results ---
    print("\n--- Error Metrics (Lower is Better) ---")
    print(f"  AbsRel : {abs_rel:.4f}")
    print(f"  SqRel  : {sq_rel:.4f}")
    print(f"  RMSE   : {rmse:.4f}")
    print(f"  RMSElog: {rmse_log:.4f}")
    print(f"  log10  : {log10:.4f}")
    print(f"  SILog  : {silog:.4f}")
    
    print("\n--- Accuracy Metrics (Higher is Better) ---")
    print(f"  d1 (1.25) : {d1:.4f}")
    print(f"  d2 (1.25^2): {d2:.4f}")
    print(f"  d3 (1.25^3): {d3:.4f}")

    print("\n--- Evaluation Complete ---")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate monocular depth estimation models.")
    
    # Add one required argument: --dir
    parser.add_argument(
        "--dir", 
        type=str, 
        required=True, 
        help="Path to the output directory (e.g., './cityscapes_output_dpt-beit-large-512')"
    )
    
    args = parser.parse_args()
    main(args)

