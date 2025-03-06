#!/usr/bin/env python3
import os
import glob
import numpy as np
import tifffile
from skimage.feature import peak_local_max
import argparse
from scipy.spatial.distance import cdist



def evaluate(predicted_cell, gt_cell, threshold=15):
    """
    predicted_cell: n x 3
    gt_cell: m x 3

    If there is no ground-truth cell within a valid distance, the cell prediction is counted as an FP
    If there are one or more ground-truth cells within a valid distance, the cell prediction is counted as a TP.
    The remaining ground-truth cells that are not matched with any cell prediction are counted as FN.

    return precision, recall, f1 score
    """
    dist = cdist(predicted_cell, gt_cell, metric='euclidean')
    n_pred, n_gt = dist.shape
    assert(n_pred != 0 and n_gt != 0)
    bool_mask = (dist <= threshold)
    tp, fp = 0, 0
    
    for i in range(len(predicted_cell)):
        neighbors = bool_mask[i].nonzero()[0]

        if len(neighbors) == 0:
            fp += 1
        else:
            gt_idx = min(neighbors, key=lambda j: dist[i, j])
            tp += 1
            bool_mask[:, gt_idx] = False
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (n_gt + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    return precision, recall, f1

def process_heatmap_file(heatmap_path, output_folder, gt_folder,min_distance=5, threshold_abs=0.1):
    """
    Process a single 3D heatmap file:
      1. Load the heatmap (3D TIFF file, shape (Z, Y, X)).
      2. Apply a threshold to remove background noise.
      3. Detect local maxima (peaks) using peak_local_max.
      4. Save the detected coordinates as a .npy file.

    Args:
        heatmap_path (str): Path to the heatmap TIFF file.
        output_folder (str): Directory where the output .npy file will be saved.
        min_distance (int): Minimum pixel distance between detected peaks.
        threshold_abs (float): Absolute threshold for peak detection.
    """
    # Load the heatmap (assumed to be 3D)
    heatmap = tifffile.imread(heatmap_path).astype(np.float32)  
    heatmap /= 255.0 
    print(f"Processing: {heatmap_path}, Shape: {heatmap.shape}, Min: {heatmap.min()}, Max: {heatmap.max()}\n")

    # Apply threshold to remove background noise
    heatmap[heatmap < threshold_abs] = 0

    # Detect local maxima (peaks) in the 3D heatmap
    # The returned coordinates array shape is (N, 3), where each row is (z, y, x)
    coords = peak_local_max(
        heatmap,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        exclude_border=False  # Set to False if peaks near the border should be detected
    )

    # transopose coordinates from (z, y, x) to (x, y, z)
    coords = coords[:, [1, 2, 0]] 

    # Sort the coordinates by z, y, and x
    coords = coords[np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))]   
    
    # Generate the output file name, e.g., "xxx.tiff" -> "xxx_peaks.npy"
    base_name = os.path.splitext(os.path.basename(heatmap_path))[0]
    case_group = base_name.split("_")[0]
    base_name = base_name.replace(case_group + "_", "")
    save_folder = os.path.join(output_folder, case_group)
    gt_folder = os.path.join(gt_folder, case_group)
    os.makedirs(save_folder, exist_ok=True)
    output_file = os.path.join(save_folder, base_name + '_points.npy')
    # Save the detected coordinates
    np.save(output_file, coords)
    print(f"Saved detected peak coordinates to: {output_file}\n")
    
    precision, recall, f1 = 0, 0, 0
    if gt_folder is not None:
        gt_file = os.path.join(gt_folder, base_name + '_points.npy')
        
        gt_point = np.load(gt_file)

        # evaluate
        precision, recall, f1 = evaluate(coords, gt_point)
        print(f"Precision: {precision}, Recall: {recall}, F1: {f1}\n")

    return precision, recall, f1
def process_folder(input_folder, output_folder, gt_folder, min_distance=5, threshold_abs=0.1):
    """
    Process all 3D heatmap files in a given folder.
    
    Args:
        input_folder (str): Path to the folder containing heatmap TIFF files.
        output_folder (str): Directory where detected peak coordinates will be saved.
        min_distance (int): Minimum pixel distance between peaks.
        threshold_abs (float): Absolute threshold for peak detection.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Find all TIFF files in the folder
    file_list = glob.glob(os.path.join(input_folder, '*.tiff'))
    file_list += glob.glob(os.path.join(input_folder, '*.tif'))
    
    if not file_list:
        print(f"No TIFF files found in {input_folder}.")
        return
    
    print(f"Found {len(file_list)} heatmap files. Processing...\n")
    
    # Process each heatmap file
    precision_list = []
    recall_list = []
    f1_list = []
    for heatmap_path in file_list:
        precision, recall, f1 = process_heatmap_file(heatmap_path, output_folder, gt_folder, min_distance, threshold_abs)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        
    print(f"Processed {len(file_list)} heatmap files. Precision: {np.mean(precision_list)}, Recall: {np.mean(recall_list)}, F1: {np.mean(f1_list)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process 3D heatmap files to extract peak coordinates and save them as .npy files.")
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Path to the folder containing predicted heatmap files.")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to the folder where the output .npy files will be saved.")
    parser.add_argument("-gt", "--gt_folder", type=str, default=None,)
    parser.add_argument("--min_distance", type=int, default=4,
                        help="Minimum pixel distance between detected peaks (default: 5).")
    parser.add_argument("--threshold_abs", type=float, default=0.2,
                        help="Absolute threshold for peak detection (default: 0.1).")
    args = parser.parse_args()
    
    process_folder(args.input_folder, args.output_folder, args.gt_folder,args.min_distance, args.threshold_abs)
