import argparse
import os
import numpy as np
import h5py
from scipy.ndimage import center_of_mass, distance_transform_edt

def compute_inter_instance_distance(mask, current_label):
    """
    Compute the distance from the current instance to other instances using edge distance.
    """
    temp_mask = np.copy(mask)
    temp_mask[temp_mask == current_label] = 0
    distance = distance_transform_edt(temp_mask != 0)
    return distance

def generate_instance_gaussian_heatmaps(mask, sigma=3, edge_decay_factor=0.5):
    """
    Generate a Gaussian heatmap for each instance in the mask with edge decay,
    and sum all instance heatmaps into a final heatmap.
    """
    labels = np.unique(mask)
    labels = labels[labels != 0]
    
    heatmaps = np.zeros_like(mask, dtype=np.float32)
    
    for label in labels:
        instance_mask = (mask == label).astype(np.float32)
        center = center_of_mass(instance_mask)
        
        # Create a coordinate grid with the same shape as the mask (3, Z, Y, X)
        grid = np.indices(mask.shape).astype(np.float32)
        grid_z, grid_y, grid_x = grid[0], grid[1], grid[2]
        
        # Compute the Gaussian distribution centered at the instance center
        gaussian = np.exp(-((grid_z - center[0])**2 
                            + (grid_y - center[1])**2 
                            + (grid_x - center[2])**2) / (2 * sigma**2))
        gaussian /= np.max(gaussian)
        
        # Compute the edge suppression factor
        edge_distance = compute_inter_instance_distance(mask, label)
        edge_suppression = np.exp(-edge_decay_factor * edge_distance)
        
        gaussian *= edge_suppression
        heatmaps += gaussian
    
    return heatmaps

def process_file(input_file, output_file, sigma=3, edge_decay_factor=0.5):
    """
    Process a single h5 file:
    - Read the "main" dataset (image) and the "1" dataset (3D coordinates).
    - Create a mask from the coordinates, assigning a unique label to each coordinate point.
    - Generate a Gaussian heatmap using the mask.
    - Save the heatmap to a new h5 file.
    """
    with h5py.File(input_file, 'r') as f:
        # Read image data (assumed to be a 3D image)
        image = f['main'][...]
        # Read 3D coordinates; assumed shape is (N, 3)
        coords = f['1'][...]
    
    # Create a mask with the same shape as the image, initialized to zeros
    mask = np.zeros((image.shape[0],image.shape[2],image.shape[3]), dtype=np.uint16)
    print(f"mask shape: {mask.shape}, coords shape: {coords.shape}")
    # Assign a unique label for each coordinate point
    for idx, coord in enumerate(coords):
        # Round the coordinate to integer indices, assuming order (z, y, x)
        y, x, z = np.round(coord).astype(int)
        # Check if the coordinate is within the image bounds
        if (0 <= z < mask.shape[0]) and (0 <= y < mask.shape[1]) and (0 <= x < mask.shape[2]):
            mask[z, y, x] = idx + 1  # Labels start from 1
    print(f"coord:{coord}")
    # Generate the Gaussian heatmap using the coordinates as centers
    heatmap = generate_instance_gaussian_heatmaps(mask, sigma=sigma, edge_decay_factor=edge_decay_factor)
    
    print(f"max heatmap: {heatmap.max()}, min heatmap: {heatmap.min()}")
    # Save the heatmap to a new h5 file in the output folder
    with h5py.File(output_file, 'w') as f_out:
        f_out.create_dataset('main', data=heatmap)
    
    print(f"Processed {input_file} and saved heatmap to {output_file}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate heatmaps from h5 files")
    parser.add_argument("--input_folder", type=str, help="Folder containing input h5 files")
    parser.add_argument("--output_folder", type=str, help="Folder where heatmap h5 files will be saved")
    args = parser.parse_args()
    
    input_folder = args.input_folder
    output_folder = args.output_folder
    
    # Check if input folder exists
    if not os.path.isdir(input_folder):
        print(f"Input folder {input_folder} does not exist.")
        return
    
    # Create output folder if it doesn't exist
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    # Process each h5 file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('_im.h5'):
            print(f"Processing {filename}")
            input_file = os.path.join(input_folder, filename)
            # Construct output file name: original file name without extension + '_heatmap.h5'
            base_name = os.path.splitext(filename)[0]
            output_file = os.path.join(output_folder, base_name + '_heatmap.h5')
            process_file(input_file, output_file)

if __name__ == '__main__':
    main()
