import os
import shutil
import glob
import tifffile
import numpy as np
import yaml
import argparse
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

#############################################
# Dataset Classes
#############################################
class HeatmapDataset3D(Dataset):
    def __init__(self, image_dir, heatmap_dir, patch_size=(96, 96, 192), transform=None):
        """
        Dataset for training. Loads 3D tiff images and their corresponding heatmaps.
        
        Args:
            image_dir (str): Directory path containing training images.
            heatmap_dir (str): Directory path containing corresponding heatmaps.
            patch_size (tuple): Desired patch size (D, H, W).
            transform (callable, optional): Data augmentation transforms.
        """
        self.image_dir = image_dir
        self.heatmap_dir = heatmap_dir
        self.image_files = sorted(glob.glob(os.path.join(image_dir, '*.tiff')))
        self.heatmap_files = sorted(glob.glob(os.path.join(heatmap_dir, '*.tiff')))
        print(f"-------------Found {len(self.image_files)} images and {len(self.heatmap_files)} heatmaps.")
        assert len(self.image_files) == len(self.heatmap_files), "Number of images and heatmaps do not match!"
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def random_crop(self, img, heatmap, crop_size):
        """
        Randomly crop a patch of size crop_size from a 3D volume.
        Assumes img and heatmap shape is (C, D, H, W).
        If any spatial dimension is smaller than the crop size, pad the volume with zeros.
        """
        C, D, H, W = img.shape
        pD, pH, pW = crop_size

        # Compute the required padding for each spatial dimension
        pad_d = max(pD - D, 0)
        pad_h = max(pH - H, 0)
        pad_w = max(pW - W, 0)

        # Split the padding equally (with the extra pixel on the right/bottom if necessary)
        pad_d_before = pad_d // 2
        pad_d_after = pad_d - pad_d_before
        pad_h_before = pad_h // 2
        pad_h_after = pad_h - pad_h_before
        pad_w_before = pad_w // 2
        pad_w_after = pad_w - pad_w_before

        # Define the pad width for each dimension: (channel, depth, height, width)
        pad_width = (
            (0, 0), 
            (pad_d_before, pad_d_after), 
            (pad_h_before, pad_h_after), 
            (pad_w_before, pad_w_after)
        )

        # Pad both the image and heatmap with zeros if necessary
        img = np.pad(img, pad_width, mode='constant', constant_values=0)
        heatmap = np.pad(heatmap, pad_width, mode='constant', constant_values=0)

        # Get new shape after padding
        _, new_D, new_H, new_W = img.shape

        # Randomly select the start indices for cropping if the dimension is larger than crop size.
        d_start = np.random.randint(0, new_D - pD + 1) if new_D > pD else 0
        h_start = np.random.randint(0, new_H - pH + 1) if new_H > pH else 0
        w_start = np.random.randint(0, new_W - pW + 1) if new_W > pW else 0

        # Crop the patch from both image and heatmap
        cropped_img = img[:, d_start:d_start+pD, h_start:h_start+pH, w_start:w_start+pW]
        cropped_heatmap = heatmap[:, d_start:d_start+pD, h_start:h_start+pH, w_start:w_start+pW]

        return cropped_img, cropped_heatmap

    def __getitem__(self, idx):
        # Read image and corresponding heatmap.
        img = tifffile.imread(self.image_files[idx])
        heatmap = tifffile.imread(self.heatmap_files[idx])
        
        # Add channel dimension if missing.
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)
        if heatmap.ndim == 3:
            heatmap = np.expand_dims(heatmap, axis=0)
        # Transpose image to (C, D, H, W) if necessary.
        if img.shape[1] == 3:
            img = np.transpose(img, (1, 0, 2, 3))
        # Convert to float32.
        img = img.astype(np.float32)
        # Clip image intensities.
        img = np.clip(img, 0, 60000)
        # Z-score normalization.
        img = (img - np.mean(img)) / np.std(img)
        heatmap = heatmap.astype(np.float32)
        
        # Apply random cropping.
        img, heatmap = self.random_crop(img, heatmap, self.patch_size)
        
        # Apply optional transforms.
        if self.transform:
            img, heatmap = self.transform(img, heatmap)
        
        # Convert numpy arrays to torch tensors.
        img_tensor = torch.from_numpy(img)
        heatmap_tensor = torch.from_numpy(heatmap)
        
        return img_tensor, heatmap_tensor


#############################################
# Training Function
#############################################
def train_model(model, dataloader, criterion, optimizer, device, num_epochs, model_dir):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, heatmaps in dataloader:
            images = images.to(device)       # Shape: (B, C, D, H, W)
            heatmaps = heatmaps.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, heatmaps)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        if epoch % 50 == 0:
            checkpoint_path = os.path.join(model_dir, f'nnunet3d_epoch_{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Saved checkpoint: {checkpoint_path}")
        logging.info(f'Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f}')
    return model

def build_nnUNet_model():
    
    model = get_network_from_plans(
        arch_class_name="dynamic_network_architectures.architectures.unet.PlainConvUNet",
        arch_kwargs= {
                    "n_stages": 6,
                    "features_per_stage": [
                        32,
                        64,
                        128,
                        256,
                        320,
                        320
                    ],
                    "conv_op": "torch.nn.modules.conv.Conv3d",
                    "kernel_sizes": [
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ]
                    ],
                    "strides": [
                        [
                            1,
                            1,
                            1
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            1,
                            2,
                            2
                        ]
                    ],
                    "n_conv_per_stage": [
                        2,
                        2,
                        2,
                        2,
                        2,
                        2
                    ],
                    "n_conv_per_stage_decoder": [
                        2,
                        2,
                        2,
                        2,
                        2
                    ],
                    "conv_bias": True,
                    "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                    "norm_op_kwargs": {
                        "eps": 1e-05,
                        "affine": True
                    },
                    "dropout_op": None,
                    "dropout_op_kwargs": None,
                    "nonlin": "torch.nn.LeakyReLU",
                    "nonlin_kwargs": {
                        "inplace": True
                    }
                },
        arch_kwargs_req_import=["conv_op", "norm_op", "dropout_op", "nonlin"],
        input_channels=3,
        output_channels=1,
        allow_init=True,
        deep_supervision=False,
    )
    return model
#############################################
# Main Function (with Optional Fold Selection)
#############################################
def main():
    # Parse YAML configuration and additional fold parameter.
    parser = argparse.ArgumentParser(description="3D U-Net 5-Fold Cross Validation Training and Testing with YAML config")
    parser.add_argument("-c", "--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument('-f',"--fold", type=int, default=None, 
                        help="Specify which fold (0-4) to train. If omitted, all folds are trained sequentially.")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create experiment folder with timestamp.
    exp_name = config['experiment'].get('name', 'default_experiment')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_exp_dir = os.path.join(config['experiment'].get('save_dir', './experiments'),
                                f"{exp_name}_{timestamp}")
    os.makedirs(base_exp_dir, exist_ok=True)
    
    # Save a copy of the config file to the experiment directory.
    config_save_path = os.path.join(base_exp_dir, "config.yaml")
    shutil.copy(args.config, config_save_path)
    
    # Create subdirectories for models, logs, and predictions.
    model_dir = os.path.join(base_exp_dir, 'models')
    log_dir = os.path.join(base_exp_dir, 'logs')
    pred_dir = os.path.join(base_exp_dir, 'predictions')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    
    # Set up logging.
    log_file = os.path.join(log_dir, 'experiment.log')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    logging.info("Experiment started.")
    logging.info(f"Configuration: {config}")
    
    # Device configuration.
    device_name = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_name)
    logging.info(f"Using device: {device}")
    
    # Data parameters.
    patch_size = tuple(config['patch']['size'])
    stride = tuple(config['patch'].get('stride', [s // 2 for s in patch_size]))
    batch_size = config.get('batch_size', 1)
    num_epochs = config.get('num_epochs', 1000)
    learning_rate = config.get('learning_rate', 1e-4)
    input_channels = config.get('input_channels', 1)
    
    # New: root directory containing the 5 folds (each fold has subdirectories "images" and "heatmaps").
    fold_data_dir = config['data']['fold_data_dir']
    
    # Determine which folds to process.
    if args.fold is not None:
        if args.fold < 0 or args.fold > 4:
            raise ValueError("Fold must be an integer between 0 and 4.")
        folds_to_train = [args.fold]
    else:
        folds_to_train = list(range(5))
    
    # Loop over the selected folds.
    for fold in folds_to_train:
        logging.info(f"========== Starting training for fold {fold} ==========")
        
        # Build training dataset by concatenating all folds except the current one.
        train_datasets = []
        for i in range(5):
            if i == fold:
                continue  # Skip the current fold (used as test set)
            curr_fold_dir = os.path.join(fold_data_dir, f"fold_{i}")
            curr_train_image_dir = os.path.join(curr_fold_dir, "images")
            curr_train_heatmap_dir = os.path.join(curr_fold_dir, "heatmaps")
            ds = HeatmapDataset3D(curr_train_image_dir, curr_train_heatmap_dir, patch_size=patch_size)
            train_datasets.append(ds)
        if len(train_datasets) == 1:
            train_dataset = train_datasets[0]
        else:
            train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
                
        # Create a new model for the current fold.
        model = build_nnUNet_model().to(device)
        logging.info(f"Model created for fold {fold}.")
        
        # Define loss function and optimizer.
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Create a directory to save model checkpoints for this fold.
        fold_model_dir = os.path.join(model_dir, f"fold_{fold}")
        os.makedirs(fold_model_dir, exist_ok=True)
        
        # Train the model for the current fold.
        logging.info(f"Training model for fold {fold}...")
        model = train_model(model, train_loader, criterion, optimizer, device, num_epochs, fold_model_dir)
        
        # Save final model for the current fold.
        final_model_path = os.path.join(fold_model_dir, 'nnunet3d_final_model.pth')
        torch.save(model.state_dict(), final_model_path)
        logging.info(f"Final model saved for fold {fold}: {final_model_path}")
        
    logging.info("5-fold cross validation completed.")

if __name__ == '__main__':
    main()
