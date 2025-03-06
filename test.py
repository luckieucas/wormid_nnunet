#!/usr/bin/env python
import os
import glob
import argparse
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

#############################################
# Sliding-window Inference Function
#############################################
def sliding_window_inference(model, image, patch_size, stride, device):
    """
    Perform sliding-window inference on a 3D volume.
    
    Args:
        model: The segmentation model.
        image: Input image tensor of shape (C, D, H, W) on CPU.
        patch_size (tuple): The patch size (pD, pH, pW).
        stride (tuple): The stride (sD, sH, sW) for sliding windows.
        device: Torch device.
    
    Returns:
        output_tensor: Full-volume prediction tensor of shape (n_classes, D, H, W).
    """
    model.eval()
    # Record original image dimensions
    original_shape = image.shape  # (C, orig_D, orig_H, orig_W)
    C, orig_D, orig_H, orig_W = original_shape
    pD, pH, pW = patch_size

    # Pad the image if any dimension is smaller than the patch size
    pad_d = max(0, pD - orig_D)
    pad_h = max(0, pH - orig_H)
    pad_w = max(0, pW - orig_W)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        # Padding format: (w_left, w_right, h_left, h_right, d_left, d_right)
        pad = (pad_w // 2, pad_w - pad_w // 2,
               pad_h // 2, pad_h - pad_h // 2,
               pad_d // 2, pad_d - pad_d // 2)
        image = F.pad(image, pad, mode='constant', value=0)
    else:
        pad = (0, 0, 0, 0, 0, 0)
    
    # Update dimensions after padding
    C, D, H, W = image.shape
    output_tensor = torch.zeros((1, D, H, W), dtype=torch.float32)
    count_tensor = torch.zeros((1, D, H, W), dtype=torch.float32)
    
    sD, sH, sW = stride

    # Calculate sliding window starting indices for each dimension
    d_starts = list(range(0, D - pD + 1, sD)) if D >= pD else [0]
    if D >= pD and (len(d_starts) == 0 or d_starts[-1] != D - pD):
        d_starts.append(D - pD)
        
    h_starts = list(range(0, H - pH + 1, sH)) if H >= pH else [0]
    if H >= pH and (len(h_starts) == 0 or h_starts[-1] != H - pH):
        h_starts.append(H - pH)
        
    w_starts = list(range(0, W - pW + 1, sW)) if W >= pW else [0]
    if W >= pW and (len(w_starts) == 0 or w_starts[-1] != W - pW):
        w_starts.append(W - pW)
    
    # Inference using sliding windows
    with torch.no_grad():
        for d in d_starts:
            for h in h_starts:
                for w in w_starts:
                    # Extract patch from image
                    patch = image[:, d:d+pD, h:h+pH, w:w+pW]  # shape: (C, pD, pH, pW)
                    patch = patch.unsqueeze(0).to(device)      # shape: (1, C, pD, pH, pW)
                    pred = model(patch)
                    # Apply Sigmoid to obtain probabilities
                    pred = torch.sigmoid(pred)  # shape: (1, n_classes, pD, pH, pW)
                    pred = pred.squeeze(0)      # shape: (n_classes, pD, pH, pW)
                    output_tensor[:, d:d+pD, h:h+pH, w:w+pW] += pred.cpu()
                    count_tensor[:, d:d+pD, h:h+pH, w:w+pW] += 1
                    
    # Average overlapping patches
    output_tensor /= count_tensor

    # Crop the output back to the original image size if padding was applied
    if any(pad):
        # pad order: (w_left, w_right, h_left, h_right, d_left, d_right)
        w_left, w_right, h_left, h_right, d_left, d_right = pad
        output_tensor = output_tensor[:, 
                                      d_left:d_left+orig_D, 
                                      h_left:h_left+orig_H, 
                                      w_left:w_left+orig_W]
    
    return output_tensor

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
# Main prediction function using argparse
#############################################
def main(args):
    # Parse patch size and stride from strings (e.g., "96,96,192")
    patch_size = tuple(map(int, args.patch_size.split(',')))
    stride = tuple(map(int, args.stride.split(',')))

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')
    print(f"Using device: {device}")

    # Create model and load saved weights.
    model = build_nnUNet_model().to(device)
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Loaded model weights.")

    # Determine if input is a file or directory
    if os.path.isdir(args.input):
        file_list = sorted(glob.glob(os.path.join(args.input, '*.tiff')))
    elif os.path.isfile(args.input):
        file_list = [args.input]
    else:
        raise ValueError("Input path is neither a file nor a directory.")

    os.makedirs(args.output_dir, exist_ok=True)

    # Loop over each file and perform prediction.
    for file_path in file_list:
        print(f"Processing: {file_path}")
        # Read the image using tifffile.
        img = tifffile.imread(file_path)
        # If image is 3D (D, H, W), add a channel dimension.
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)
        # Convert image to float32.
        img = img.astype(np.float32)
        if img.shape[1] == 3:
            img = np.transpose(img, (1, 0, 2, 3))
        # Convert to torch tensor.
        img_tensor = torch.from_numpy(img)  # shape: (C, D, H, W)
        # clip img.
        img_tensor = torch.clamp(img_tensor, 0, 60000)
        # z-score
        img_tensor = (img_tensor - img_tensor.mean()) / img_tensor.std()

        # Check if sliding-window inference is needed.
        C, D, H, W = img_tensor.shape
        if (D > patch_size[0]) or (H > patch_size[1]) or (W > patch_size[2]):
            print("Using sliding-window inference.")
            # Make sure the tensor is on CPU for sliding-window function.
            img_tensor_cpu = img_tensor.to('cpu')
            pred = sliding_window_inference(model, img_tensor_cpu, patch_size, stride, device)
        else:
            # Direct inference
            img_tensor = img_tensor.unsqueeze(0).to(device)  # (1, C, D, H, W)
            with torch.no_grad():
                pred = model(img_tensor).squeeze(0)  # (n_classes, D, H, W)

        
        # Convert prediction to numpy array.
        pred_np = pred.cpu().numpy()
        print(f"Prediction max: {pred_np.max()}, min: {pred_np.min()}")
        # Normalize prediction between 0 and 1.
        pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min())
        pred_np = pred_np * 255
        # If prediction has 1 channel, remove channel dimension.
        if pred_np.shape[0] == 1:
            pred_np = pred_np[0]
        
        # Create output filename.
        base_name = os.path.basename(file_path)
        out_path = os.path.join(args.output_dir, base_name)
        # Save prediction as tif.
        tifffile.imwrite(out_path, pred_np.astype(np.uint8))
        print(f"Saved prediction to: {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="3D TIF Prediction with UNet3DSmall")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input file or directory containing tif files.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the prediction results.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved model state dictionary (e.g., unet3d_small_model.pth).")
    parser.add_argument("--patch_size", type=str, default="32,96,64",
                        help="Patch size (D,H,W) for inference (default: 96,96,192).")
    parser.add_argument("--stride", type=str, default="16,48,32",
                        help="Stride (D,H,W) for sliding window inference (default: 48,48,96).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for inference ('cuda' or 'cpu').")
    args = parser.parse_args()
    main(args)
