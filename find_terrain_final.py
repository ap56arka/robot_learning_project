from PIL import Image
from typing import Dict, List, Optional
from tqdm.notebook import tqdm
from eai2025_lab2_perception.lab_utils.model_loaders import load_sam_model
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import rerun as rr
import sys
import os
sys.path.append(os.path.abspath("MoGe"))
device = "cuda" if torch.cuda.is_available() else "cpu"
from moge.model.v2 import MoGeModel # Let's try MoGe-2
import subprocess
import time
from scipy.stats import mode
import json

#GLOBAL OPTIONS
DO_MOGE = True
#base_dir = "../masks/output_rpl_staircase"
base_dir = "../masks/output_rpl_staircase_better_masks"
#base_dir = "../masks/output_dronelabarka_2"
files_per_folder = 50 
output_json = "traversability_scores_dronelab.json"
save_folder = "../output_visualizations_dronelab"

def normalize_slopes_inverted(slopes):
    """
    Normalize a list of positive slope values to the range [0, 1],
    where smaller slopes map to values closer to 1 (more traversable)
    and larger slopes map to values closer to 0 (less traversable).

    Parameters:
        slopes (list or np.ndarray): List of positive slope values.

    Returns:
        list: Inverted normalized slope values between 0 and 1.
    """
    import numpy as np

    slopes = np.array(slopes, dtype=float)
    if np.any(slopes < 0):
        raise ValueError("All slope values must be positive.")

    min_val = np.min(slopes)
    max_val = np.max(slopes)

    if max_val == min_val:
        return [1.0] * len(slopes)  # All equal => fully traversable

    normalized = (slopes - min_val) / (max_val - min_val)
    #print(f"Max - Min slope: {max_val - min_val}")
    #print(f"Max slope: {max_val}, Min slope: {min_val}")
    #print(f"Slopes: {slopes}")
    inverted = normalized  # Invert: small slopes → high value
    inverted_rounded = [round(val, 3) for val in inverted]
    return inverted_rounded

def visualize_sam_grid_proposals(image: np.ndarray, proposals: list):
    """
    Visualize all the final SAM proposals (after IoU filtering and confidence thresholding).
    Each proposal should have keys: ['mask', 'area', 'point', 'confidence'].
    """
    if len(proposals) == 0:
        print("⚠️ No proposals to visualize.")
        return

    # Convert to RGB if grayscale
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')

    # Sort by area (larger first)
    proposals_sorted = sorted(proposals, key=lambda x: x['area'], reverse=True)

    for i, proposal in enumerate(proposals):
        mask = proposal['mask']
        conf = proposal.get('confidence', 0.0)
        point = proposal.get('point', [0, 0])

        # Random transparent color
        color_mask = np.random.random(3)
        color = np.concatenate([color_mask, [0.5]])  # RGBA

        # Create overlay
        overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
        overlay[mask] = color

        plt.imshow(overlay)

        # Optionally mark the grid point
        plt.scatter(point[0], point[1], c=[color_mask], marker='x', s=40)
        plt.text(point[0] + 5, point[1] + 5, f"{i}", color='white', fontsize=8)

    plt.title(f"Grid-based SAM Proposals ({len(proposals)} masks)")
    plt.show()
'''
def visualize_sam_grid_proposals_terrain_save(frame_num, image: np.ndarray, proposals: list, labels, save_path: Optional[str] = None):
    """
    Visualize all the final SAM proposals (after IoU filtering and confidence thresholding)
    and save the visualization to a file if save_path is provided.
    The figure will NOT be displayed.
    """
    import os
    if len(proposals) == 0:
        print("⚠️ No proposals to visualize.")
        return

    # Convert to RGB if grayscale
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')

    # Sort by area (larger first)
    proposals_sorted = sorted(proposals, key=lambda x: x['area'], reverse=True)

    for i, proposal in enumerate(proposals):
        mask = proposal['mask']
        point = proposal.get('point', [0, 0])

        # Random transparent color
        color_mask = np.random.random(3)
        color = np.concatenate([color_mask, [0.5]])  # RGBA

        # Create overlay
        overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
        overlay[mask] = color

        plt.imshow(overlay)

        # Optionally mark the grid point
        plt.scatter(point[0], point[1], c=[color_mask], marker='x', s=40)
        plt.text(point[0] + 5, point[1] + 5, f"{labels[i]}", color='white', fontsize=8)

    plt.title(f"Frame {frame_num}: SAM Proposals ({len(proposals)} masks) with terrain scores")

    # Save figure if save_path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    # Close figure to free memory
    plt.close()'''


def visualize_sam_grid_proposals_terrain_save(frame_num, image: np.ndarray, proposals: list, labels, save_path: Optional[str] = None):
    """
    Visualize all the final SAM proposals (after IoU filtering and confidence thresholding)
    and save the visualization to a file if save_path is provided.
    The figure will NOT be displayed.

    Colors:
        - flat: green
        - non-flat: red
        - unknown: black
    """

    if len(proposals) == 0:
        print(f"⚠️ No proposals to visualize for frame {frame_num}.")
        return

    # Convert to RGB if grayscale
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')

    # Sort by area (larger first)
    proposals_sorted = sorted(proposals, key=lambda x: x['area'], reverse=True)

    for i, proposal in enumerate(proposals):
        mask = proposal['mask']
        point = proposal.get('point', [0, 0])
        label = labels[i].lower() if i < len(labels) else "unknown"

        # Assign color based on label
        if label == "flat":
            color_mask = np.array([0.0, 1.0, 0.0])  # Green
        elif label == "non-flat":
            color_mask = np.array([1.0, 0.0, 0.0])  # Red
        else:
            color_mask = np.array([0.0, 0.0, 0.0])  # Black (unknown)

        color = np.concatenate([color_mask, [0.4]])  # RGBA with transparency

        # Create overlay
        overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
        overlay[mask] = color
        plt.imshow(overlay)

        # Mark the point and label
        #plt.scatter(point[0], point[1], c=[color_mask], marker='x', s=40)
        #plt.text(point[0] + 5, point[1] + 5, f"{label}", color='white', fontsize=8)

    plt.title(f"Frame {frame_num}: SAM Proposals ({len(proposals)} masks) with terrain labels")

    # Save figure if save_path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ Saved visualization for frame {frame_num} → {save_path}")

    # Close figure to free memory
    plt.close()



def visualize_sam_grid_proposals_terrain(image: np.ndarray, proposals: list, labels):
    """
    Visualize all the final SAM proposals (after IoU filtering and confidence thresholding).
    Each proposal should have keys: ['mask', 'area', 'point', 'confidence'].
    """
    if len(proposals) == 0:
        print("⚠️ No proposals to visualize.")
        return

    # Convert to RGB if grayscale
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')

    # Sort by area (larger first)
    proposals_sorted = sorted(proposals, key=lambda x: x['area'], reverse=True)

    for i, proposal in enumerate(proposals):
        mask = proposal['mask']
        conf = proposal.get('confidence', 0.0)
        point = proposal.get('point', [0, 0])

        # Random transparent color
        color_mask = np.random.random(3)
        color = np.concatenate([color_mask, [0.5]])  # RGBA

        # Create overlay
        overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
        overlay[mask] = color

        plt.imshow(overlay)

        # Optionally mark the grid point
        plt.scatter(point[0], point[1], c=[color_mask], marker='x', s=40)
        plt.text(point[0] + 5, point[1] + 5, f"{labels[i]}", color='white', fontsize=8)

    plt.title(f"Grid-based SAM Proposals ({len(proposals)} masks) with terrain scores")
    plt.show()

#image1 = Image.open("../images/"+image_name).convert("RGB")
#plt.imshow(image1)


all_frames_data = {}

for frame_num in range(349,350):
    proposals = []
    chunk_idx = frame_num // files_per_folder
    frame_idx = frame_num % files_per_folder

    # build paths
    folder_name = f"output_rpl_staircase_chunk_{chunk_idx}_npz"#For RPl staircase
    #folder_name = f"output_dronelabarka_chunk_{chunk_idx}_npz"
    #npz_name = f"video_segments_frame_{frame_idx:04d}.npz"
    npz_name = f"output_rpl_staircase_frame_{frame_idx:04d}_masks.npz"#RPL staircase#output_rpl_staircase_frame_0000_masks.npz
    #npz_name = f"output_dronelabarka_frame_{frame_idx:04d}_masks.npz"
    npz_path = os.path.join(base_dir, folder_name, npz_name)

    # load npz
    if os.path.exists(npz_path):
        mask_pre = np.load(npz_path)
        # example: print contents
        #print(f"Loaded frame {frame_num} from {npz_path}")
        #print("Arrays inside:", proposals.files)

        # Example usage (replace with your own logic)
        # arr = data['some_array_name']

        
    else:
        print(f"Missing file for frame {frame_num}: {npz_path}")
    image_name = f"../output_frames/rpl_original_frames/frame_{frame_num:05d}.jpg"  # Image file to uses, for RPL staircase
    #image_name = f"../output_frames_dronelab/drone_original_frames/frame_{frame_num:05d}.jpg"

    input_image_bgr = cv2.imread(image_name)
    for key in mask_pre.files:
        mask_pre_j = mask_pre[key]
        mask_pre_j = mask_pre_j
        #Upsampling the masks
        mask_resized_pre_j = cv2.resize(mask_pre_j.astype(np.uint8), 
                          (input_image_bgr.shape[1], input_image_bgr.shape[0]),  # width, height
                          interpolation=cv2.INTER_NEAREST)  # preserve mask values

        # Optional: convert back to bool
        mask_resized_pre_j = mask_resized_pre_j.astype(bool)
        area = np.sum(mask_resized_pre_j)
        ys, xs = np.where(mask_resized_pre_j)  # coordinates of True pixels
        if len(xs) > 0:
            rand_idx = np.random.randint(0, len(xs))
            point = [int(xs[rand_idx]), int(ys[rand_idx])]  # x, y
        else:
            point = [0, 0]  # fallback if mask is empty

        # For simplicity, we won't have point and confidence here
        proposal = {
            'mask': mask_resized_pre_j,
            'area': area,
            'point': point,
            'confidence': 0.0  # Placeholder
        }
        proposals.append(proposal)
    
    if frame_num == 0:
        print(f"Mask shape for frame {frame_num}: ", mask_pre['0'].shape)
        print(f"Image shape for frame {frame_num}: ", cv2.imread(image_name).shape)
        visualize_sam_grid_proposals(input_image_bgr, proposals)

    if DO_MOGE:
        # -----------------------------
        # Load model
        # -----------------------------
        print("---------------------------------------------------")
        print(f"Processing frame {frame_num} for Depth...")
        print("---------------------------------------------------")
        model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)                             

        # -----------------------------
        # Read and preprocess image (lightweight)
        # -----------------------------
        

        # Resize for faster inference
        scale_factor = 0.25  # 1/4 resolution
        h, w = input_image_bgr.shape[:2]
        #print(f"Original image size: {w}x{h}, Resized to: {int(w*scale_factor)}x{int(h*scale_factor)}")
        H_small, W_small = int(input_image_bgr.shape[0]*scale_factor), int(input_image_bgr.shape[1]*scale_factor)
        input_image_small = cv2.resize(input_image_bgr, (W_small, H_small))
        input_image_small = cv2.resize(input_image_bgr, (int(w*scale_factor), int(h*scale_factor)))
        input_image_rgb = cv2.cvtColor(input_image_small, cv2.COLOR_BGR2RGB)
        input_tensor = torch.tensor(input_image_rgb / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

        # -----------------------------
        # Inference
        # -----------------------------
        output = model.infer(input_tensor)

        points = output["points"]      # (H, W, 3)
        normals = output["normal"]     # (H, W, 3)
        mask = output["mask"]
        depth = output["depth"]        # (H, W)
        show_depth = False
        if show_depth:
            plt.figure(figsize=(10,5))
            plt.imshow(depth.cpu().numpy(), cmap='plasma')
            plt.colorbar(label='Depth (m)')
            plt.title('Estimated Depth Map')
            plt.axis('off')
            plt.show()


        # -----------------------------
        # Extract valid points and downsample early
        # -----------------------------
        mask_cpu = mask.cpu().numpy().flatten()
        valid_idx = np.where(mask_cpu)[0]
        print(f"Total points: {points.shape[0]*points.shape[1]}, Valid points: {len(valid_idx)}")

        # Sample a smaller subset for visualization
        num_samples = min(len(valid_idx), len(valid_idx))#Take all for now
        sample_idx = np.random.choice(valid_idx, size=num_samples, replace=False)

        # Flatten arrays and select samples
        points_valid = points.cpu().numpy().reshape(-1,3)[sample_idx]
        normals_valid = normals.cpu().numpy().reshape(-1,3)[sample_idx]
        colors_valid = input_image_rgb.reshape(-1,3)[sample_idx] / 255.0
        all_points = points.cpu().numpy().reshape(-1,3)

        # Flip Y axis for OpenCV → Cartesian conversion
        points_valid[:,1] = points_valid[:,1]#
        all_points[:, 1] = all_points[:, 1]#
        normals_valid[:,1] = normals_valid[:,1]#

        rr.init("moge_pointcloud", spawn=True)
        #rr.notebook_show(width=1200, height=800)

        # Log the 3D points (with colors)
        rr.log("points", rr.Points3D(all_points, colors=colors_valid))

        # -----------------------------
        labels = []
        slopes = []
        for i, proposal in enumerate(proposals):
            mask_ = proposals[i]['mask']
            mask_downscaled_ = cv2.resize(
                    mask_.astype(np.uint8),
                    (W_small, H_small),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            # 1️⃣ Extract the pixel coordinates and depth values where mask is True
            ys, xs = np.where(mask_downscaled_)          # pixel coordinates (row, col)
            zs = depth.cpu().numpy()[ys, xs]                  # depth values at those pixels
            
            if len(xs) == 0:
                print(f"Proposal {i}: No valid pixels in mask.")
                labels.append("unknown")
                slopes.append(0)
                continue
            # Stack into N x 3 points: (X, Y, Z)
            points_ = np.column_stack((xs, ys, zs))
            #points = np.column_stack((xs.cpu().numpy(), ys.cpu().numpy(), zs.cpu().numpy()))
            # Optional: remove any points where depth is NaN
            points_ = points_[~np.isnan(points_).any(axis=1)]
            print(f"Proposal {i}: Extracted {points_.shape[0]} 3D points from mask.")

            max_vals = np.max(points_, axis=0) + 1e-8  # avoid division by zero
            normalized_points = points_ / max_vals
            points_ = normalized_points * 100  # scale to a larger range for better PCA stability
            rr.log(
                f"pca_segment_{i}/points",
                rr.Points3D(points_, colors=np.array([[0.2, 0.8, 1.0]]))  # cyan points
            )

            y_min, y_max = 0.0, 100.0   # Y axis range
            z_min, z_max = 0.0, 100.0   # Z axis range
            num_points = 50              # resolution

            # Create a grid in Y-Z
            yy, zz = np.meshgrid(
                np.linspace(y_min, y_max, num_points),
                np.linspace(z_min, z_max, num_points)
            )

            # For Y-Z plane, X can be 0
            xx = np.zeros_like(yy)

            # Stack into N x 3 points
            yz_plane_points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

            # Log to Rerun
            rr.log(
                "yz_plane",
                rr.Points3D(yz_plane_points, colors=np.array([[0.5, 0.5, 0.5]]))  # gray
            )

            origin = np.array([[0.0, 0.0, 0.0]])

            # Define axis vectors (length 50 to match your scale)
            vectors = np.array([
                [50.0, 0.0, 0.0],  # X-axis
                [0.0, 50.0, 0.0],  # Y-axis
                [0.0, 0.0, 50.0]   # Z-axis
            ])

            # Define colors for RGB
            colors = np.array([
                [1.0, 0.0, 0.0],  # Red for X
                [0.0, 1.0, 0.0],  # Green for Y
                [0.0, 0.0, 1.0]   # Blue for Z
            ])

            # Log axes as arrows in Rerun
            rr.log(
                "axes",
                rr.Arrows3D(
                    origins=np.tile(origin, (3, 1)),  # repeat origin for 3 axes
                    vectors=vectors,
                    colors=colors
                )
            )

            mean_x = np.mean(points_[:, 0])

            # Select points close to mean_x (within a small epsilon)
            epsilon = 1.0  # adjust tolerance
            mask_x = np.abs(points_[:, 0] - mean_x) < epsilon
            points_slice = points_[mask_x]

            if points_slice.shape[0] < 2:
                print("Not enough points near mean_x to compute derivative.")
                label = "unknown"
                avg_slope = 0
            else:
                # Sort by Z to make derivative meaningful
                points_slice = points_slice[np.argsort(points_slice[:, 2])]

                # Compute differences
                dz = np.diff(points_slice[:, 2])
                dy = np.diff(points_slice[:, 1])

                eps = 1e-2  # or 0.1, adjust based on units
                valid = np.abs(dy) > eps
                dz_filtered = dz[valid]
                dy_filtered = dy[valid]
                dzdy = dz_filtered / dy_filtered
                avg_slope = np.mean(dzdy)
                # Debug: print slope differences
                #print(f"dz_filtered: {dz_filtered} and dy_filtered: {dy_filtered}")
                
                if len(dzdy) > 1:
                    # Compute slope difference
                    slope_start = dzdy[0]
                    slope_end = dzdy[-1]
                    slope_diff = slope_end - slope_start

                    print(f"Segment {i}:")
                    print(f"  Start slope: {slope_start:.4f}")
                    print(f"  End slope:   {slope_end:.4f}")
                    print(f"  Δslope (end - start): {slope_diff:.4f}")
                else:
                    print(f"Not enough points in segment {i} to compute slope difference.")
                    label = "unknown"
                    avg_slope = 0


                # Average derivative#Depth Y
                #avg_slope = np.mean(dzdy)
                print(f"Segment {i}: Average dZ/dY at mean X={mean_x:.2f}: {avg_slope:.4f}")

                threshold = 0.8  # adjust based on expected slope
                label = "flat" if np.abs(avg_slope) > threshold else "non-flat"
            labels.append(label)
            slopes.append(np.abs(avg_slope))
        traversability = normalize_slopes_inverted(slopes)
        frame_data = {str(i): float(score) for i, score in enumerate(traversability)}
        all_frames_data[str(frame_num)] = frame_data

            #print("-----------------------------------------------------------------------------------")
        print(f"Labels for segments: {labels}")
        #print(f"Traversability scores for segments: {traversability}")
        if frame_num == 0 or frame_num == 1:
            visualize_sam_grid_proposals_terrain(input_image_bgr, proposals, labels)
            pass
        save_file_flat = os.path.join(save_folder, f"op_frame_dronelab_{frame_num:05d}_terrain.png")
        visualize_sam_grid_proposals_terrain_save(frame_num,input_image_bgr, proposals, labels, save_path=save_file_flat)
        plt.show()
        mask_pre.close()
    # Save to JSON
    with open(output_json, "w") as f:
        json.dump(all_frames_data, f, indent=4)

print(f"Saved traversability scores to {output_json}")
