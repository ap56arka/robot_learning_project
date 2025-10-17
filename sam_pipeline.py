# Import required libraries
import os
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import torch
from tqdm.notebook import tqdm
from transformers import (
    SamModel,
    SamProcessor,
    Sam2VideoModel,
    Sam2VideoProcessor
)
from transformers.video_utils import load_video
import av

# ===========================================
# Configure the script
# Video pre processing
video_path = "test_sam_video.mp4"
start_frame = 10                # Start saving from this frame index
num_frames = 110                 # Number of frames to save
num_frames_to_use = 50          # number of frames to sample
target_width = 480


#SAM grip proposal configs
grid_size = 6
confidence_threshold = 0.3

# Save output folders
folder_name = "test_sam_video"
viz_folder = "sam2_viz_test_sam_video"
output_path = "sam2_tracked_test_video.mp4"
#=============================================
def load_sam_model(model_size: str = "base", device: str = None) -> Tuple:
    """
    Load SAM model from Hugging Face using transformers.

    Args:
        model_size: Model size - 'base', 'large', 'huge'
        device: Device to use (cuda/cpu)

    Returns:
        Tuple of (model, processor, device)
    """
    print(f"Loading SAM model ({model_size})...")

    # Model configurations from HuggingFace
    model_configs = {
        'base': 'facebook/sam-vit-base',
        'large': 'facebook/sam-vit-large',
        'huge': 'facebook/sam-vit-huge'
    }

    model_id = model_configs.get(model_size, model_configs['base'])

    processor = SamProcessor.from_pretrained(model_id)
    model = SamModel.from_pretrained(model_id)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    print(f"SAM model loaded on device: {device}")
    return model, processor, str(device)


def visualize_sam_grid_proposals(image: np.ndarray, proposals: list, save_path: Optional[str] = None):
    """
    Visualize all the final SAM proposals (after IoU filtering and confidence thresholding)
    and optionally save the figure as a PNG image.

    Args:
        image (np.ndarray): Input image (RGB).
        proposals (list): List of proposal dicts with keys ['mask', 'area', 'point', 'confidence'].
        save_path (str, optional): If provided, saves the visualization as a PNG file.
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

    for proposal in proposals_sorted:
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
        plt.text(point[0] + 5, point[1] + 5, f"{conf:.2f}", color='white', fontsize=8)

    plt.title(f"Grid-based SAM Proposals ({len(proposals)} masks)")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        print(f"✅ Saved visualization to: {save_path}")

    plt.show()

def generate_sam_proposals(image: Image.Image,
                          sam_model,
                          sam_processor,
                          device: str,
                          grid_size: int = 6,
                          confidence_threshold: float = 0.5) -> List[Dict]:
    """Generate object proposals using SAM with a grid of point prompts."""
    width, height = image.size
    
    # Generate grid of point prompts
    x_points = np.linspace(width * 0.1, width * 0.9, grid_size)
    y_points = np.linspace(height * 0.1, height * 0.9, grid_size)
    
    proposals = []
    processed_masks = []
    
    tqdm.write(f"Generating SAM proposals with {grid_size}x{grid_size} grid...")
    
    for i, x in enumerate(x_points):
        for j, y in enumerate(y_points):
            input_points = [[[x, y]]]
            
            try:
                inputs = sam_processor(
                    images=image,
                    input_points=input_points,
                    return_tensors="pt"
                )
                
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = sam_model(**inputs)
                
                masks = sam_processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu()
                )
                
                batch_masks = masks[0]
                if len(batch_masks) == 0:
                    continue
                
                point_masks = batch_masks[0]
                if len(point_masks) == 0:
                    continue
                
                best_mask_idx = 0
                best_score = 0.0

                if hasattr(outputs, 'iou_scores') and outputs.iou_scores is not None:
                    try:
                        # Extract IoU scores for [batch 0, point 0, all masks]
                        iou_scores = outputs.iou_scores[0, 0, :].cpu().numpy()

                        if len(iou_scores) > 0:
                            best_mask_idx = int(np.argmax(iou_scores))
                            best_score = float(np.max(iou_scores))

                            # Skip if below threshold
                            if best_score < confidence_threshold:
                                continue
                    except Exception as e:
                        tqdm.write(f"IoU extraction failed: {e}")
                        best_mask_idx = 0
                        best_score = 0.0

                
                mask = point_masks[best_mask_idx]
                
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.cpu().numpy().astype(bool)
                else:
                    mask_np = np.array(mask).astype(bool)
                
                # Check for duplicates
                is_duplicate = False
                for existing_mask in processed_masks:
                    overlap = np.sum(mask_np & existing_mask)
                    union = np.sum(mask_np | existing_mask)
                    if union > 0 and overlap / union > 0.8:
                        is_duplicate = True
                        break
                
                if not is_duplicate and np.sum(mask_np) > 100:
                    proposals.append({
                        'mask': mask_np,
                        'area': np.sum(mask_np),
                        'point': [x, y],
                        'confidence': best_score
                    })
                    processed_masks.append(mask_np)
                    
            except Exception as e:
                continue
    
    tqdm.write(f"Generated {len(proposals)} unique segment proposals")
    return proposals

def resize_image_keep_aspect(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    return image.resize((target_width, target_height), Image.BILINEAR)


# Create folder named after the video (without extension)
os.makedirs(folder_name, exist_ok=True)

# Open the video
cap = cv2.VideoCapture(video_path)
frame_idx = 0     # actual frame index in the video
save_idx = 0      # index for saved frames, starts from 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Only save frames in the configured range
    if frame_idx >= start_frame and save_idx < num_frames:
        frame_filename = os.path.join(folder_name, f"frame_{save_idx:04d}.png")
        cv2.imwrite(frame_filename, frame)
        save_idx += 1

    frame_idx += 1

    # Stop if we've saved the required number of frames
    if save_idx >= num_frames:
        break

cap.release()
print(f"Saved {save_idx} frames to folder '{folder_name}' starting from frame {start_frame}")


# Folder containing video frames

# Get list of all images in the folder (png/jpg)
frame_files = [f for f in os.listdir(folder_name) if f.endswith((".png", ".jpg", ".jpeg"))]

frame_files = sorted(frame_files) 

# Load the first frame
first_frame_path = os.path.join(folder_name, frame_files[0])
image = Image.open(first_frame_path).convert("RGB")

# Resize the first frame before generating proposals
w, target_height = image.size
image_resized = resize_image_keep_aspect(image, target_width=target_width, target_height=target_height)

print(f"Loaded first frame: {first_frame_path}")

sam_model, sam_processor, device = load_sam_model(model_size='base')
proposals = generate_sam_proposals(
    image_resized,
    sam_model,
    sam_processor,
    device,
    grid_size=grid_size,
    confidence_threshold=confidence_threshold
)

# Convert PIL image → NumPy array for plotting
image_np = np.array(image_resized)
# Visualize proposals
image_output_path = os.path.join(folder_name, "sam_grid_proposals.png")
visualize_sam_grid_proposals(image_np, proposals, save_path=image_output_path)

#Load SAM 2 Video
print("Trying to load SAM2 video")
sam2_video_model = Sam2VideoModel.from_pretrained("facebook/sam2.1-hiera-tiny").to(device, dtype=torch.bfloat16)
sam2_video_processor = Sam2VideoProcessor.from_pretrained("facebook/sam2.1-hiera-tiny")

# Sample frames from the stored video frames

# Get list of all saved frame files
frame_files = [f for f in os.listdir(folder_name) if f.endswith((".png", ".jpg", ".jpeg"))]
frame_files = sorted(frame_files)  # ensure proper order

# Load only the sampled frames as PIL images
total_saved_frames = len(frame_files)
sample_indices = np.linspace(0, total_saved_frames - 1, num_frames_to_use, dtype=int)
video_frames = [Image.open(os.path.join(folder_name, frame_files[i])).convert("RGB") for i in sample_indices]

print(f"Loaded {len(video_frames)} sampled frames from folder '{folder_name}'")

## Resize frames to a smaller resolution (e.g., 480px width, maintain aspect ratio)
target_width = 480
video_frames_resized = []

for frame in video_frames:
    # Convert numpy array → PIL Image
    if isinstance(frame, np.ndarray):
        frame = Image.fromarray(frame)

    frame_resized = resize_image_keep_aspect(frame, target_width, target_height)
    video_frames_resized.append(frame_resized)

print("All frames resized!")

# Initialize SAM2 inference session with resized frames
inference_session = sam2_video_processor.init_video_session(
    video=video_frames_resized,
    inference_device=device,  # could also use 'cpu' if GPU memory is tight
    dtype=torch.bfloat16,
)
print("Initialised inference session with resized 50 frames")

# Collect all masks and IDs together
obj_ids = list(range(len(proposals)))
masks = [p["mask"].astype(np.uint8) for p in proposals]

# Add them in one call
sam2_video_processor.add_inputs_to_inference_session(
    inference_session=inference_session,
    frame_idx=0,
    obj_ids=obj_ids,          # list of IDs
    input_masks=masks,        # list of masks
    original_size=video_frames[0].size,
)
print(f"✅ Added {len(proposals)} initial masks to the SAM2 session")

# --- Run segmentation on the first frame to register starting frame ---
outputs = sam2_video_model(
    inference_session=inference_session,
    frame_idx=0
)

video_res_masks = sam2_video_processor.post_process_masks(
    [outputs.pred_masks],
    original_sizes=[[inference_session.video_height, inference_session.video_width]],
    binarize=False
)[0]

print(f"✅ Segmentation on first frame done. Shape: {video_res_masks.shape}")

# --- Run propagation (streaming) ---
video_segments = {}

print("🚀 Propagating masks across video...")

# Folder to save visualized frames
os.makedirs(viz_folder, exist_ok=True)

# Random colors for each object
colors = np.random.randint(0, 255, (len(inference_session.obj_ids), 3), dtype=np.uint8)

for sam2_video_output in sam2_video_model.propagate_in_video_iterator(inference_session):
    # Convert tensor masks → numpy
    video_res_masks = sam2_video_processor.post_process_masks(
        [sam2_video_output.pred_masks],
        original_sizes=[[inference_session.video_height, inference_session.video_width]],
        binarize=False,
    )[0]

    # Store frame-wise masks
    frame_idx = sam2_video_output.frame_idx
    video_segments[frame_idx] = {
        obj_id: video_res_masks[i] for i, obj_id in enumerate(inference_session.obj_ids)
    }
    
    '''
    The variable video_segments ccontains all the masks for all the objects (with id) in each frame
    Variables:
    - video_segments (dict): Contains all masks for all objects across all frames.
        Format:
            {
                frame_idx_0: {
                    obj_id_0: mask_numpy_array_0,
                    obj_id_1: mask_numpy_array_1,
                    ...
                },
                frame_idx_1: {
                    obj_id_0: mask_numpy_array_0,
                    obj_id_1: mask_numpy_array_1,
                    ...
                },
                ...
            }
        Where:
            - frame_idx: int, index of the frame in the video (0-based).
            - obj_id: int or str, unique identifier for each detected object.
            - mask_numpy_array: np.ndarray of shape (H, W), dtype=np.uint8, with values 0 or 1,
            representing the segmentation mask of the object in that frame.
    '''

    # Get the corresponding resized frame
    frame_pil = video_frames_resized[frame_idx]
    frame = np.array(frame_pil).copy()  # RGB

    # Overlay all object masks
    for obj_id, mask in video_segments[frame_idx].items():
        if mask is None:
            continue

        if isinstance(mask, torch.Tensor):
            mask = (mask > 0.5).cpu().numpy().astype(np.uint8)
        else:
            mask = (mask > 0.5).astype(np.uint8)

        color = colors[int(obj_id) % len(colors)]

        for c in range(3):
            frame[:, :, c] = np.where(mask == 1,
                                      0.6 * frame[:, :, c] + 0.4 * color[c],
                                      frame[:, :, c])

    # Save visualized frame
    out_path = os.path.join(viz_folder, f"frame_{frame_idx:04d}.png")
    Image.fromarray(frame.astype(np.uint8)).save(out_path)

print(f"✅ Tracked {len(inference_session.obj_ids)} objects through {len(video_segments)} frames")

fps = 5  # adjust as needed

# Get list of frame files sorted by frame index
frame_files = sorted([f for f in os.listdir(viz_folder) if f.endswith(".png")])

# Read first frame to get size
first_frame = cv2.imread(os.path.join(viz_folder, frame_files[0]))
frame_h, frame_w = first_frame.shape[:2]

# Initialize video writer
out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_w, frame_h)
)

# Write frames to video
for frame_file in frame_files:
    frame_path = os.path.join(viz_folder, frame_file)
    frame = cv2.imread(frame_path)  # BGR, uint8
    out.write(frame)

out.release()
print(f"🎬 Video saved at {output_path}")