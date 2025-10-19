import os
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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
import re
import json

def extract_video_frames(
    video_path: str,
    start_frame: int = 0,
    end_frame: int = None,
    frame_step: int = 2,
    frames_per_chunk: int = 100,
    output_root: str = None
):

    # Basic setup
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_root is None:
        output_root = video_name  # Parent folder is just the video name

    os.makedirs(output_root, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(end_frame if end_frame is not None else total_frames, total_frames)

    print(f"Video: {video_name}")
    print(f"Total frames: {total_frames}")
    print(f"Extracting from frame {start_frame} to {end_frame}")
    print(f"Saving once every {frame_step} frames")
    print(f"{frames_per_chunk} frames per chunk")

    # Initialize
    frame_idx = 0
    save_count = 0
    chunk_idx = 0
    current_chunk_dir = os.path.join(output_root, f"{video_name}_chunk_{chunk_idx}")
    os.makedirs(current_chunk_dir, exist_ok=True)

    # Read video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < start_frame:
            frame_idx += 1
            continue
        if frame_idx >= end_frame:
            break

        # Save frame if matches step
        if (frame_idx - start_frame) % frame_step == 0:
            frame_filename = os.path.join(current_chunk_dir, f"frame_{frame_idx:04d}.png")
            cv2.imwrite(frame_filename, frame)
            save_count += 1

            # Create new chunk only if we expect more frames to come
            if save_count % frames_per_chunk == 0:
                # Peek ahead: if more frames remain
                if frame_idx + frame_step < end_frame:
                    chunk_idx += 1
                    current_chunk_dir = os.path.join(output_root, f"{video_name}_chunk_{chunk_idx}")
                    os.makedirs(current_chunk_dir, exist_ok=True)

        frame_idx += 1

    cap.release()
    print(f"Saved {save_count} frames across {chunk_idx + 1} chunk folders inside '{output_root}'")

    return chunk_idx

def load_first_frame_from_chunk(video_name: str, chunk_number: int, base_path: str = "."):
    
    chunk_folder = os.path.join(base_path, video_name, f"{video_name}_chunk_{chunk_number}")

    if not os.path.exists(chunk_folder):
        raise FileNotFoundError(f"Chunk folder not found: {chunk_folder}")

    # Get all PNG files in sorted order
    image_files = sorted([
        f for f in os.listdir(chunk_folder)
        if f.lower().endswith(".png")
    ])

    if not image_files:
        raise FileNotFoundError(f"No PNG images found in {chunk_folder}")

    # Load the first image as RGB
    first_frame_path = os.path.join(chunk_folder, image_files[0])
    image = Image.open(first_frame_path).convert("RGB")

    print(f"\nLoaded first frame: {first_frame_path} from chunk {chunk_number}")
    return image

def resize_image(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    return image.resize((target_width, target_height), Image.BILINEAR)

def load_and_resize_chunk_frames(video_name: str, chunk_num: int, target_width, target_height, base_path: str = "."):

    chunk_folder = os.path.join(base_path, f"{video_name}", f"{video_name}_chunk_{chunk_num}")
    
    if not os.path.exists(chunk_folder):
        raise FileNotFoundError(f"Chunk folder not found: {chunk_folder}")
    
    # List and sort all images in the chunk folder
    frame_files = sorted([f for f in os.listdir(chunk_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    
    if not frame_files:
        raise FileNotFoundError(f"No images found in {chunk_folder}")
    
    # Load all frames
    frames = [Image.open(os.path.join(chunk_folder, f)).convert("RGB") for f in frame_files]
    
    
    # Resize all frames
    frames_resized = []
    for frame in frames:
        frames_resized.append(resize_image(frame, target_width, target_height))
    
    print(f"Loaded and resized {len(frames_resized)} frames from chunk {chunk_num}")
    return frames_resized

def create_output_folder_structure(video_name: str, total_chunks: int, base_path: str = "."):
    parent_folder = os.path.join(base_path, f"output_{video_name}")
    os.makedirs(parent_folder, exist_ok=True)

    chunk_folders = []
    for chunk_idx in range(total_chunks + 1):
        chunk_folder = os.path.join(parent_folder, f"output_{video_name}_chunk_{chunk_idx}")
        os.makedirs(chunk_folder, exist_ok=True)
        chunk_folders.append(chunk_folder)

    print(f"Created parent folder: {parent_folder}")
    print(f"Created {len(chunk_folders)} chunk subfolders")

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

def visualize_sam_grid_proposals(image: np.ndarray, proposals: list, save_path: Optional[str] = None):
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

def overlay_and_save_frame(frame_pil, masks_dict, colors, video_name, chunk_num, frame_idx, base_output_path="."):
    """Overlay object masks on a frame using their corresponding colors."""
    frame = np.array(frame_pil).copy()  # Convert to numpy RGB
    for obj_id, mask in masks_dict.items():
        if mask is None:
            continue

        if isinstance(mask, torch.Tensor):
            mask = (mask > 0.5).cpu().numpy().astype(np.uint8)
        else:
            mask = (mask > 0.5).astype(np.uint8)

        color = colors[int(obj_id) % len(colors)]
        for c in range(3):
            frame[:, :, c] = np.where(
                mask == 1,
                0.6 * frame[:, :, c] + 0.4 * color[c],
                frame[:, :, c],
            )

    # Define output folder and path
    chunk_folder = os.path.join(base_output_path, f"output_{video_name}", f"output_{video_name}_chunk_{chunk_num}")
    os.makedirs(chunk_folder, exist_ok=True)

    output_filename = f"output_{video_name}_frame_{frame_idx:04d}.png"
    out_path = os.path.join(chunk_folder, output_filename)

    # Save as PNG
    Image.fromarray(frame.astype(np.uint8)).save(out_path)
    
def visualize_frame_masks(
    masks_dict: dict,
    colors_dict: dict,
    video_name: str,
    chunk_num: int,
    frame_idx: int,
    frame_shape: tuple,
    base_output_path: str = ".",
    blend_ratio: float = 1.0,
    circle_brightness: float = 0.75,
    outline_darkness: float = 0.5,
    background_gray: int = 20,
    text_offset: int = 0,
    circle_radius: int = 6
):
    H, W = frame_shape
    #print(frame_shape)
    frame = np.full((H, W, 3), background_gray, dtype=np.uint8)

    # Convert masks to numpy arrays and compute areas
    masks_list = []
    for obj_id, mask in masks_dict.items():
        #print(f"Mask shape : {mask.shape}")
        if mask is None:
            continue
        if isinstance(mask, np.ndarray):
            #print(f"Mask shape : {mask.shape}")
            mask_np = (mask.squeeze() > 0.5).astype(np.uint8)
        else:
            mask_np = (mask.detach().cpu().squeeze() > 0.5).numpy().astype(np.uint8)
        area = np.sum(mask_np)
        if area > 0:
            masks_list.append({'obj_id': obj_id, 'mask': mask_np, 'area': area})

    # Sort masks by area descending (largest first)
    masks_list.sort(key=lambda x: x['area'], reverse=True)

    # Dictionary to store final 2D masks per object
    final_masks = {mask_dict['obj_id']: np.zeros((H, W), dtype=np.uint8) for mask_dict in masks_list}

    # Render masks with area priority
    for mask_dict in masks_list:
        mask = mask_dict['mask']
        obj_id = mask_dict['obj_id']
        color = np.array(colors_dict[obj_id], dtype=np.uint8)
        frame[mask == 1] = color
    
    masks_list_sorted_masks = sorted(masks_list, key=lambda x: x['area'], reverse=False)

    occupied_pixels = np.zeros((H, W), dtype=bool)  # tracks pixels assigned to smaller objects

    for mask_dict in masks_list_sorted_masks:
        mask = mask_dict['mask'].astype(bool)
        obj_id = mask_dict['obj_id']

        # Remove pixels already taken by smaller objects
        mask_clean = mask & (~occupied_pixels)

        final_masks[obj_id][mask_clean] = 1
        # Update occupied pixels
        occupied_pixels = occupied_pixels | mask_clean

    # Convert to PIL
    image_pil = Image.fromarray(frame.astype(np.uint8))
    draw = ImageDraw.Draw(image_pil)

    # Font setup
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 10)
    except IOError:
        font = ImageFont.load_default()

    # Draw circles and numbers ONLY on the first frame
    if frame_idx == 0:
        for mask_dict in masks_list:
            obj_id = mask_dict['obj_id']
            color = np.array(colors_dict[obj_id], dtype=np.uint8)

            # Find all pixels in frame that have this mask's color
            matches = np.all(frame == color, axis=-1)
            ys, xs = np.where(matches)
            if len(xs) == 0:
                continue

            # Pick a random pixel
            idx = np.random.randint(0, len(xs))
            cx, cy = xs[idx], ys[idx]

            # Brightened fill and darker outline
            fill_color = tuple(np.clip(color * circle_brightness, 0, 255).astype(np.uint8).tolist())
            outline_color = tuple(np.clip(color * outline_darkness, 0, 255).astype(np.uint8).tolist())

            # Draw circle
            r = circle_radius
            draw.ellipse((cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2), fill=outline_color)
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill_color)

            # Draw number
            draw.text((cx, cy + text_offset), str(obj_id),
                      fill=(255, 255, 255),
                      font=font,
                      anchor="mm")

    # Save output
    chunk_folder = os.path.join(base_output_path,
                                f"output_{video_name}",
                                f"output_{video_name}_chunk_{chunk_num}")
    os.makedirs(chunk_folder, exist_ok=True)
    out_path = os.path.join(chunk_folder, f"output_{video_name}_frame_{frame_idx:04d}.png")
    image_pil.save(out_path, format="PNG")

    final_masks_str_keys = {str(k): v for k, v in final_masks.items()}
    npz_out_path = os.path.join(chunk_folder, f"output_{video_name}_frame_{frame_idx:04d}_masks.npz")
    np.savez_compressed(npz_out_path, **final_masks_str_keys)

def concatenate_chunks_to_video(
    parent_folder: str,
    chunk_folder_pattern: str = "_chunk_",
    frame_file_pattern: str = r"_frame_(\d+)\.png",
    output_path: str = "final_video.mp4",
    fps: int = 4
):

    # Helper to extract chunk index
    def get_chunk_index(folder_name):
        m = re.search(rf"{re.escape(chunk_folder_pattern)}(\d+)$", folder_name)
        return int(m.group(1)) if m else -1

    # Get all chunk folders sorted by chunk index
    chunk_folders = sorted(
        [os.path.join(parent_folder, d) for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))],
        key=get_chunk_index
    )

    all_frame_files = []

    # Collect frames from all chunks
    for chunk_folder in chunk_folders:
        frame_files = []
        for f in os.listdir(chunk_folder):
            if f.lower().endswith(".png"):
                match = re.search(frame_file_pattern, f, re.IGNORECASE)
                if match:
                    frame_files.append((int(match.group(1)), os.path.join(chunk_folder, f)))
        frame_files_sorted = [f for idx, f in sorted(frame_files, key=lambda x: x[0])]
        all_frame_files.extend(frame_files_sorted)

    if not all_frame_files:
        raise RuntimeError("No frames found to create video.")

    # Read first frame to get video size
    first_frame = cv2.imread(all_frame_files[0])
    if first_frame is None:
        raise RuntimeError(f"Cannot read first frame: {all_frame_files[0]}")

    frame_h, frame_w = first_frame.shape[:2]

    # Initialize video writer
    video_writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_w, frame_h)
    )

    # Write all frames
    for frame_file in all_frame_files:
        frame = cv2.imread(frame_file)
        if frame is not None:
            video_writer.write(frame)

    video_writer.release()
    print(f"✅ Final video saved at {output_path} ({len(all_frame_files)} frames, {fps} FPS)")

# ===========================================

# Video pre processing config
video_path = "dronelabarka.mp4"
video_name = os.path.splitext(os.path.basename(video_path))[0]
start_frame = 2         # Starting frame number (starting number is 0)
end_frame = None        # entire video will be processed
frame_step = 2          # Every nth frame will be saved
frames_per_chunk = 50   # SAM will propogate for N frames until resampling
output_root = None      # Will be saved with same video name

# Image resizing parameters
target_width = 480
target_height = None

#SAM grip proposal configs
grid_size = 8
confidence_threshold = 0.5

# ================================================

print("\n===== Preprocessing input video =====")
total_chunks = extract_video_frames(video_path, start_frame, end_frame, frame_step, frames_per_chunk, output_root)

print("\n===== Creating output folders =====")
create_output_folder_structure(video_name, total_chunks)

print("\n===== Loading SAM =====")
sam_model, sam_processor, device = load_sam_model(model_size='base')
sam2_video_model = Sam2VideoModel.from_pretrained("facebook/sam2.1-hiera-tiny").to(device, dtype=torch.bfloat16)
sam2_video_processor = Sam2VideoProcessor.from_pretrained("facebook/sam2.1-hiera-tiny")

for chunk_num in range(total_chunks + 1):

    print(f"\n====== Processing chunk {chunk_num} ======")

    chunk_folder = os.path.join(f"output_{video_name}", f"output_{video_name}_chunk_{chunk_num}")
    first_image = load_first_frame_from_chunk(video_name, chunk_num)

    if chunk_num == 0:
        image_width, image_height = first_image.size
        target_width = target_width or image_width
        target_height = int(image_height / image_width * target_width)
    
    first_image_resized = resize_image(first_image, target_width=target_width, target_height=target_height)
    print(f"First frame from chunk {chunk_num} resized")

    #Propose masks on first image in chunk
    proposals = generate_sam_proposals(first_image_resized, sam_model, sam_processor, device, grid_size=grid_size, confidence_threshold=confidence_threshold)

    #Visualise and save proposals in chunk folder
    image_np = np.array(first_image_resized)
    image_output_path = os.path.join(chunk_folder, "sam_grid_proposals.png")
    visualize_sam_grid_proposals(image_np, proposals, save_path=image_output_path)

    # Resize all the images in the chunk for video processing
    video_frames_resized = load_and_resize_chunk_frames(video_name, chunk_num, target_width, target_height)

    # Initialize SAM2 inference session with resized frames
    inference_session = sam2_video_processor.init_video_session(video=video_frames_resized, inference_device=device, dtype=torch.bfloat16)
    print(f"Initialised inference session for chunk : {chunk_num} containing {len(video_frames_resized)} images")

    #Add the masks of the proposals given by SAM
    obj_ids = list(range(len(proposals)))
    masks = [p["mask"].astype(np.uint8) for p in proposals]

    # Add them in one call
    sam2_video_processor.add_inputs_to_inference_session(
        inference_session=inference_session,
        frame_idx=0,
        obj_ids=obj_ids,          # list of IDs
        input_masks=masks,        # list of masks
        original_size=video_frames_resized[0].size,
    )
    print(f"Added {len(proposals)} initial masks to the SAM2 session")

    # Run segmentation on the first frame to register starting frame
    outputs = sam2_video_model(inference_session=inference_session, frame_idx=0)

    video_res_masks = sam2_video_processor.post_process_masks(
    [outputs.pred_masks],
    original_sizes=[[inference_session.video_height, inference_session.video_width]],
    binarize=True
    )[0]

    print(f"Segmentation on first frame done. Shape: {video_res_masks.shape}")

    # --- Run propagation (streaming) ---
    video_segments = {}
    print("Propagating masks across video... \n")

    # Assign a unique color for each object ID in the session
    obj_ids = list(inference_session.obj_ids)
    rng = np.random.default_rng(42)
    colors_dict = {obj_id: (rng.random(3) * 255).astype(np.uint8) for obj_id in obj_ids}

    colors_dict_serializable = {int(k): v.tolist() for k, v in colors_dict.items()}
    # Save to JSON
    chunk_number = chunk_num  # assuming chunk_num is defined in your loop
    json_filename = f"color_dict_chunk_{chunk_number}.json"
    json_path = os.path.join(f"output_{video_name}", f"output_{video_name}_chunk_{chunk_number}", json_filename)

    with open(json_path, "w") as f:
        json.dump(colors_dict_serializable, f, indent=4)
    print(f"✅ Saved colors dictionary to {json_path}")

    for sam2_video_output in sam2_video_model.propagate_in_video_iterator(inference_session):

        # Convert tensor masks → numpy
        video_res_masks = sam2_video_processor.post_process_masks(
            [sam2_video_output.pred_masks],
            original_sizes=[[inference_session.video_height, inference_session.video_width]],
            binarize=True,
        )[0]

        # Store frame-wise masks
        frame_idx = sam2_video_output.frame_idx
        video_segments[frame_idx] = {
            obj_id: video_res_masks[i] for i, obj_id in enumerate(inference_session.obj_ids)
        }
        
        #Overlay masks and store each frame in chunk
        frame_pil = video_frames_resized[frame_idx]
        #overlay_and_save_frame(frame_pil, video_segments[frame_idx], colors, video_name, chunk_num, frame_idx)
        visualize_frame_masks(video_segments[frame_idx], colors_dict, video_name, chunk_num, frame_idx, (target_height, target_width))


print("\n===== Creating video =====")
output_folder = f"output_{video_name}"
output_path = f"{video_name}_final_output.mp4"
concatenate_chunks_to_video(parent_folder=output_folder, output_path=output_path, fps=4)

input_folder = f"{video_name}"
input_path = f"{video_name}_input.mp4"
concatenate_chunks_to_video(parent_folder=input_folder, output_path=input_path, frame_file_pattern= r"frame_(\d+)\.png", fps=4)











