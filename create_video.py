def concatenate_chunks_to_video(
    parent_folder: str,
    chunk_folder_pattern: str = "_chunk_",
    frame_file_pattern: str = r"frame_(\d+)\.png",
    output_path: str = "final_video.mp4",
    fps: int = 4
):
    import os, re, cv2

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
    target_height = int(480 * frame_h / frame_w)
    target_size = (480, target_height)
    # Initialize video writer
    video_writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        target_size
    )
    # Write all frames
    target_size = (480, target_height)
    count = 0
    for frame_file in all_frame_files:
        frame = cv2.imread(frame_file)
        frame_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        if frame is not None:
            video_writer.write(frame_resized)

    video_writer.release()
    print(f"✅ Final video saved at {output_path} ({len(all_frame_files)} frames, {fps} FPS)")



output_folder = f"dronelabarka"
output_path = f"drone_labarka_input.mp4"
concatenate_chunks_to_video(parent_folder=output_folder, output_path=output_path, frame_file_pattern= r"frame_(\d+)\.png", fps=4)
