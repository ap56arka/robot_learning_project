# SAM2 Video Segmentation Pipeline 🎥🧠

This repository contains a **complete video segmentation pipeline** using **Meta's Segment Anything (SAM)** and **SAM2** models, implemented via the Hugging Face Transformers library.

The pipeline automates video frame extraction, segmentation proposal generation, temporal mask propagation across video chunks, visualization, and final video reconstruction.

---

## 🚀 Features

- **Frame Extraction:**  
  Splits an input video into frame chunks for manageable batch processing.

- **Grid-based SAM Proposals:**  
  Generates object proposals using a spatial grid of point prompts on the first frame of each chunk.

- **SAM2 Video Segmentation:**  
  Propagates SAM-generated masks across frames using the **SAM2 Video Model** (`facebook/sam2.1-hiera-tiny`).

- **Visualization:**  
  Overlays masks with unique colors and saves annotated frames along with per-frame mask data (`.npz`).

- **Video Reconstruction:**  
  Combines processed frames back into a final segmented output video.

---

## 🧩 Dependencies

Make sure you have the following Python packages installed:

```bash
pip install torch torchvision torchaudio
pip install transformers=4.56.2 huggingfacehub=0.34.2 av tqdm pillow opencv-python matplotlib numpy
```
For the script to rub propoerly, you must have thet test video in the same directory of the 
```python
python3 sam_pipeline_resampling.py
```

