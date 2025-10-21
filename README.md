# VLM Pipeline for Robot Traversability Planning

A pipeline that uses Vision Language Models (VLM) to assess terrain traversability and plan navigation paths for robots using only camera input.

### Prerequisites
- Python 3.8+
- Gemini API key

### Installation

1. **Create virtual environment**
   ```bash
   python -m venv vlm_env
   source vlm_env/bin/activate  # On Windows: vlm_env\Scripts\activate
   pip install -r final_gemini_requirements.txt
#### Get Gemini API Key
- Visit Google AI Studio
- Create API key for Gemini  
- Keep the key ready for setup

#### Setup & Run
- Download required files and place in VLM-pipeline folder:
  - Two test videos (drone_lab_masked.mp4, drone_lab_original.mp4)
  - `traversability_scores_dronelab.json` terrain score file
  - Zip folder (output_drone_lab.zip)


#### Configure API Key
- Open `Video_VLM.ipynb`
- Change the path at two places in main function and the process_video_frames function 
- In the main function, replace:
```python
  API_KEY = "your_actual_gemini_api_key_here"
