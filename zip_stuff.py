import os
import zipfile
# Parent folder containing all chunk folders
parent_folder = "output_dronelabarka"

# Iterate through all chunk folders
for folder_name in os.listdir(parent_folder):
    chunk_folder = os.path.join(parent_folder, folder_name)
    if os.path.isdir(chunk_folder) and folder_name.startswith("output_dronelabarka_chunk_"):
        zip_name = f"{chunk_folder}_npz.zip"
        
        # Create zip for this chunk
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in os.listdir(chunk_folder):
                if file.endswith(".npz"):
                    file_path = os.path.join(chunk_folder, file)
                    zipf.write(file_path, arcname=file)
        
        print(f"✅ Zipped all .npz files in '{chunk_folder}' into '{zip_name}'")

