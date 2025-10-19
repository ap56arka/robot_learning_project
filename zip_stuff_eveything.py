import os
import zipfile

parent_folder = "output_dronelabarka"
parent_zip_name = f"{parent_folder}.zip"

# List all chunk zip files
chunk_zips = [
    os.path.join(parent_folder, f)
    for f in os.listdir(parent_folder)
    if f.endswith("_npz.zip")
]

# Create parent zip and add all chunk zips
with zipfile.ZipFile(parent_zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for chunk_zip in chunk_zips:
        zipf.write(chunk_zip, arcname=os.path.basename(chunk_zip))

print(f"✅ All chunk zip files combined into '{parent_zip_name}'")
