import os
import pydicom
import pandas as pd
from PIL import Image
import numpy as np
import shutil
# PATHS (using forward slashes)
base_path = "C:/Users/VIRAJ/Downloads/archive/TheChineseMammographyDatabase"
dicom_root = f"{base_path}/CMMD"
metadata_csv = f"{base_path}/CMMD_clinicaldata_revision.csv"
output_base = "C:/Users/VIRAJ/OneDrive/Documents/AI-based-cancer-detection-project/data/processed"

# Output directories
dicom_output = f"{output_base}/malignant_dicoms"
png_output = f"{output_base}/malignant_png"
os.makedirs(dicom_output, exist_ok=True)
os.makedirs(png_output, exist_ok=True)

def process_dicom(dcm_path, patient_id, filename):
    """Process a single DICOM file"""
    try:
        # 1. Copy original DICOM
        dicom_dest = f"{dicom_output}/{patient_id}_{filename}"
        shutil.copy2(dcm_path, dicom_dest)
        
        # 2. Convert to PNG
        ds = pydicom.dcmread(dcm_path)
        img_array = ds.pixel_array
        
        # Normalize to 8-bit
        if img_array.dtype != np.uint8:
            img_array = ((img_array - img_array.min()) / 
                        (img_array.max() - img_array.min()) * 255).astype(np.uint8)
        
        # Save PNG
        png_filename = f"{patient_id}_{os.path.splitext(filename)[0]}.png"
        Image.fromarray(img_array).save(f"{png_output}/{png_filename}")
        
        return True
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return False

# Main processing
df = pd.read_csv(metadata_csv)
malignant_cases = df[df["classification"] == "Malignant"]
print(f"Found {len(malignant_cases)} malignant cases")

success_count = 0
for patient_id in malignant_cases["ID1"]:
    patient_path = f"{dicom_root}/{patient_id}"
    if not os.path.exists(patient_path):
        continue
        
    print(f"\nProcessing {patient_id}:")
    for root, _, files in os.walk(patient_path):
        for file in files:
            if file.lower().endswith(".dcm"):
                if process_dicom(
                    dcm_path=f"{root}/{file}",
                    patient_id=patient_id,
                    filename=file
                ):
                    success_count += 1

print("\n=== RESULTS ===")
print(f"Successfully processed {success_count} DICOM files")
print(f"DICOM copies: {dicom_output}")
print(f"PNG images: {png_output}")

