import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Configuration
JPEG_ROOT = "C:/Users/VIRAJ/Downloads/archive/jpeg"
OUTPUT_DIR = r"C:/Users/VIRAJ/OneDrive\Documents/AI-based-cancer-detection-project/data/processed/malignant"
TARGET_SIZE = (224, 224)

def extract_study_id(dicom_path):
    """Extract study ID from DICOM path"""
    parts = dicom_path.split('/')
    return parts[-2] if len(parts) > 1 else None

def find_jpeg_file(study_id, jpeg_root):
    """Find JPEG file matching study ID"""
    for root, _, files in os.walk(jpeg_root):
        for file in files:
            if study_id in root or study_id in file:
                if file.lower().endswith(('.jpg', '.jpeg')):
                    return os.path.join(root, file)
    return None

def main():
    # Create output directories
    os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "test"), exist_ok=True)

    # Process both train and test sets
    for dataset_type in ['train', 'test']:
        csv_path = f"C:/Users/VIRAJ/Downloads/archive/csv/mass_case_description_{dataset_type}_set.csv"
        
        try:
            df = pd.read_csv(csv_path)
            malignant_cases = df[df['pathology'].str.upper() == 'MALIGNANT']
            print(f"Found {len(malignant_cases)} malignant cases in {dataset_type} set")

            # Process each malignant case
            success_count = 0
            for _, row in tqdm(malignant_cases.iterrows(), total=len(malignant_cases), desc=f"Processing {dataset_type} set"):
                try:
                    # Extract study ID from image file path
                    study_id = extract_study_id(row['image file path'])
                    if not study_id:
                        continue

                    # Find corresponding JPEG file
                    jpeg_path = find_jpeg_file(study_id, JPEG_ROOT)
                    if not jpeg_path:
                        continue

                    # Process and save image
                    img = Image.open(jpeg_path).convert('L')  # Convert to grayscale
                    img = img.resize(TARGET_SIZE)
                    
                    output_path = os.path.join(OUTPUT_DIR, dataset_type, f"{study_id}.png")
                    img.save(output_path)
                    success_count += 1

                except Exception as e:
                    continue

            print(f"Successfully processed {success_count} malignant cases from {dataset_type} set")

        except Exception as e:
            print(f"Error processing {dataset_type} set: {str(e)}")

if __name__ == "__main__":
    main()