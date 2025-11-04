import os
import pandas as pd
from PIL import Image
import glob

# =============================================
# CONFIGURATION (UPDATE THESE PATHS)
# =============================================
# Input Paths (using your exact structure)
csv_dir = "C:/Users/VIRAJ/OneDrive/Documents/AI-based-cancer-detection-project/data/raw/CBIS_DDM/csv"
jpeg_dir = "C:/Users/VIRAJ/OneDrive/Documents/AI-based-cancer-detection-project/data/raw/CBIS_DDM/jpeg"
output_dir = "C:/Users/VIRAJ/OneDrive/Documents/AI-based-cancer-detection-project/data/processed/CBIS-DDM"

# =============================================
# DATA LOADING
# =============================================
def load_data():
    """Load all CSV files with proper path handling"""
    clinical_files = [
        os.path.join(csv_dir, "calc_case_description_train_set.csv"),
        os.path.join(csv_dir, "calc_case_description_test_set.csv"),
        os.path.join(csv_dir, "mass_case_description_train_set.csv"),
        os.path.join(csv_dir, "mass_case_description_test_set.csv")
    ]
    
    clinical_dfs = []
    for file in clinical_files:
        df = pd.read_csv(file)
        # Standardize column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        clinical_dfs.append(df)
    
    clinical_df = pd.concat(clinical_dfs)
    clinical_df['patient_id_clean'] = clinical_df['patient_id'].str.extract(r'(\d+)')[0].str.zfill(5)
    
    # Load DICOM info
    dicom_df = pd.read_csv(os.path.join(csv_dir, "dicom_info.csv"))
    dicom_df.columns = dicom_df.columns.str.lower().str.replace(' ', '_')
    dicom_df['patient_id_clean'] = dicom_df['patientid'].str.extract(r'(\d+)')[0].str.zfill(5)
    
    return pd.merge(clinical_df, dicom_df, on='patient_id_clean', how='left')

# =============================================
# IMAGE PROCESSING (RELIABLE PATH FINDING)
# =============================================
def find_jpeg_file(row):
    """Robust JPEG finder that matches your exact folder structure"""
    # Try DICOM path first (e.g., 'CBIS-DDSM/jpeg/UID/1-1.jpg')
    if pd.notna(row.get('image_path')):
        try:
            parts = row['image_path'].split('/')
            if len(parts) >= 4 and parts[1] == 'jpeg':
                uid = parts[2]
                filename = parts[3]
                full_path = os.path.join(jpeg_dir, uid, filename)
                if os.path.exists(full_path):
                    return full_path
        except:
            pass
    
    # Try original image_file_path if DICOM path fails
    if pd.notna(row.get('image_file_path')):
        full_path = os.path.join(jpeg_dir, row['image_file_path'])
        if os.path.exists(full_path):
            return full_path
    
    # Final fallback: search by UID
    if pd.notna(row.get('image_path')):
        try:
            uid = row['image_path'].split('/')[2]
            jpeg_folder = os.path.join(jpeg_dir, uid)
            if os.path.exists(jpeg_folder):
                # Find first JPEG in folder
                for file in os.listdir(jpeg_folder):
                    if file.lower().endswith(('.jpg', '.jpeg')):
                        return os.path.join(jpeg_folder, file)
        except:
            pass
    
    return None

def process_images(df):
    """Process images with comprehensive error handling"""
    os.makedirs(os.path.join(output_dir, "BENIGN"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "MALIGNANT"), exist_ok=True)
    
    stats = {'benign': 0, 'malignant': 0, 'missing': 0, 'errors': 0}
    
    for idx, row in df.iterrows():
        try:
            jpeg_path = find_jpeg_file(row)
            if not jpeg_path:
                stats['missing'] += 1
                continue
            
            # Determine label
            pathology = str(row['pathology']).lower()
            label = "BENIGN" if 'benign' in pathology else "MALIGNANT"
            
            # Save image
            output_path = os.path.join(
                output_dir,
                label,
                f"{row['patient_id_clean']}_{idx}.png"
            )
            
            Image.open(jpeg_path).convert("RGB").save(output_path)
            stats[label.lower()] += 1
            
        except Exception as e:
            stats['errors'] += 1
            print(f"Error with {row['patient_id']}: {str(e)[:100]}")
    
    # Print report
    print("\n=== PROCESSING REPORT ===")
    print(f"✅ BENIGN: {stats['benign']} | MALIGNANT: {stats['malignant']}")
    print(f"❌ MISSING: {stats['missing']} | ERRORS: {stats['errors']}")

# =============================================
# MAIN EXECUTION
# =============================================
if __name__ == "__main__":
    print("=== STARTING PROCESSING ===")
    
    # Verify paths
    print("\n=== PATH VERIFICATION ===")
    print(f"JPEG directory exists: {os.path.exists(jpeg_dir)}")
    print(f"CSV directory exists: {os.path.exists(csv_dir)}")
    
    # Load and process data
    merged_df = load_data()
    print(f"\nLoaded {len(merged_df)} records")
    
    # Debug sample
    sample = merged_df.iloc[0]
    print("\n=== SAMPLE RECORD ===")
    print(f"Patient ID: {sample['patient_id_clean']}")
    print(f"DICOM path: {sample.get('image_path')}")
    print(f"Found JPEG: {find_jpeg_file(sample)}")
    
    # Process all images
    process_images(merged_df)
    print("\n=== COMPLETE ===")