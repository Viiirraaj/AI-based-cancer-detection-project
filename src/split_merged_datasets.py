import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# Set random seed for reproducibility
random.seed(42)

# ========== CONFIGURATION (PRESERVED YOUR PATHS) ==========
BASE_DIR = Path("C:/Users/VIRAJ/OneDrive/Documents/AI-based-cancer-detection-project/data/processed/merged dataset")
OUTPUT_DIR = Path("C:/Users/VIRAJ/OneDrive/Documents/AI-based-cancer-detection-project/data/model data")

# Splits (70% train, 15% val, 15% test)
SPLIT_RATIOS = {'train': 0.7, 'val': 0.15, 'test': 0.15}
CLASSES = ['benign', 'malignant']
# =========================================================

def create_balanced_splits():
    """Create balanced dataset splits while preserving original paths"""
    # 1. Prepare output directories
    for split in SPLIT_RATIOS.keys():
        for label in CLASSES:
            (OUTPUT_DIR / split / label).mkdir(parents=True, exist_ok=True)

    # 2. Collect and count images
    class_counts = {}
    image_paths = {}
    
    for label in CLASSES:
        label_dir = BASE_DIR / label
        images = list(label_dir.glob("*.png"))
        
        if not images:
            raise FileNotFoundError(f"❌ No PNG images found in {label_dir}")
        
        random.shuffle(images)
        class_counts[label] = len(images)
        image_paths[label] = images

    # 3. Determine smallest class size for balancing
    min_count = min(class_counts.values())
    print(f"\n⚖️ Balancing classes to {min_count} samples each")
    
    # 4. Create balanced splits
    for label in CLASSES:
        # Take random subset if needed
        images = random.sample(image_paths[label], min_count) if class_counts[label] > min_count else image_paths[label]
        
        # Calculate split indices
        train_end = int(SPLIT_RATIOS['train'] * min_count)
        val_end = train_end + int(SPLIT_RATIOS['val'] * min_count)
        
        # Split the data
        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }
        
        # Copy files with progress bar
        for split_name, img_list in splits.items():
            print(f"\n📁 Copying {label} {split_name} set ({len(img_list)} images)...")
            for img_path in tqdm(img_list, desc=label):
                dest = OUTPUT_DIR / split_name / label / img_path.name
                shutil.copy(img_path, dest)
        
        print(f"✅ {label}: {len(images)} → Train:{len(splits['train'])} Val:{len(splits['val'])} Test:{len(splits['test'])}")

if __name__ == "__main__":
    print(f"🔍 Source directory: {BASE_DIR}")
    print(f"🎯 Target directory: {OUTPUT_DIR}")
    
    try:
        create_balanced_splits()
        print("\n🎉 Successfully created balanced dataset splits!")
        print(f"   Train: {SPLIT_RATIOS['train']*100}%")
        print(f"   Val:   {SPLIT_RATIOS['val']*100}%")
        print(f"   Test:  {SPLIT_RATIOS['test']*100}%")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please verify:")
        print(f"1. {BASE_DIR} exists and contains 'benign/' and 'malignant/' folders")
        print(f"2. Both class folders contain PNG images")