import json
import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from collections import Counter
from PIL import Image

def create_loaders(
    data_root,
    batch_size,
    img_size,
    in_chans=1,
    num_workers=4,
    weighted_sampling=True,
    augment=True,
    output_dir="C:/Users/VIRAJ/OneDrive/Documents/AI-based-cancer-detection-project/output"
):
    """Enhanced data loader with better validation"""
    os.makedirs(output_dir, exist_ok=True)

    # Base transforms with device-aware normalization
    base_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=in_chans),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Augmentation pipeline
    train_transform = transforms.Compose([
        transforms.Resize((img_size + 20, img_size + 20)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.08, contrast=0.08),
        transforms.Grayscale(num_output_channels=in_chans),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]) if augment else base_transform

    # Dataset validation
    def validate_dataset(path, name):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{name} directory not found: {path}")
        if not os.listdir(path):
            raise RuntimeError(f"{name} directory is empty: {path}")

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    validate_dataset(train_dir, "Training")
    if not os.path.isdir(val_dir):
        print(f"Warning: Validation directory not found at {val_dir}")
    if not os.path.isdir(test_dir):
        print(f"Warning: Test directory not found at {test_dir}")

    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, train_transform)
    val_dataset = datasets.ImageFolder(val_dir, base_transform) if os.path.isdir(val_dir) else None
    test_dataset = datasets.ImageFolder(test_dir, base_transform) if os.path.isdir(test_dir) else None

    # Class balancing
    class_names = train_dataset.classes
    labels = [label for _, label in train_dataset.samples]
    counts = Counter(labels)
    
    if weighted_sampling:
        class_weights = torch.tensor([1.0 / max(counts[i], 1) for i in range(len(class_names))], 
                                   dtype=torch.float32)
        sampler = WeightedRandomSampler(
            weights=[class_weights[label] for _, label in train_dataset.samples],
            num_samples=len(train_dataset),
            replacement=True
        )
    else:
        sampler = None

    # DataLoader configuration
    loader_args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
        'persistent_workers': num_workers > 0
    }

    train_loader = DataLoader(train_dataset, sampler=sampler, 
                             shuffle=sampler is None, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args) if val_dataset else None
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args) if test_dataset else None

    # Save dataset info
    dataset_info = {
        'class_names': class_names,
        'class_distribution': dict(counts),
        'input_shape': (in_chans, img_size, img_size)
    }
    with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f, indent=2)

    return train_loader, val_loader, test_loader, class_names

def create_test_loader(data_root, batch_size, img_size, in_chans=1, num_workers=4):
    """Robust test-only loader"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=in_chans),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_dir = os.path.join(data_root, "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    test_dataset = datasets.ImageFolder(test_dir, transform)
    if not test_dataset.samples:
        raise RuntimeError(f"No test images found in {test_dir}")

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    ), test_dataset.classes