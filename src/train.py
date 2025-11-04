import os
import json
import time
from datetime import datetime
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
# Add these imports
from sklearn.metrics import classification_report, confusion_matrix
from visualize import visualize_gradcam, generate_sample_visualizations  # If using separate file
# Import model and dataset components
from model import HybridModel
from dataset_loader import create_loaders

# Configuration
CONFIG = {
    "SAVE_DIR": "C:/Users/VIRAJ/OneDrive/Documents/AI-based-cancer-detection-project/output",
    "MODEL_SAVE": "best_model.pt",
    "LOG_FILE": "training.log",
    "METRICS_FILE": "training_metrics.json",
    "CAM_DIR": "gradcam",
    "DATA_ROOT": "C:/Users/VIRAJ/OneDrive/Documents/AI-based-cancer-detection-project/data/model data",
    "IMG_SIZE": 224,
    "IN_CHANS": 1,
    "BATCH_SIZE": 16,
    "NUM_WORKERS": 4,
    "EPOCHS": 69,
    "LR": 1e-4,
    "WEIGHT_DECAY": 1e-5,
    "AMP": True,
    "WEIGHTED_SAMPLING": True,
    "AUGMENT": True
}

def compute_metrics(outputs, labels):
    """Calculate accuracy and other metrics"""
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy, preds.cpu().numpy(), labels.cpu().numpy()

def validate(model, val_loader, device, criterion):
    """Enhanced validation with metrics"""
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            val_loss += loss.item()
            
            _, preds, labels = compute_metrics(outputs, y)
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    val_loss /= len(val_loader)
    return val_loss, val_acc

def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch):
    """Enhanced training epoch with metrics"""
    model.train()
    epoch_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['EPOCHS']}")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        # Forward pass with AMP
        with torch.amp.autocast(device_type=device.type, enabled=CONFIG["AMP"]):
            outputs = model(x)
            loss = criterion(outputs, y)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # Calculate metrics
        acc, preds, labels = compute_metrics(outputs, y)
        epoch_loss += loss.item()
        all_preds.extend(preds)
        all_labels.extend(labels)
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{acc:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.1e}"
        })
    
    train_loss = epoch_loss / len(train_loader)
    train_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return train_loss, train_acc

def log_metrics(epoch, train_loss, train_acc, val_loss, val_acc, lr, best_acc):
    """Enhanced logging with all metrics"""
    metrics = {
        "epoch": epoch,
        "time": datetime.now().isoformat(),
        "train_loss": round(train_loss, 4),
        "train_acc": round(train_acc, 4),
        "val_loss": round(val_loss, 4),
        "val_acc": round(val_acc, 4),
        "lr": round(lr, 8),
        "best_acc": round(best_acc, 4)
    }
    
    # Save to JSON file
    with open(os.path.join(CONFIG["SAVE_DIR"], CONFIG["METRICS_FILE"]), "a") as f:
        json.dump(metrics, f)
        f.write("\n")
    
    # Print to console
    print(f"\nEpoch {epoch}/{CONFIG['EPOCHS']}:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
    print(f"  Best Val Acc: {best_acc:.4f} | LR: {lr:.2e}")

def initialize_system():
    """Initialize all components with proper device handling"""
    # Setup directories
    os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)
    os.makedirs(os.path.join(CONFIG["SAVE_DIR"], CONFIG["CAM_DIR"]), exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_weights, class_names = create_loaders(
        data_root=CONFIG["DATA_ROOT"],
        batch_size=CONFIG["BATCH_SIZE"],
        img_size=CONFIG["IMG_SIZE"],
        in_chans=CONFIG["IN_CHANS"],
        num_workers=CONFIG["NUM_WORKERS"],
        weighted_sampling=CONFIG["WEIGHTED_SAMPLING"],
        augment=CONFIG["AUGMENT"]
    )
    
    # Initialize model and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridModel(
        num_classes=len(class_names),
        in_chans=CONFIG["IN_CHANS"]
    ).to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["LR"],
        weight_decay=CONFIG["WEIGHT_DECAY"]
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["EPOCHS"] * len(train_loader)
    )
    
    return {
        "device": device,
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_names": class_names
    }

def main():
    """Main training loop with comprehensive metrics"""
    # Initialize system
    system = initialize_system()
    device = system["device"]
    model = system["model"]
    criterion = system["criterion"]
    optimizer = system["optimizer"]
    scheduler = system["scheduler"]
    train_loader = system["train_loader"]
    val_loader = system["val_loader"]
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler(enabled=CONFIG["AMP"])
    
    best_acc = 0.0
    for epoch in range(1, CONFIG["EPOCHS"] + 1):
        # Training phase
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        
        # Validation phase
        val_loss, val_acc = validate(model, val_loader, device, criterion)
        
        # Update best accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "config": CONFIG
            }, os.path.join(CONFIG["SAVE_DIR"], CONFIG["MODEL_SAVE"]))
        
        # Log metrics
        log_metrics(
            epoch, train_loss, train_acc, val_loss, val_acc,
            optimizer.param_groups[0]["lr"], best_acc
        )
    # After training completes, add this:
    print("\nTraining completed. Running final evaluation...")
    
    # Load best model
    checkpoint = torch.load(os.path.join(CONFIG["SAVE_DIR"], CONFIG["MODEL_SAVE"]))
    model.load_state_dict(checkpoint['model_state'])
    
    # Test set evaluation
    test_loss, test_acc = validate(model, system["test_loader"], device, criterion)
    print(f"\nFinal Test Performance:")
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

        # Performance analysis
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in system["test_loader"]:
            outputs = model(x.to(device))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=system["class_names"]))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    # Save metrics
    metrics = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'classification_report': classification_report(
            all_labels, all_preds, 
            target_names=system["class_names"],
            output_dict=True
        ),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
    }
    with open(os.path.join(CONFIG["SAVE_DIR"], 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()