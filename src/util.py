import json
import os
from datetime import datetime
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import device

def setup_output_dir(config):
    """Create all necessary output directories"""
    os.makedirs(config["SAVE_DIR"], exist_ok=True)
    os.makedirs(os.path.join(config["SAVE_DIR"], config["CAM_DIR"]), exist_ok=True)
    
    # Save config for reproducibility
    with open(os.path.join(config["SAVE_DIR"], "config.json"), "w") as f:
        json.dump(config, f, indent=2)

def save_gradcam_samples(model, loader, class_names, save_dir):
    """Save sample Grad-CAM visualizations"""
    cam = GradCAM(model, model.cnn.blocks[-1])
    os.makedirs(save_dir, exist_ok=True)

    for i, (x, y) in enumerate(loader):
        if i >= 10:  # Save first 10 samples
            break
        path = os.path.join(
            save_dir,
            f"{class_names[y[0].item()]}_sample_{i}.png"
        )
        _ = cam(x.to(device), save_path=path)