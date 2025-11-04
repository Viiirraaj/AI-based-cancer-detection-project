import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from typing import Union
from skimage import measure

def save_medical_visualization(
    input_tensor: Union[torch.Tensor, np.ndarray],
    heatmap: Union[np.ndarray, torch.Tensor],
    pred_class: int,
    true_class: int,
    class_names: list,
    save_path: str,
    dpi: int = 150,
    show_contours: bool = True
):
    """Enhanced medical visualization with clinical annotations"""
    try:
        # Input validation
        if len(class_names) != 2:
            raise ValueError("class_names must contain exactly 2 elements")
        
        if not isinstance(pred_class, int) or not isinstance(true_class, int):
            raise TypeError("Class indices must be integers")
        
        # Convert and normalize input
        if isinstance(input_tensor, torch.Tensor):
            img = input_tensor.detach().cpu().numpy()
        else:
            img = np.array(input_tensor)
        
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.detach().cpu().numpy()
        
        # Ensure proper dimensionality
        img = np.squeeze(img)
        heatmap = np.squeeze(heatmap)
        
        if img.ndim not in (2, 3):
            raise ValueError("Input image must be 2D or 3D")
        
        if heatmap.ndim != 2:
            raise ValueError("Heatmap must be 2D")
        
        # Normalize
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        plt.subplots_adjust(wspace=0.05)
        
        # Original image
        ax1.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax1.set_title(
            f"True: {class_names[true_class]}",
            fontsize=12,
            color='green' if true_class == 0 else 'red',
            pad=10
        )
        ax1.axis('off')
        
        # Heatmap overlay
        ax2.imshow(img, cmap='gray', vmin=0, vmax=1)
        hmap = ax2.imshow(heatmap, alpha=0.5, cmap='jet', vmin=0, vmax=1)
        
        # Add contours
        if show_contours:
            try:
                contours = measure.find_contours(heatmap, 0.5)
                for contour in contours:
                    ax2.plot(contour[:, 1], contour[:, 0], 
                            linewidth=1.5, color='cyan', linestyle='--')
            except Exception as e:
                print(f"Contour generation failed: {str(e)}")
        
        ax2.set_title(
            f"Pred: {class_names[pred_class]}",
            fontsize=12,
            color='green' if pred_class == true_class else 'red',
            pad=10
        )
        ax2.axis('off')
        
        # Add colorbar
        cbar = fig.colorbar(hmap, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Intensity', rotation=270, labelpad=15)
        
        # Add diagnostic markers
        if pred_class != true_class:
            fig.text(
                0.5, 0.02,
                f"ALERT: MISCLASSIFIED (True: {class_names[true_class]}, Pred: {class_names[pred_class]})",
                ha='center', va='center',
                bbox=dict(facecolor='red', alpha=0.3, edgecolor='none'),
                fontsize=10
            )
        
        # Add prediction confidence
        fig.text(
            0.5, 0.95,
            f"Diagnosis: {'✓ CORRECT' if pred_class == true_class else '✗ INCORRECT'}",
            ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
            fontsize=12
        )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save figure
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi, pad_inches=0.1)
        plt.close(fig)
        
        # Verify file was created
        if not os.path.exists(save_path):
            raise RuntimeError(f"Failed to create file: {save_path}")
            
        return True
        
    except Exception as e:
        print(f"✗ Visualization failed for {save_path}: {str(e)}")
        plt.close('all')
        return False

def check_visualization_directory(vis_dir):
    """Check if visualization directory is accessible"""
    print(f"\nChecking visualization directory: {vis_dir}")
    
    if not os.path.exists(vis_dir):
        print("Directory doesn't exist! Creating it...")
        os.makedirs(vis_dir, exist_ok=True)
    
    # Check write permissions
    test_file = os.path.join(vis_dir, "test_write.txt")
    try:
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print("✓ Directory is writable")
        return True
    except Exception as e:
        print(f"✗ Cannot write to directory: {e}")
        return False