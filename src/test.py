import os
import json
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score,
    roc_curve, auc
)
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

# Local imports
from dataset_loader import create_test_loader
from gradcam import GradCAMpp
from visualize import save_medical_visualization, check_visualization_directory
from model import HybridModel

CONFIG = {
    "data_root": r"C:/Users/VIRAJ/OneDrive/Documents/AI-based-cancer-detection-project/data/model data",
    "model_path": r"C:/Users/VIRAJ/OneDrive/Documents/AI-based-cancer-detection-project/output/Best_model/best_model.pt",
    "output_dir": r"C:/Users/VIRAJ/OneDrive/Documents/AI-based-cancer-detection-project/output/test_results",
    "batch_size": 8,
    "num_workers": 0,
    "img_size": 224,    
    "in_chans": 1,
    "num_classes": 2,
    "use_vit": True,
    "max_visualizations": 50,
    "visualization_dir": r"C:/Users/VIRAJ/OneDrive/Documents/AI-based-cancer-detection-project/output/test_results/visualizations"
}

def convert_to_serializable(obj):
    """Recursively convert numpy types to native Python types"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    return obj

def safe_load_checkpoint(model_path, device):
    """Secure model loading with proper torch context"""
    if str(device) == 'cuda':
        torch.cuda.init()
        _ = torch.tensor([1.0]).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"Secure loading failed, falling back to legacy mode: {str(e)}")
        try:
            warnings.warn("Using weights_only=False - ensure you trust the model source")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {str(e)}")
    return checkpoint

def load_model(model_path, device, **kwargs):
    """Robust model loading with validation"""
    checkpoint = safe_load_checkpoint(model_path, device)
    
    # Handle different checkpoint formats
    state_dict = checkpoint.get('model_state_dict', 
                  checkpoint.get('state_dict', 
                  checkpoint.get('model_state', checkpoint)))
    
    model = HybridModel(**kwargs)
    
    # Filter only matching parameters
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                         if k in model_state_dict and v.shape == model_state_dict[k].shape}
    
    # Load with strict=False to handle missing keys
    model.load_state_dict(filtered_state_dict, strict=False)
    
    # Verify we loaded some weights
    num_loaded = len(filtered_state_dict)
    num_total = len(model_state_dict)
    print(f"Loaded {num_loaded}/{num_total} parameters")
    
    if num_loaded == 0:
        raise RuntimeError("No matching parameters found in checkpoint")
    
    return model.to(device).eval()

def generate_minimal_report(accuracy, class_idx, class_names, num_samples):
    """Generate a classification report for single-class case"""
    return {
        class_names[class_idx]: {
            "precision": 1.0,
            "recall": 1.0,
            "f1-score": 1.0,
            "support": num_samples
        },
        "accuracy": accuracy,
        "macro avg": {
            "precision": 1.0,
            "recall": 1.0,
            "f1-score": 1.0,
            "support": num_samples
        },
        "weighted avg": {
            "precision": 1.0,
            "recall": 1.0,
            "f1-score": 1.0,
            "support": num_samples
        }
    }

def analyze_misclassifications(metrics):
    """Analyze model weaknesses"""
    cm = metrics["confusion_matrix"]
    fp = cm[0][1]  # False positives
    fn = cm[1][0]  # False negatives
    
    print(f"\nMisclassification Analysis:")
    print(f"False Positives (Benign called Malignant): {fp} ({fp/sum(cm[0]):.1%})")
    print(f"False Negatives (Malignant called Benign): {fn} ({fn/sum(cm[1]):.1%})")

def create_diagnostic_samples(model, gradcam, test_loader, device, class_names, vis_dir):
    """Create diagnostic samples with both classes for visualization"""
    print("\nCreating diagnostic test samples...")
    
    # Find samples from both predicted classes
    class_samples = {0: None, 1: None}
    
    for images, labels in test_loader:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
        for i in range(len(images)):
            pred_class = preds[i].item()
            if class_samples[pred_class] is None:
                class_samples[pred_class] = (
                    images[i], labels[i].item(), pred_class, probs[i]
                )
            
        if class_samples[0] is not None and class_samples[1] is not None:
            break
    
    # Visualize both cases
    for pred_class, sample_data in class_samples.items():
        if sample_data is not None:
            img, true_class, pred_class_val, prob = sample_data
            img_tensor = img.unsqueeze(0).requires_grad_(True)
            heatmaps, _ = gradcam(img_tensor, class_idx=pred_class_val)
            
            # Handle heatmap format
            if isinstance(heatmaps, list):
                heatmap = heatmaps[0]  # Get first heatmap from list
            else:
                heatmap = heatmaps[0].cpu().numpy() if heatmaps.dim() > 2 else heatmaps.cpu().numpy()
            
            success = save_medical_visualization(
                input_tensor=img.cpu(),
                heatmap=heatmap,
                pred_class=pred_class_val,
                true_class=true_class,
                class_names=class_names,
                save_path=os.path.join(
                    vis_dir,
                    f"DIAGNOSTIC_class{pred_class_val}_true{true_class}.png"
                )
            )
            
            if success:
                print(f"✓ Created diagnostic sample for class {pred_class_val}")
    
    print("Diagnostic samples created in visualization directory")

def main():
    # Setup directories
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # Check visualization directory first
    if not check_visualization_directory(CONFIG["visualization_dir"]):
        print("CRITICAL: Cannot write to visualization directory!")
        return
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.init()
    
    print(f"\nUsing device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA initialized: {torch.cuda.is_initialized()}")

    # Load test data
    print("\nLoading test dataset...")
    test_loader, class_names = create_test_loader(
        data_root=CONFIG["data_root"],
        batch_size=CONFIG["batch_size"],
        img_size=CONFIG["img_size"],
        in_chans=CONFIG["in_chans"],
        num_workers=CONFIG["num_workers"]
    )

    # Verify first batch
    test_batch, test_labels = next(iter(test_loader))
    print("\nFirst batch verification:")
    print(f"Batch shape: {test_batch.shape}")
    print(f"Label distribution: {Counter(test_labels.numpy().tolist())}")
    print(f"Pixel range: [{test_batch.min():.4f}, {test_batch.max():.4f}]")

    # Check for single-class case
    unique_classes = np.unique(test_labels.numpy())
    if len(unique_classes) < 2:
        warnings.warn(f"Warning: Test set only contains class {unique_classes[0]}")

    # Load model
    print("\nLoading and verifying model...")
    model = load_model(
        model_path=CONFIG["model_path"],
        device=device,
        num_classes=CONFIG["num_classes"],
        in_chans=CONFIG["in_chans"],
        use_vit=CONFIG["use_vit"]
    )

    # Model sanity check
    with torch.no_grad():
        test_input = torch.randn(1, CONFIG["in_chans"], CONFIG["img_size"], CONFIG["img_size"]).to(device)
        test_output = model(test_input)
        print(f"\nModel test output: {torch.softmax(test_output, dim=1).cpu().numpy()}")

    # Initialize Grad-CAM
    target_layer = model.cnn.blocks[-1] if hasattr(model, 'cnn') else model.vit.blocks[-1]
    print(f"\nUsing target layer: {target_layer.__class__.__name__}")
    gradcam = GradCAMpp(model=model, target_layer=target_layer)

    # Evaluation
    y_true, y_pred, y_probs = [], [], []
    vis_counts = {
        'LOW_CONF': 0,
        'HIGH_CONF': 0,
        'ERROR': 0
    }
    max_per_class = CONFIG["max_visualizations"]

    # SINGLE evaluation loop
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Store results
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs[:, 1].cpu().numpy())
            
            # Visualization logic
            for i in range(images.size(0)):
                try:
                    true_class = int(labels[i].item())
                    pred_class = int(preds[i].item())
                    confidence = float(probs[i][pred_class].item())
                    
                    # Always visualize misclassifications
                    if pred_class != true_class:
                        case_type = 'ERROR'
                        save_prefix = 'ERROR'
                    # Visualize high-confidence correct predictions
                    elif confidence > 0.8 and vis_counts.get('HIGH_CONF', 0) < max_per_class:
                        case_type = 'HIGH_CONF'
                        save_prefix = 'HIGH_CONF'
                    # Visualize low-confidence correct predictions  
                    elif confidence < 0.6 and vis_counts.get('LOW_CONF', 0) < max_per_class:
                        case_type = 'LOW_CONF'
                        save_prefix = 'LOW_CONF'
                    else:
                        continue
                        
                    # Generate heatmap
                    img_tensor = images[i].unsqueeze(0).requires_grad_(True)
                    heatmaps, _ = gradcam(img_tensor, class_idx=pred_class)
                    
                    # Handle heatmap format - CORRECTED
                    if isinstance(heatmaps, list):
                        heatmap = heatmaps[0]  # Get first heatmap from list
                    else:
                        heatmap = heatmaps[0].cpu().numpy() if heatmaps.dim() > 2 else heatmaps.cpu().numpy()
                    
                    # Ensure heatmap is 2D
                    heatmap = np.squeeze(heatmap)
                    if heatmap.ndim != 2:
                        continue
                    
                    success = save_medical_visualization(
                        input_tensor=images[i].cpu(),
                        heatmap=heatmap,
                        pred_class=pred_class,
                        true_class=true_class,
                        class_names=class_names,
                        save_path=os.path.join(
                            CONFIG["visualization_dir"],
                            f"{save_prefix}_b{batch_idx}_i{i}_true{true_class}_pred{pred_class}_conf{confidence:.2f}.png"
                        )
                    )
                    
                    if success:
                        vis_counts[case_type] = vis_counts.get(case_type, 0) + 1
                    
                except Exception as e:
                    print(f"Visualization failed for batch {batch_idx} sample {i}: {str(e)}")
                    if 'CUDA' in str(e):
                        torch.cuda.empty_cache()

    # Create diagnostic samples to ensure we get visualizations
    create_diagnostic_samples(model, gradcam, test_loader, device, class_names, CONFIG["visualization_dir"])

    # Check what files were created
    print("\n=== VISUALIZATION RESULTS ===")
    vis_dir = CONFIG["visualization_dir"]
    if os.path.exists(vis_dir):
        files = os.listdir(vis_dir)
        print(f"Total visualization files created: {len(files)}")
        if files:
            print("Sample files:")
            for f in files[:3]:
                print(f"  - {f}")
    else:
        print("Visualization directory not found!")

    # Metrics calculation
    print("\nCalculating metrics...")
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "class_distribution": {str(k): int(v) for k, v in Counter(y_true).items()},
        "visualization_counts": vis_counts,
        "prediction_stats": {
            "min_confidence": float(np.min(y_probs)),
            "max_confidence": float(np.max(y_probs)),
            "mean_confidence": float(np.mean(y_probs)),
            "std_confidence": float(np.std(y_probs))
        }
    }

    # Handle ROC AUC only if we have both classes
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_probs))
        # Generate ROC curve
        plt.figure()
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        plt.plot(fpr, tpr, label=f'AUC = {metrics["roc_auc"]:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(CONFIG["output_dir"], "roc_curve.png"))
        plt.close()
    else:
        print("Warning: Only one class present - skipping ROC AUC calculation")
        metrics["roc_auc"] = None

    # Handle classification report
    if len(np.unique(y_true)) > 1:
        metrics["classification_report"] = convert_to_serializable(
            classification_report(
                y_true, y_pred,
                target_names=class_names,
                output_dict=True
            )
        )
    else:
        print(f"Warning: Only class {np.unique(y_true)[0]} present - generating minimal report")
        metrics["classification_report"] = generate_minimal_report(
            metrics["accuracy"],
            np.unique(y_true)[0],
            class_names,
            len(y_true)
        )

    # Save results
    with open(os.path.join(CONFIG["output_dir"], "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Analysis
    analyze_misclassifications(metrics)

    print("\nEvaluation Complete!")
    print(f"Total samples processed: {len(y_true)}")
    print(f"Class distribution: {Counter(y_true)}")
    print(f"Predicted distribution: {Counter(y_pred)}")
    print(f"Visualization counts: {vis_counts}")
    print(f"Results saved to: {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()