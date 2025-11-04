import torch
import numpy as np
import cv2
from typing import Union, List, Tuple

class GradCAMpp:
    """
    Robust Grad-CAM++ implementation with:
    - Proper type handling
    - Batch support
    - Error checking
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks with proper tensor handling
        def forward_hook(module, inp, out):
            self.activations = out  # Keep as tensor

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]  # Keep as tensor

        # Safe hook registration
        try:
            self.target_layer.register_forward_hook(forward_hook)
            if hasattr(self.target_layer, 'register_full_backward_hook'):
                self.target_layer.register_full_backward_hook(backward_hook)
            else:
                self.target_layer.register_backward_hook(backward_hook)
        except Exception as e:
            raise RuntimeError(f"Hook registration failed: {str(e)}")

    def __call__(self, input_tensor: torch.Tensor, class_idx: Union[int, torch.Tensor, None] = None) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Args:
            input_tensor: Input image tensor (B,C,H,W)
            class_idx: Optional class index (int or tensor)
        
        Returns:
            Tuple of (heatmaps, class_indices) as numpy arrays
        """
        # Input validation
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError(f"input_tensor must be torch.Tensor, got {type(input_tensor)}")
        
        # Store original model state
        original_training = self.model.training
        self.model.eval()
        
        # Forward pass with gradients
        with torch.enable_grad():
            output = self.model(input_tensor)
            
            # Handle class_idx
            if class_idx is None:
                class_idx = torch.argmax(output, dim=1)
            elif isinstance(class_idx, int):
                class_idx = torch.tensor([class_idx], device=input_tensor.device)
            elif not isinstance(class_idx, torch.Tensor):
                raise TypeError(f"class_idx must be int or Tensor, got {type(class_idx)}")
            
            # Ensure proper shape (B,)
            if class_idx.dim() == 0:
                class_idx = class_idx.unsqueeze(0)
            
            # Create one-hot target
            one_hot = torch.zeros_like(output)
            one_hot.scatter_(1, class_idx.unsqueeze(1), 1.0)
            
            # Backward pass
            self.model.zero_grad()
            output.backward(gradient=one_hot, retain_graph=True)

        # Verify we captured gradients
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients or activations")
        
        # Convert to numpy safely
        with torch.no_grad():
            grads = self.gradients.cpu().numpy()  # (B, C, H, W)
            acts = self.activations.cpu().numpy()  # (B, C, H, W)
            class_idx = class_idx.cpu().numpy()  # (B,)

        # Compute heatmaps
        batch_heatmaps = []
        weights = np.mean(grads, axis=(2, 3))  # (B, C)
        
        for i in range(grads.shape[0]):
            cam = np.sum(weights[i][:, None, None] * acts[i], axis=0)
            cam = np.maximum(cam, 0)  # ReLU
            cam = cam / (cam.max() + 1e-10)  # Normalize
            cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
            batch_heatmaps.append(cam)
        
        # Restore model state
        self.model.train(original_training)
        
        return batch_heatmaps, class_idx