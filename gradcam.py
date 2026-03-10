"""Grad-CAM for defect region localization in abnormal images."""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path


class GradCAM:
    """Grad-CAM for visualizing defect regions."""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            g = grad_output[0] if isinstance(grad_output, tuple) else grad_output
            self.gradients = g.detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap."""
        self.model.eval()
        input_tensor.requires_grad_(True)
        
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        output[0, target_class].backward()
        
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3))
        # Weighted combination of activation maps
        cam = (weights[:, :, None, None] * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam, output
    
    def overlay_heatmap(self, cam, original_img, alpha=0.5):
        """Overlay heatmap on original image."""
        if isinstance(original_img, Image.Image):
            img = np.array(original_img.resize((cam.shape[1], cam.shape[0])))
        else:
            img = original_img
        
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        
        cam_resized = np.uint8(255 * cam)
        heatmap = plt.cm.jet(cam_resized)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        
        overlay = (alpha * heatmap + (1 - alpha) * img).astype(np.uint8)
        return overlay, heatmap


def get_defect_regions(cam: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Get binary mask of suspicious defect regions.
    threshold: values above this are considered defect regions (0-1).
    """
    return (cam > threshold).astype(np.uint8)


def draw_defect_boxes(cam: np.ndarray, threshold: float = 0.5, min_area: int = 100):
    """
    Get bounding boxes around defect regions from CAM.
    Returns list of (x1, y1, x2, y2) boxes.
    """
    try:
        import cv2
    except ImportError:
        return []  # Fallback if opencv not installed
    mask = get_defect_regions(cam, threshold)
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, x + w, y + h))
    
    return boxes
