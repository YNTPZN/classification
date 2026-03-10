"""Defect classifier with transfer learning."""

import torch
import torch.nn as nn
from torchvision import models


def build_classifier(model_name: str = "efficientnet_b0", num_classes: int = 2):
    """Build classifier with pretrained backbone."""
    if model_name == "efficientnet_b0":
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "resnet50":
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return backbone


def get_feature_layer(model, model_name: str):
    """Get the last conv layer for Grad-CAM."""
    if model_name == "efficientnet_b0":
        return model.features[-1]  # Last block
    elif model_name == "resnet50":
        return model.layer4
    raise ValueError(f"Unknown model: {model_name}")
