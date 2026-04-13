"""
Model definition for X-ray image classification.

Uses transfer learning with pre-trained ImageNet models,
replacing the final classifier to match the number of classes.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes, model_name="resnet18", pretrained=True,
                freeze_backbone=False):
    """
    Build a classification model based on a pre-trained backbone.

    Args:
        num_classes: Number of output classes.
        model_name: Backbone architecture ('resnet18', 'resnet34', 'resnet50').
        pretrained: Whether to use ImageNet pre-trained weights.
        freeze_backbone: If True, freeze all backbone parameters.

    Returns:
        A PyTorch nn.Module ready for training.
    """
    weights_map = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
        "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
    }

    if model_name not in weights_map:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Choose from {list(weights_map.keys())}"
        )

    factory, weights = weights_map[model_name]
    model = factory(weights=weights if pretrained else None)

    # Optionally freeze backbone
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the fully-connected head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )

    return model


def load_model(checkpoint_path, num_classes, model_name="resnet18", device="cpu"):
    """Load a trained model from a checkpoint file."""
    model = build_model(num_classes, model_name=model_name, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
