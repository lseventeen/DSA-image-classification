"""
Model definition for medical image classification using MONAI.

Provides MONAI network architectures commonly used in medical imaging:
DenseNet, EfficientNet, and SE-ResNet.
"""

import torch
from monai.networks.nets import (
    DenseNet121,
    DenseNet169,
    DenseNet201,
    EfficientNetBN,
    SEResNet50,
)

# Models whose constructor uses ``out_channels`` for the number of classes.
_DENSENET_MODELS = {
    "densenet121": DenseNet121,
    "densenet169": DenseNet169,
    "densenet201": DenseNet201,
}

# Models whose constructor uses ``num_classes``.
_EFFICIENTNET_NAMES = {
    "efficientnet-b0",
    "efficientnet-b1",
    "efficientnet-b2",
    "efficientnet-b3",
    "efficientnet-b4",
    "efficientnet-b5",
    "efficientnet-b6",
    "efficientnet-b7",
}

SUPPORTED_MODELS = (
    list(_DENSENET_MODELS.keys())
    + sorted(_EFFICIENTNET_NAMES)
    + ["se_resnet50"]
)


def build_model(
    num_classes,
    model_name="densenet121",
    in_channels=1,
    pretrained=True,
    freeze_backbone=False,
    dropout_prob=0.2,
):
    """Build a MONAI classification model.

    Args:
        num_classes: Number of output classes.
        model_name: Architecture name (see ``SUPPORTED_MODELS``).
        in_channels: Number of input channels (1 for grayscale medical images).
        pretrained: Whether to load pre-trained weights from the MONAI
            model zoo (DenseNet / EfficientNet only).
        freeze_backbone: If ``True``, freeze all feature-extraction layers
            and only train the classifier head.
        dropout_prob: Dropout probability (used by DenseNet).

    Returns:
        A PyTorch ``nn.Module`` ready for training.
    """
    if model_name in _DENSENET_MODELS:
        model_cls = _DENSENET_MODELS[model_name]
        model = model_cls(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=num_classes,
            pretrained=pretrained,
            dropout_prob=dropout_prob,
        )
    elif model_name in _EFFICIENTNET_NAMES:
        model = EfficientNetBN(
            model_name,
            spatial_dims=2,
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained,
        )
    elif model_name == "se_resnet50":
        model = SEResNet50(
            spatial_dims=2,
            in_channels=in_channels,
            num_classes=num_classes,
        )
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Choose from {SUPPORTED_MODELS}"
        )

    if freeze_backbone:
        _freeze_backbone(model, model_name)

    return model


def _freeze_backbone(model, model_name):
    """Freeze all layers except the final classifier head."""
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the classifier head
    if model_name in _DENSENET_MODELS:
        for param in model.class_layers.parameters():
            param.requires_grad = True
    elif model_name in _EFFICIENTNET_NAMES:
        for param in model._fc.parameters():
            param.requires_grad = True
    elif model_name == "se_resnet50":
        for param in model.last_linear.parameters():
            param.requires_grad = True


def load_model(checkpoint_path, num_classes, model_name="densenet121",
               in_channels=1, device="cpu"):
    """Load a trained model from a checkpoint file."""
    model = build_model(
        num_classes,
        model_name=model_name,
        in_channels=in_channels,
        pretrained=False,
    )
    state_dict = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
