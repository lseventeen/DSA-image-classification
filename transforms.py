"""
Data augmentation and transformation pipelines using MONAI for medical images.

Uses MONAI dictionary-based transforms for a robust medical imaging pipeline
that handles various image formats (TIFF, PNG, JPEG) and bit depths natively.

Augmentation strategy inspired by nnU-Net:
- Elastic deformation
- Rotation and scaling
- Gaussian noise and blur
- Brightness / contrast / gamma adjustments
- Mirroring on all spatial axes
- Simulation of low resolution
- Coarse dropout (similar to CutOut)
"""

from monai import transforms as mt
from monai.data.image_reader import PILReader

import config


class EnsureSingleChanneld(mt.MapTransform):
    """Convert multi-channel images to single channel (grayscale) by averaging.

    Medical images (e.g. X-ray) are inherently grayscale.  Some files may be
    stored as RGB; this transform collapses them to a single channel so that
    the model receives consistent ``(1, H, W)`` input.
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            if img.shape[0] > 1:
                d[key] = img.float().mean(0, keepdim=True)
        return d


def _base_transforms():
    """Shared deterministic pre-processing transforms (loading, resizing)."""
    return [
        mt.LoadImaged(keys=["image"], image_only=True,
                      reader=PILReader()),
        mt.EnsureChannelFirstd(keys=["image"]),
        EnsureSingleChanneld(keys=["image"]),
        mt.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        mt.Resized(keys=["image"], spatial_size=config.IMG_SIZE,
                   size_mode="longest"),
        mt.SpatialPadd(keys=["image"],
                       spatial_size=(config.IMG_SIZE, config.IMG_SIZE),
                       mode="constant"),
    ]


def get_train_transforms():
    """
    Training transforms with nnU-Net-inspired augmentations.

    Since the dataset is small (~500 images), aggressive augmentation
    helps prevent overfitting.  The augmentation pipeline follows nnU-Net
    best practices for medical image segmentation / classification:

    1. Spatial: elastic deformation, rotation, scaling, mirroring
    2. Intensity: Gaussian noise, Gaussian blur, brightness, contrast, gamma
    3. Regularization: coarse dropout (CutOut-like)
    """
    base = _base_transforms()
    augmentations = [
        # --- Spatial augmentations ---
        mt.RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
        mt.RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        mt.RandRotated(keys=["image"], range_x=0.52, prob=0.5,
                       padding_mode="zeros"),
        mt.RandZoomd(keys=["image"], min_zoom=0.7, max_zoom=1.4, prob=0.3,
                     padding_mode="constant"),
        mt.RandAffined(
            keys=["image"], prob=0.3,
            rotate_range=(0.26,),
            shear_range=(0.1, 0.1),
            scale_range=(0.1, 0.1),
            padding_mode="zeros",
        ),
        # --- Intensity augmentations (nnU-Net style) ---
        mt.RandGaussianNoised(keys=["image"], prob=0.2,
                              mean=0.0, std=0.1),
        mt.RandGaussianSmoothd(
            keys=["image"], prob=0.2,
            sigma_x=(0.5, 1.15),
            sigma_y=(0.5, 1.15),
        ),
        mt.RandAdjustContrastd(keys=["image"], prob=0.3,
                               gamma=(0.7, 1.5)),
        mt.RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.3),
        mt.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
        # --- Regularisation: coarse dropout (CutOut) ---
        mt.RandCoarseDropoutd(
            keys=["image"], holes=6,
            spatial_size=(16, 16),
            fill_value=0, prob=0.2,
        ),
        mt.EnsureTyped(keys=["image", "label"]),
    ]
    return mt.Compose(base + augmentations)


def get_val_transforms():
    """Test transforms — deterministic, no augmentation."""
    base = _base_transforms()
    return mt.Compose(base + [mt.EnsureTyped(keys=["image", "label"])])
