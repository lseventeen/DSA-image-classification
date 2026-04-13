"""
Data augmentation and transformation pipelines using MONAI for medical images.

Uses MONAI dictionary-based transforms for a robust medical imaging pipeline
that handles various image formats (TIFF, PNG, JPEG) and bit depths natively.
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


def get_train_transforms():
    """
    Training transforms with medical-image-specific augmentations.

    Since the dataset is small (~500 images), aggressive augmentation
    helps prevent overfitting.

    Resize strategy: downsample the long edge to ``IMG_SIZE`` while
    maintaining the aspect ratio, then zero-pad to a square.
    """
    return mt.Compose([
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
        # --- augmentation ---
        mt.RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
        mt.RandFlipd(keys=["image"], prob=0.3, spatial_axis=0),
        mt.RandRotated(keys=["image"], range_x=0.26, prob=0.5,
                       padding_mode="zeros"),
        mt.RandZoomd(keys=["image"], min_zoom=0.9, max_zoom=1.1, prob=0.5,
                     padding_mode="constant"),
        mt.RandGaussianNoised(keys=["image"], prob=0.3,
                              mean=0.0, std=0.05),
        mt.RandAdjustContrastd(keys=["image"], prob=0.3,
                               gamma=(0.8, 1.2)),
        mt.EnsureTyped(keys=["image", "label"]),
    ])


def get_val_transforms():
    """Test transforms — deterministic, no augmentation.

    Resize strategy: downsample the long edge to ``IMG_SIZE`` while
    maintaining the aspect ratio, then zero-pad to a square.
    """
    return mt.Compose([
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
        mt.EnsureTyped(keys=["image", "label"]),
    ])
