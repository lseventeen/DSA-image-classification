"""
Custom dataset for loading X-ray TIFF images from folder structure.

Expected directory layout:
    data/
    ├── class_1/
    │   ├── image001.tif
    │   ├── image002.tiff
    │   └── ...
    ├── class_2/
    │   └── ...
    └── ...
"""

import os
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

import config
from transforms import get_train_transforms, get_val_transforms


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


class XRayDataset(Dataset):
    """Dataset for X-ray images stored in class-based folder structure."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Path to the data directory containing class folders.
            transform: Optional torchvision transforms to apply.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []   # List of (image_path, label_index)
        self.classes = sorted(
            [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        )
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            for fname in sorted(os.listdir(cls_dir)):
                ext = Path(fname).suffix.lower()
                if ext in SUPPORTED_EXTENSIONS:
                    self.samples.append(
                        (cls_dir / fname, self.class_to_idx[cls_name])
                    )

        if not self.samples:
            raise RuntimeError(
                f"No images found in {root_dir}. "
                f"Supported formats: {SUPPORTED_EXTENSIONS}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = self._load_image(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label

    @staticmethod
    def _load_image(path):
        """
        Load a TIFF or standard image and convert to RGB PIL Image.

        Handles various TIFF bit-depths (8-bit, 16-bit, float) by
        normalizing to 0-255 uint8 range before converting to RGB.
        """
        path = str(path)
        ext = Path(path).suffix.lower()

        if ext in {".tif", ".tiff"}:
            img_array = tifffile.imread(path)

            # Handle multi-page TIFF — take first frame
            if img_array.ndim == 3 and img_array.shape[0] > 1:
                # Could be (pages, H, W) or (H, W, C)
                if img_array.shape[2] in (3, 4):
                    pass  # (H, W, C) — keep as is
                else:
                    img_array = img_array[0]  # take first page

            # Normalize to uint8
            img_array = _normalize_to_uint8(img_array)

            image = Image.fromarray(img_array)
        else:
            image = Image.open(path)

        # Ensure RGB (pre-trained models expect 3-channel input)
        return image.convert("RGB")


def _normalize_to_uint8(arr):
    """Normalize an array of any dtype to uint8 [0, 255]."""
    arr = arr.astype(np.float64)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
    return arr.astype(np.uint8)


def create_data_loaders(data_dir=None, batch_size=None, num_workers=None, seed=None):
    """
    Create train / val / test DataLoaders with stratified splitting.

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    data_dir = data_dir or config.DATA_DIR
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers or config.NUM_WORKERS
    seed = seed or config.SEED

    # Load full dataset (no transforms yet, applied per-subset below)
    full_dataset = XRayDataset(data_dir)
    class_names = full_dataset.classes
    labels = [s[1] for s in full_dataset.samples]

    # --- First split: train vs (val + test) ---
    splitter1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.VAL_RATIO + config.TEST_RATIO,
        random_state=seed,
    )
    train_idx, valtest_idx = next(splitter1.split(labels, labels))

    # --- Second split: val vs test ---
    valtest_labels = [labels[i] for i in valtest_idx]
    relative_test_ratio = config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    splitter2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=relative_test_ratio,
        random_state=seed,
    )
    val_rel_idx, test_rel_idx = next(
        splitter2.split(valtest_labels, valtest_labels)
    )
    val_idx = valtest_idx[val_rel_idx]
    test_idx = valtest_idx[test_rel_idx]

    # Build subset datasets with appropriate transforms
    train_ds = _SubsetWithTransform(full_dataset, train_idx, get_train_transforms())
    val_ds = _SubsetWithTransform(full_dataset, val_idx, get_val_transforms())
    test_ds = _SubsetWithTransform(full_dataset, test_idx, get_val_transforms())

    print(f"Dataset split — Train: {len(train_ds)}, "
          f"Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"Classes ({len(class_names)}): {class_names}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names


class _SubsetWithTransform(Dataset):
    """Subset wrapper that applies a specific transform."""

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[self.indices[idx]]
        image = self.dataset._load_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label
