"""
Dataset utilities for medical image classification using MONAI.

Builds dictionary-based data lists from a class-folder layout and creates
MONAI ``CacheDataset`` / ``DataLoader`` instances with stratified splitting.

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

from monai.data import CacheDataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

import config
from transforms import get_train_transforms, get_val_transforms


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def build_data_list(root_dir):
    """Scan *root_dir* and return ``(data_list, class_names)``.

    Each element of *data_list* is a dictionary
    ``{"image": <str path>, "label": <int>}`` suitable for MONAI transforms.
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {root_dir}\n"
            "Please place your image folders under the 'data/' directory."
        )

    class_names = sorted(
        [d.name for d in root_dir.iterdir() if d.is_dir()]
    )
    if not class_names:
        raise ValueError(f"No subdirectories found in {root_dir}")

    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    data_list = []
    for cls_name in class_names:
        cls_dir = root_dir / cls_name
        for fname in sorted(os.listdir(cls_dir)):
            ext = Path(fname).suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                data_list.append({
                    "image": str(cls_dir / fname),
                    "label": class_to_idx[cls_name],
                })

    if not data_list:
        raise RuntimeError(
            f"No images found in {root_dir}. "
            f"Supported formats: {SUPPORTED_EXTENSIONS}"
        )
    return data_list, class_names


def create_data_loaders(data_dir=None, batch_size=None, num_workers=None,
                        seed=None):
    """Create train / val / test DataLoaders with stratified splitting.

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    data_dir = data_dir or config.DATA_DIR
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers or config.NUM_WORKERS
    seed = seed or config.SEED

    data_list, class_names = build_data_list(data_dir)
    labels = [d["label"] for d in data_list]

    # --- First split: train vs (val + test) ---
    splitter1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.VAL_RATIO + config.TEST_RATIO,
        random_state=seed,
    )
    train_idx, valtest_idx = next(splitter1.split(labels, labels))

    # --- Second split: val vs test ---
    valtest_labels = [labels[i] for i in valtest_idx]
    relative_test_ratio = (
        config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    )
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

    train_data = [data_list[i] for i in train_idx]
    val_data = [data_list[i] for i in val_idx]
    test_data = [data_list[i] for i in test_idx]

    # CacheDataset caches deterministic transforms; random augmentations
    # are re-applied every epoch automatically.
    train_ds = CacheDataset(
        data=train_data,
        transform=get_train_transforms(),
        cache_rate=1.0,
        num_workers=num_workers,
    )
    val_ds = CacheDataset(
        data=val_data,
        transform=get_val_transforms(),
        cache_rate=1.0,
        num_workers=num_workers,
    )
    test_ds = CacheDataset(
        data=test_data,
        transform=get_val_transforms(),
        cache_rate=1.0,
        num_workers=num_workers,
    )

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
