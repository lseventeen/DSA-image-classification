"""
Shared pytest fixtures for DSA-image-classification tests.

Creates temporary synthetic data directories and lightweight model fixtures
so that every test module can focus on its own logic without duplicating setup.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# Make the project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Synthetic image helpers
# ---------------------------------------------------------------------------

def _create_synthetic_image(path: Path, size=(64, 64), mode="L"):
    """Create a random grayscale (or RGB) image and save it."""
    if mode == "L":
        arr = np.random.randint(0, 256, size=size, dtype=np.uint8)
    else:
        arr = np.random.randint(0, 256, size=(*size, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode=mode)
    img.save(path)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_data_dir(tmp_path):
    """Create a temporary data directory with two classes, 6 images each."""
    classes = ["class_a", "class_b"]
    for cls in classes:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(6):
            _create_synthetic_image(cls_dir / f"img_{i:03d}.png")
    return tmp_path, classes


@pytest.fixture()
def synthetic_rgb_data_dir(tmp_path):
    """Create a temporary data directory with RGB images."""
    classes = ["cat", "dog"]
    for cls in classes:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(4):
            _create_synthetic_image(cls_dir / f"img_{i:03d}.png", mode="RGB")
    return tmp_path, classes


@pytest.fixture()
def small_model():
    """Build a small DenseNet121 for 2 classes (CPU, no pretrained)."""
    from model import build_model
    return build_model(
        num_classes=2,
        model_name="densenet121",
        in_channels=1,
        pretrained=False,
        dropout_prob=0.0,
    )


@pytest.fixture()
def dummy_batch():
    """A random (B, 1, 64, 64) tensor and label tensor for 2 classes."""
    images = torch.randn(2, 1, 64, 64)
    labels = torch.tensor([0, 1])
    return images, labels


@pytest.fixture()
def training_history():
    """A small fake training history dict."""
    return {
        "train_loss": [0.8, 0.6, 0.4, 0.3, 0.25],
        "train_acc":  [0.5, 0.65, 0.75, 0.82, 0.88],
        "test_loss":  [0.9, 0.7, 0.5, 0.45, 0.42],
        "test_acc":   [0.45, 0.60, 0.70, 0.76, 0.80],
    }


@pytest.fixture()
def history_json(tmp_path, training_history):
    """Write training history to a JSON file and return its path."""
    path = tmp_path / "training_history.json"
    with open(path, "w") as f:
        json.dump(training_history, f)
    return path
