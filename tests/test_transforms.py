"""Tests for transforms.py — MONAI transform pipelines."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from transforms import EnsureSingleChanneld, get_train_transforms, get_val_transforms


def _make_image(tmp_path, size=(64, 64), mode="L", name="test.png"):
    """Helper: save a synthetic image and return a MONAI-style data dict."""
    if mode == "L":
        arr = np.random.randint(0, 256, size=size, dtype=np.uint8)
    else:
        arr = np.random.randint(0, 256, size=(*size, 3), dtype=np.uint8)
    path = tmp_path / name
    Image.fromarray(arr, mode=mode).save(path)
    return {"image": str(path), "label": 0}


class TestValTransforms:
    """Validation transforms should produce deterministic (C, H, W) tensors."""

    def test_output_shape(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "IMG_SIZE", 64)
        data = _make_image(tmp_path)
        t = get_val_transforms()
        result = t(data)
        img = result["image"]
        assert isinstance(img, torch.Tensor)
        assert img.ndim == 3  # (C, H, W)
        assert img.shape[0] == 1  # single channel

    def test_intensity_range(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "IMG_SIZE", 64)
        data = _make_image(tmp_path)
        t = get_val_transforms()
        img = t(data)["image"]
        assert img.min() >= 0.0
        assert img.max() <= 1.0 + 1e-6

    def test_deterministic(self, tmp_path, monkeypatch):
        """Running val transforms twice should give the same result."""
        monkeypatch.setattr(config, "IMG_SIZE", 64)
        data = _make_image(tmp_path)
        t = get_val_transforms()
        r1 = t(data)["image"]
        r2 = t(data)["image"]
        assert torch.allclose(r1, r2)


class TestTrainTransforms:
    """Training transforms should produce valid tensors (may differ each run)."""

    def test_output_shape(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "IMG_SIZE", 64)
        data = _make_image(tmp_path)
        t = get_train_transforms()
        result = t(data)
        img = result["image"]
        assert isinstance(img, torch.Tensor)
        assert img.ndim == 3
        assert img.shape[0] == 1

    def test_rgb_to_grayscale(self, tmp_path, monkeypatch):
        """RGB images should be collapsed to 1-channel."""
        monkeypatch.setattr(config, "IMG_SIZE", 64)
        data = _make_image(tmp_path, mode="RGB", name="rgb.png")
        t = get_val_transforms()
        img = t(data)["image"]
        assert img.shape[0] == 1


class TestEnsureSingleChanneld:
    """EnsureSingleChanneld should average multi-channel images."""

    def test_collapses_rgb(self):
        transform = EnsureSingleChanneld(keys=["image"])
        data = {"image": torch.randn(3, 64, 64)}
        result = transform(data)
        assert result["image"].shape[0] == 1

    def test_keeps_single_channel(self):
        transform = EnsureSingleChanneld(keys=["image"])
        original = torch.randn(1, 64, 64)
        data = {"image": original.clone()}
        result = transform(data)
        assert result["image"].shape[0] == 1
        assert torch.equal(result["image"], original)
