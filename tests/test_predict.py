"""Tests for predict.py — single-image prediction utility."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from model import build_model
from predict import predict_single
from transforms import get_val_transforms


def _save_dummy_image(path, size=(64, 64)):
    arr = np.random.randint(0, 256, size=size, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


class TestPredictSingle:
    """predict_single should return a class name, confidence, and probs."""

    def test_returns_valid_prediction(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "IMG_SIZE", 64)
        img_path = tmp_path / "test_img.png"
        _save_dummy_image(img_path)

        model = build_model(3, model_name="densenet121",
                            in_channels=1, pretrained=False)
        model.eval()
        transform = get_val_transforms()
        class_names = ["alpha", "beta", "gamma"]

        cls_name, conf, probs = predict_single(
            img_path, model, transform, class_names, device="cpu",
        )
        assert cls_name in class_names
        assert 0.0 <= conf <= 1.0
        assert len(probs) == 3
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-5)

    def test_confidence_matches_max_prob(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "IMG_SIZE", 64)
        img_path = tmp_path / "test_img2.png"
        _save_dummy_image(img_path)

        model = build_model(2, model_name="densenet121",
                            in_channels=1, pretrained=False)
        model.eval()
        transform = get_val_transforms()

        cls_name, conf, probs = predict_single(
            img_path, model, transform, ["a", "b"], device="cpu",
        )
        assert conf == pytest.approx(probs.max(), abs=1e-5)

    def test_multiple_images_produce_valid_probs(self, tmp_path, monkeypatch):
        """Multiple different images should all produce valid probability vectors."""
        monkeypatch.setattr(config, "IMG_SIZE", 64)

        model = build_model(3, model_name="densenet121",
                            in_channels=1, pretrained=False)
        model.eval()
        transform = get_val_transforms()
        class_names = ["a", "b", "c"]

        for i, val in enumerate([0, 128, 255]):
            img_path = tmp_path / f"img_{i}.png"
            arr = np.full((64, 64), val, dtype=np.uint8)
            Image.fromarray(arr, mode="L").save(img_path)

            cls, conf, probs = predict_single(
                img_path, model, transform, class_names, device="cpu",
            )
            assert cls in class_names
            assert 0.0 <= conf <= 1.0
            assert len(probs) == 3
            np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-5)
