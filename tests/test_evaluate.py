"""Tests for evaluate.py — metrics collection, plotting, and reporting."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from dataset import create_data_loaders
from evaluate import collect_predictions, plot_confusion_matrix, plot_training_history
from model import build_model


class _FakeLoader:
    """Minimal iterable that mimics a MONAI DataLoader."""

    def __init__(self, images, labels, batch_size=2):
        self._images = images
        self._labels = labels
        self._bs = batch_size

    def __iter__(self):
        for i in range(0, len(self._labels), self._bs):
            yield {
                "image": self._images[i : i + self._bs],
                "label": self._labels[i : i + self._bs],
            }

    def __len__(self):
        return (len(self._labels) + self._bs - 1) // self._bs


class TestCollectPredictions:
    """collect_predictions should return arrays of labels, preds, probs."""

    def test_output_shapes(self, small_model):
        n = 6
        images = torch.randn(n, 1, 64, 64)
        labels = torch.randint(0, 2, (n,))
        loader = _FakeLoader(images, labels, batch_size=3)

        gt, preds, probs = collect_predictions(small_model, loader, device="cpu")
        assert gt.shape == (n,)
        assert preds.shape == (n,)
        assert probs.shape == (n, 2)

    def test_probs_sum_to_one(self, small_model):
        n = 4
        images = torch.randn(n, 1, 64, 64)
        labels = torch.randint(0, 2, (n,))
        loader = _FakeLoader(images, labels, batch_size=2)

        _, _, probs = collect_predictions(small_model, loader, device="cpu")
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_preds_are_valid_classes(self, small_model):
        n = 4
        images = torch.randn(n, 1, 64, 64)
        labels = torch.randint(0, 2, (n,))
        loader = _FakeLoader(images, labels, batch_size=2)

        _, preds, _ = collect_predictions(small_model, loader, device="cpu")
        assert all(0 <= p < 2 for p in preds)


class TestPlotConfusionMatrix:
    """plot_confusion_matrix should save a PNG file."""

    def test_saves_image(self, tmp_path):
        labels = np.array([0, 0, 1, 1, 0, 1])
        preds = np.array([0, 1, 1, 1, 0, 0])
        save_path = tmp_path / "cm.png"

        plot_confusion_matrix(labels, preds, ["class_a", "class_b"], save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0


class TestPlotTrainingHistory:
    """plot_training_history should generate training curve plots."""

    def test_saves_image(self, history_json, tmp_path):
        save_path = tmp_path / "curves.png"
        plot_training_history(history_json, save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0


class TestEvaluateOnTestSetImages:
    """End-to-end evaluation should work correctly on test-set images."""

    def test_collect_predictions_on_test_set(self, synthetic_data_dir, tmp_path, monkeypatch):
        """collect_predictions should produce valid outputs on real test-set DataLoader."""
        data_dir, _ = synthetic_data_dir
        monkeypatch.setattr(config, "DATA_DIR", data_dir)
        monkeypatch.setattr(config, "IMG_SIZE", 32)
        monkeypatch.setattr(config, "SPLIT_FILE", tmp_path / "split.json")
        monkeypatch.setattr(config, "TEST_RATIO", 0.5)

        _, test_loader, class_names = create_data_loaders(
            data_dir=data_dir, batch_size=2, num_workers=0, seed=42,
        )
        num_classes = len(class_names)
        model = build_model(
            num_classes, model_name="densenet121",
            in_channels=1, pretrained=False,
        )

        labels, preds, probs = collect_predictions(model, test_loader, device="cpu")

        test_size = len(test_loader.dataset)
        assert labels.shape == (test_size,)
        assert preds.shape == (test_size,)
        assert probs.shape == (test_size, num_classes)
        # All predictions should be valid class indices
        assert all(0 <= p < num_classes for p in preds)
        # Probabilities should sum to 1
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_confusion_matrix_on_test_set(self, synthetic_data_dir, tmp_path, monkeypatch):
        """Confusion matrix should be generated from test-set predictions."""
        data_dir, _ = synthetic_data_dir
        monkeypatch.setattr(config, "DATA_DIR", data_dir)
        monkeypatch.setattr(config, "IMG_SIZE", 32)
        monkeypatch.setattr(config, "SPLIT_FILE", tmp_path / "split.json")
        monkeypatch.setattr(config, "TEST_RATIO", 0.5)

        _, test_loader, class_names = create_data_loaders(
            data_dir=data_dir, batch_size=2, num_workers=0, seed=42,
        )
        model = build_model(
            len(class_names), model_name="densenet121",
            in_channels=1, pretrained=False,
        )

        labels, preds, _ = collect_predictions(model, test_loader, device="cpu")

        cm_path = tmp_path / "test_cm.png"
        plot_confusion_matrix(labels, preds, class_names, cm_path)
        assert cm_path.exists()
        assert cm_path.stat().st_size > 0
