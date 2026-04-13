"""Tests for evaluate.py — metrics collection, plotting, and reporting."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate import collect_predictions, plot_confusion_matrix, plot_training_history


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
