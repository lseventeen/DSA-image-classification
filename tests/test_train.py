"""Tests for train.py — training loop, checkpoint helpers, and epoch routines."""

import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train import _save_checkpoint, _load_checkpoint, train_one_epoch, validate


class _FakeLoader:
    """Minimal iterable mimicking a MONAI DataLoader for training/validation."""

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


class TestTrainOneEpoch:
    """train_one_epoch should return loss and accuracy."""

    def test_returns_loss_and_acc(self, small_model):
        n = 4
        images = torch.randn(n, 1, 64, 64)
        labels = torch.randint(0, 2, (n,))
        loader = _FakeLoader(images, labels, batch_size=2)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)

        loss, acc = train_one_epoch(
            small_model, loader, criterion, optimizer,
            device="cpu", scaler=None, grad_clip=0.0,
        )
        assert isinstance(loss, float)
        assert 0.0 <= acc <= 1.0

    def test_with_grad_clip(self, small_model):
        n = 4
        images = torch.randn(n, 1, 64, 64)
        labels = torch.randint(0, 2, (n,))
        loader = _FakeLoader(images, labels, batch_size=2)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)

        loss, acc = train_one_epoch(
            small_model, loader, criterion, optimizer,
            device="cpu", scaler=None, grad_clip=1.0,
        )
        assert isinstance(loss, float)


class TestValidate:
    """validate should return loss and accuracy without modifying the model."""

    def test_returns_loss_and_acc(self, small_model):
        n = 4
        images = torch.randn(n, 1, 64, 64)
        labels = torch.randint(0, 2, (n,))
        loader = _FakeLoader(images, labels, batch_size=2)

        criterion = nn.CrossEntropyLoss()
        loss, acc = validate(small_model, loader, criterion, device="cpu")
        assert isinstance(loss, float)
        assert 0.0 <= acc <= 1.0


class TestCheckpoint:
    """Checkpoint save/load round-trip should restore training state."""

    def test_save_load_roundtrip(self, small_model, tmp_path):
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)
        history = {"train_loss": [0.5], "test_loss": [0.6],
                   "train_acc": [0.7], "test_acc": [0.65]}

        ckpt_path = tmp_path / "ckpt.pth"
        _save_checkpoint(ckpt_path, small_model, optimizer, scheduler,
                         scaler=None, epoch=3, best_test_acc=0.65,
                         history=history)
        assert ckpt_path.exists()

        # Reload
        model2 = type(small_model)(  # cannot easily reconstruct — use build_model
            spatial_dims=2, in_channels=1, out_channels=2, pretrained=False
        )
        opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=5)

        epoch, best_acc, hist = _load_checkpoint(
            ckpt_path, model2, opt2, sched2, scaler=None, device="cpu",
        )
        assert epoch == 3
        assert best_acc == pytest.approx(0.65)
        assert hist["train_loss"] == [0.5]
