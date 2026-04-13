"""Tests for model.py — model building, freezing, and loading."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import SUPPORTED_MODELS, build_model, load_model


class TestBuildModel:
    """build_model should return a valid nn.Module for every supported arch."""

    @pytest.mark.parametrize("name", ["densenet121", "efficientnet-b0", "se_resnet50"])
    def test_supported_architectures(self, name):
        model = build_model(
            num_classes=3, model_name=name,
            in_channels=1, pretrained=False,
        )
        assert model is not None
        # Forward pass with a small dummy input
        x = torch.randn(1, 1, 64, 64)
        out = model(x)
        assert out.shape == (1, 3)

    def test_unsupported_model_raises(self):
        with pytest.raises(ValueError, match="Unsupported model"):
            build_model(num_classes=2, model_name="nonexistent_model", pretrained=False)

    def test_output_shape_matches_num_classes(self):
        for nc in [2, 5, 10]:
            model = build_model(nc, model_name="densenet121", pretrained=False)
            out = model(torch.randn(1, 1, 64, 64))
            assert out.shape[1] == nc

    def test_multichannel_input(self):
        model = build_model(2, model_name="densenet121", in_channels=3, pretrained=False)
        out = model(torch.randn(1, 3, 64, 64))
        assert out.shape == (1, 2)


class TestFreezeBackbone:
    """Freezing should leave only the classifier head trainable."""

    def test_freeze_densenet(self):
        model = build_model(
            num_classes=2, model_name="densenet121",
            pretrained=False, freeze_backbone=True,
        )
        # class_layers should be trainable
        for p in model.class_layers.parameters():
            assert p.requires_grad
        # Some feature params should be frozen
        frozen = [p for p in model.features.parameters() if not p.requires_grad]
        assert len(frozen) > 0

    def test_freeze_efficientnet(self):
        model = build_model(
            num_classes=2, model_name="efficientnet-b0",
            pretrained=False, freeze_backbone=True,
        )
        for p in model._fc.parameters():
            assert p.requires_grad

    def test_freeze_seresnet(self):
        model = build_model(
            num_classes=2, model_name="se_resnet50",
            pretrained=False, freeze_backbone=True,
        )
        for p in model.last_linear.parameters():
            assert p.requires_grad


class TestLoadModel:
    """load_model should restore weights from a checkpoint file."""

    def test_load_from_checkpoint(self, tmp_path):
        model = build_model(2, model_name="densenet121", pretrained=False)
        ckpt_path = tmp_path / "test_model.pth"
        torch.save(model.state_dict(), ckpt_path)

        loaded = load_model(ckpt_path, num_classes=2,
                            model_name="densenet121", device="cpu")
        # Weights should match
        for (n1, p1), (n2, p2) in zip(
            model.state_dict().items(), loaded.state_dict().items()
        ):
            assert n1 == n2
            assert torch.equal(p1, p2), f"Mismatch in {n1}"

    def test_load_model_is_eval_mode(self, tmp_path):
        model = build_model(2, model_name="densenet121", pretrained=False)
        ckpt_path = tmp_path / "test_model.pth"
        torch.save(model.state_dict(), ckpt_path)

        loaded = load_model(ckpt_path, num_classes=2,
                            model_name="densenet121", device="cpu")
        assert not loaded.training


class TestSupportedModels:
    """SUPPORTED_MODELS list should be populated."""

    def test_supported_models_not_empty(self):
        assert len(SUPPORTED_MODELS) > 0

    def test_contains_key_architectures(self):
        for name in ["densenet121", "efficientnet-b0", "se_resnet50"]:
            assert name in SUPPORTED_MODELS
