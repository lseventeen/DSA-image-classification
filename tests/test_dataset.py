"""Tests for dataset.py — data listing, splitting, and DataLoader creation."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from dataset import SUPPORTED_EXTENSIONS, build_data_list, create_data_loaders


class TestBuildDataList:
    """build_data_list should discover images grouped by class folder."""

    def test_discovers_all_images(self, synthetic_data_dir):
        data_dir, classes = synthetic_data_dir
        data_list, class_names = build_data_list(data_dir)
        assert class_names == sorted(classes)
        assert len(data_list) == 12  # 2 classes × 6 images

    def test_labels_are_correct(self, synthetic_data_dir):
        data_dir, _ = synthetic_data_dir
        data_list, class_names = build_data_list(data_dir)
        for item in data_list:
            assert 0 <= item["label"] < len(class_names)
            # Label should match the folder name
            folder = Path(item["image"]).parent.name
            assert class_names[item["label"]] == folder

    def test_raises_on_missing_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_data_list(tmp_path / "nonexistent")

    def test_raises_on_empty_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError):
            build_data_list(empty)

    def test_ignores_unsupported_extensions(self, tmp_path):
        cls_dir = tmp_path / "cls"
        cls_dir.mkdir()
        (cls_dir / "readme.txt").write_text("ignore me")
        (cls_dir / "data.csv").write_text("a,b")
        with pytest.raises(RuntimeError, match="No images found"):
            build_data_list(tmp_path)


class TestCreateDataLoaders:
    """create_data_loaders should return functional DataLoader pairs."""

    def test_returns_train_and_test_loaders(self, synthetic_data_dir, tmp_path, monkeypatch):
        data_dir, _ = synthetic_data_dir
        monkeypatch.setattr(config, "DATA_DIR", data_dir)
        monkeypatch.setattr(config, "IMG_SIZE", 32)
        monkeypatch.setattr(config, "SPLIT_FILE", tmp_path / "split.json")
        monkeypatch.setattr(config, "TEST_RATIO", 0.5)

        train_loader, test_loader, class_names = create_data_loaders(
            data_dir=data_dir, batch_size=2, num_workers=0, seed=42,
        )
        assert len(class_names) == 2
        assert len(train_loader.dataset) + len(test_loader.dataset) == 12

    def test_split_file_caching(self, synthetic_data_dir, tmp_path, monkeypatch):
        """A second call should load from the cached split file."""
        data_dir, _ = synthetic_data_dir
        split_file = tmp_path / "split.json"
        monkeypatch.setattr(config, "DATA_DIR", data_dir)
        monkeypatch.setattr(config, "IMG_SIZE", 32)
        monkeypatch.setattr(config, "SPLIT_FILE", split_file)
        monkeypatch.setattr(config, "TEST_RATIO", 0.5)

        create_data_loaders(data_dir=data_dir, batch_size=2, num_workers=0, seed=42)
        assert split_file.exists()

        # Second call should load from cache
        train_loader, test_loader, class_names = create_data_loaders(
            data_dir=data_dir, batch_size=2, num_workers=0, seed=42,
        )
        assert len(class_names) == 2

    def test_batch_content(self, synthetic_data_dir, tmp_path, monkeypatch):
        """Batches should contain 'image' tensors and 'label' tensors."""
        data_dir, _ = synthetic_data_dir
        monkeypatch.setattr(config, "DATA_DIR", data_dir)
        monkeypatch.setattr(config, "IMG_SIZE", 32)
        monkeypatch.setattr(config, "SPLIT_FILE", tmp_path / "split.json")
        monkeypatch.setattr(config, "TEST_RATIO", 0.5)

        train_loader, _, _ = create_data_loaders(
            data_dir=data_dir, batch_size=2, num_workers=0, seed=42,
        )
        batch = next(iter(train_loader))
        assert "image" in batch
        assert "label" in batch
        assert batch["image"].ndim == 4  # (B, C, H, W)


class TestSupportedExtensions:
    """SUPPORTED_EXTENSIONS should include common medical image formats."""

    def test_contains_tif(self):
        assert ".tif" in SUPPORTED_EXTENSIONS
        assert ".tiff" in SUPPORTED_EXTENSIONS

    def test_contains_common_formats(self):
        assert ".png" in SUPPORTED_EXTENSIONS
        assert ".jpg" in SUPPORTED_EXTENSIONS
        assert ".jpeg" in SUPPORTED_EXTENSIONS
