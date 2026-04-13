"""Tests for dataset.py — data listing, splitting, and DataLoader creation."""

import json
import sys
from pathlib import Path

import pytest
import torch

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


class TestTestSetImages:
    """Tests focused on the test-set split: content, format, and correctness."""

    def test_test_batch_content(self, synthetic_data_dir, tmp_path, monkeypatch):
        """Test-set batches should contain 'image' and 'label' tensors."""
        data_dir, _ = synthetic_data_dir
        monkeypatch.setattr(config, "DATA_DIR", data_dir)
        monkeypatch.setattr(config, "IMG_SIZE", 32)
        monkeypatch.setattr(config, "SPLIT_FILE", tmp_path / "split.json")
        monkeypatch.setattr(config, "TEST_RATIO", 0.5)

        _, test_loader, _ = create_data_loaders(
            data_dir=data_dir, batch_size=2, num_workers=0, seed=42,
        )
        batch = next(iter(test_loader))
        assert "image" in batch
        assert "label" in batch
        assert batch["image"].ndim == 4  # (B, C, H, W)
        assert batch["image"].shape[1] == 1  # single channel

    def test_test_labels_valid(self, synthetic_data_dir, tmp_path, monkeypatch):
        """Every label in the test set should be a valid class index."""
        data_dir, _ = synthetic_data_dir
        monkeypatch.setattr(config, "DATA_DIR", data_dir)
        monkeypatch.setattr(config, "IMG_SIZE", 32)
        monkeypatch.setattr(config, "SPLIT_FILE", tmp_path / "split.json")
        monkeypatch.setattr(config, "TEST_RATIO", 0.5)

        _, test_loader, class_names = create_data_loaders(
            data_dir=data_dir, batch_size=2, num_workers=0, seed=42,
        )
        num_classes = len(class_names)
        for batch in test_loader:
            labels = batch["label"]
            assert (labels >= 0).all()
            assert (labels < num_classes).all()

    def test_test_images_disjoint_from_train(self, synthetic_data_dir, tmp_path, monkeypatch):
        """Test-set images must not overlap with the training set."""
        data_dir, _ = synthetic_data_dir
        split_file = tmp_path / "split.json"
        monkeypatch.setattr(config, "DATA_DIR", data_dir)
        monkeypatch.setattr(config, "IMG_SIZE", 32)
        monkeypatch.setattr(config, "SPLIT_FILE", split_file)
        monkeypatch.setattr(config, "TEST_RATIO", 0.5)

        create_data_loaders(data_dir=data_dir, batch_size=2, num_workers=0, seed=42)

        with open(split_file) as f:
            split = json.load(f)
        train_paths = {item["image"] for item in split["train"]}
        test_paths = {item["image"] for item in split["test"]}
        assert train_paths.isdisjoint(test_paths), "Train and test sets share images"

    def test_test_set_ratio(self, synthetic_data_dir, tmp_path, monkeypatch):
        """Test set size should respect the configured TEST_RATIO."""
        data_dir, _ = synthetic_data_dir
        split_file = tmp_path / "split.json"
        monkeypatch.setattr(config, "DATA_DIR", data_dir)
        monkeypatch.setattr(config, "IMG_SIZE", 32)
        monkeypatch.setattr(config, "SPLIT_FILE", split_file)
        monkeypatch.setattr(config, "TEST_RATIO", 0.5)

        _, test_loader, _ = create_data_loaders(
            data_dir=data_dir, batch_size=2, num_workers=0, seed=42,
        )
        total = 12  # 2 classes × 6 images
        test_size = len(test_loader.dataset)
        expected = int(total * 0.5)
        assert test_size == expected

    def test_test_set_deterministic(self, synthetic_data_dir, tmp_path, monkeypatch):
        """Running val transforms on the same test image twice should give identical results."""
        data_dir, _ = synthetic_data_dir
        split_file = tmp_path / "split.json"
        monkeypatch.setattr(config, "DATA_DIR", data_dir)
        monkeypatch.setattr(config, "IMG_SIZE", 32)
        monkeypatch.setattr(config, "SPLIT_FILE", split_file)
        monkeypatch.setattr(config, "TEST_RATIO", 0.5)

        _, test_loader1, _ = create_data_loaders(
            data_dir=data_dir, batch_size=6, num_workers=0, seed=42,
        )
        _, test_loader2, _ = create_data_loaders(
            data_dir=data_dir, batch_size=6, num_workers=0, seed=42,
        )
        batch1 = next(iter(test_loader1))
        batch2 = next(iter(test_loader2))
        assert torch.allclose(batch1["image"], batch2["image"])

    def test_test_set_covers_all_classes(self, synthetic_data_dir, tmp_path, monkeypatch):
        """The test set should contain images from every class."""
        data_dir, _ = synthetic_data_dir
        split_file = tmp_path / "split.json"
        monkeypatch.setattr(config, "DATA_DIR", data_dir)
        monkeypatch.setattr(config, "IMG_SIZE", 32)
        monkeypatch.setattr(config, "SPLIT_FILE", split_file)
        monkeypatch.setattr(config, "TEST_RATIO", 0.5)

        _, test_loader, class_names = create_data_loaders(
            data_dir=data_dir, batch_size=12, num_workers=0, seed=42,
        )
        all_labels = set()
        for batch in test_loader:
            all_labels.update(batch["label"].tolist())
        assert all_labels == set(range(len(class_names)))


class TestSupportedExtensions:
    """SUPPORTED_EXTENSIONS should include common medical image formats."""

    def test_contains_tif(self):
        assert ".tif" in SUPPORTED_EXTENSIONS
        assert ".tiff" in SUPPORTED_EXTENSIONS

    def test_contains_common_formats(self):
        assert ".png" in SUPPORTED_EXTENSIONS
        assert ".jpg" in SUPPORTED_EXTENSIONS
        assert ".jpeg" in SUPPORTED_EXTENSIONS
