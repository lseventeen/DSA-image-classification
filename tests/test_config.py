"""Tests for config.py — directory setup and class-name discovery."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class TestSetupDirs:
    """config.setup_dirs() should create output directories."""

    def test_creates_output_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")
        monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp_path / "out" / "ckpt")
        monkeypatch.setattr(config, "LOG_DIR", tmp_path / "out" / "logs")

        config.setup_dirs()

        assert (tmp_path / "out").is_dir()
        assert (tmp_path / "out" / "ckpt").is_dir()
        assert (tmp_path / "out" / "logs").is_dir()

    def test_idempotent(self, tmp_path, monkeypatch):
        """Calling setup_dirs twice should not raise."""
        monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")
        monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp_path / "out" / "ckpt")
        monkeypatch.setattr(config, "LOG_DIR", tmp_path / "out" / "logs")

        config.setup_dirs()
        config.setup_dirs()  # no error


class TestGetClassNames:
    """config.get_class_names() should discover sub-folder names."""

    def test_returns_sorted_class_names(self, synthetic_data_dir, monkeypatch):
        data_dir, expected = synthetic_data_dir
        monkeypatch.setattr(config, "DATA_DIR", data_dir)

        names = config.get_class_names()
        assert names == sorted(expected)

    def test_raises_on_missing_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "DATA_DIR", tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError):
            config.get_class_names()

    def test_raises_on_empty_dir(self, tmp_path, monkeypatch):
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setattr(config, "DATA_DIR", empty)
        with pytest.raises(ValueError):
            config.get_class_names()
