import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest
import yaml


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path)


@pytest.fixture
def temp_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary file for testing."""
    temp_file = temp_dir / "test_file.txt"
    temp_file.write_text("test content")
    yield temp_file


@pytest.fixture
def mock_config() -> dict:
    """Provide a mock configuration dictionary."""
    return {
        "model": {
            "name": "test_model",
            "architecture": "GATsSPG",
            "num_layers": 3,
            "hidden_dim": 256,
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "num_epochs": 100,
            "optimizer": "adam",
        },
        "data": {
            "train_path": "/path/to/train",
            "val_path": "/path/to/val",
            "test_path": "/path/to/test",
            "num_workers": 4,
        },
        "experiment": {
            "name": "test_experiment",
            "output_dir": "/path/to/output",
            "checkpoint_dir": "/path/to/checkpoints",
        },
    }


@pytest.fixture
def sample_yaml_config(temp_dir: Path, mock_config: dict) -> Path:
    """Create a sample YAML configuration file."""
    config_path = temp_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(mock_config, f)
    return config_path


@pytest.fixture
def sample_image_data() -> dict:
    """Provide sample image metadata."""
    return {
        "image_id": "test_001",
        "width": 640,
        "height": 480,
        "channels": 3,
        "format": "RGB",
        "path": "/path/to/image.jpg",
    }


@pytest.fixture
def sample_annotation_data() -> dict:
    """Provide sample annotation data."""
    return {
        "annotations": [
            {
                "id": 1,
                "category": "object",
                "bbox": [100, 100, 200, 200],
                "keypoints": [[150, 150], [250, 250]],
                "confidence": 0.95,
            }
        ],
        "metadata": {
            "version": "1.0",
            "timestamp": "2024-01-01T00:00:00",
        },
    }


@pytest.fixture
def mock_model_checkpoint(temp_dir: Path) -> Path:
    """Create a mock model checkpoint file."""
    checkpoint_path = temp_dir / "model_checkpoint.pth"
    checkpoint_path.write_text("mock checkpoint data")
    return checkpoint_path


@pytest.fixture
def environment_setup(monkeypatch):
    """Set up environment variables for testing."""
    test_env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "WANDB_MODE": "offline",
        "HYDRA_FULL_ERROR": "1",
        "TEST_MODE": "true",
    }
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    return test_env


@pytest.fixture(autouse=True)
def cleanup_wandb(monkeypatch):
    """Disable wandb during tests."""
    monkeypatch.setenv("WANDB_MODE", "disabled")


@pytest.fixture
def mock_dataset_structure(temp_dir: Path) -> Path:
    """Create a mock dataset directory structure."""
    dataset_dir = temp_dir / "mock_dataset"
    (dataset_dir / "train" / "images").mkdir(parents=True)
    (dataset_dir / "train" / "annotations").mkdir(parents=True)
    (dataset_dir / "val" / "images").mkdir(parents=True)
    (dataset_dir / "val" / "annotations").mkdir(parents=True)
    (dataset_dir / "test" / "images").mkdir(parents=True)
    (dataset_dir / "test" / "annotations").mkdir(parents=True)
    
    # Create sample files
    (dataset_dir / "train" / "images" / "img_001.jpg").write_text("mock image")
    (dataset_dir / "train" / "annotations" / "ann_001.json").write_text("{}")
    
    return dataset_dir


@pytest.fixture
def capture_logs(caplog):
    """Fixture to capture log messages during tests."""
    with caplog.at_level("DEBUG"):
        yield caplog