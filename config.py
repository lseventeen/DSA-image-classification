"""
Project configuration for medical image classification (MONAI).
"""

import os
from pathlib import Path

# ============ Path Configuration ============
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

# ============ Data Configuration ============
IMG_SIZE = 224          # Input image size for the model
NUM_WORKERS = 4         # DataLoader workers
TRAIN_RATIO = 0.7       # Training set ratio
VAL_RATIO = 0.15        # Validation set ratio
TEST_RATIO = 0.15       # Test set ratio

# ============ Model Configuration ============
# Options: densenet121, densenet169, densenet201, efficientnet-b0, se_resnet50
MODEL_NAME = "densenet121"
IN_CHANNELS = 1          # 1 for grayscale medical images
PRETRAINED = True         # Use pre-trained weights (MONAI model zoo)
FREEZE_BACKBONE = False   # Whether to freeze backbone layers initially
DROPOUT_PROB = 0.2        # Dropout probability for DenseNet

# ============ Training Configuration ============
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SCHEDULER_STEP = 10     # StepLR step size
SCHEDULER_GAMMA = 0.5   # StepLR gamma
EARLY_STOPPING_PATIENCE = 10  # Early stopping patience

# ============ Random Seed ============
SEED = 42


def setup_dirs():
    """Create output directories if they don't exist."""
    for d in [OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def get_class_names():
    """Discover class names from the data directory folder structure."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data directory not found: {DATA_DIR}\n"
            "Please place your image folders under the 'data/' directory."
        )
    class_names = sorted(
        [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
    )
    if not class_names:
        raise ValueError(f"No subdirectories found in {DATA_DIR}")
    return class_names
