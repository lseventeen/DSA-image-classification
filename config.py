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
IMG_SIZE = 512          # Input image size for the model
NUM_WORKERS = 4         # DataLoader workers
TRAIN_RATIO = 0.8       # Training set ratio
TEST_RATIO = 0.2        # Test set ratio
SPLIT_FILE = DATA_DIR / "split.json"  # Cached train/test split

# ============ Model Configuration ============
# Options: densenet121, densenet169, densenet201, efficientnet-b0, se_resnet50
MODEL_NAME = "efficientnet-b0"
IN_CHANNELS = 1          # 1 for grayscale medical images
PRETRAINED = True         # Use pre-trained weights (MONAI model zoo)
FREEZE_BACKBONE = False   # Whether to freeze backbone layers initially
DROPOUT_PROB = 0.2        # Dropout probability for DenseNet

# ============ Training Configuration ============
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SCHEDULER_TYPE = "cosine"  # "step" or "cosine" (nnU-Net style poly/cosine)
SCHEDULER_STEP = 10        # StepLR step size (only used when SCHEDULER_TYPE="step")
SCHEDULER_GAMMA = 0.5      # StepLR gamma (only used when SCHEDULER_TYPE="step")
EARLY_STOPPING_PATIENCE = 10  # Early stopping patience
GRAD_CLIP_MAX_NORM = 1.0   # Max gradient norm for clipping (0 to disable)
USE_AMP = True              # Use automatic mixed precision training

# ============ Wandb Configuration ============
WANDB_ENABLED = True        # Enable/disable wandb logging
WANDB_PROJECT = "DSA-image-classification"
WANDB_ENTITY = None         # Wandb team/user name (None = default)
WANDB_LOG_FREQ = 1          # Log metrics every N epochs

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
