"""
Evaluation script — generates detailed metrics and visualizations.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint outputs/checkpoints/best_model.pth
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

import config
from dataset import create_data_loaders
from model import build_model


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run inference on the entire loader and return predictions + labels."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for batch_data in tqdm(loader, desc="Evaluating"):
        images = batch_data["image"].to(device)
        labels = batch_data["label"]

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(labels, preds, class_names, save_path):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {save_path}")


def plot_training_history(history_path, save_path):
    """Plot training curves from saved history JSON."""
    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["test_loss"], label="Test Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Test Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["test_acc"], label="Test Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Test Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {save_path}")


def evaluate(args):
    """Run full evaluation pipeline."""
    config.setup_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load metadata
    meta_path = config.CHECKPOINT_DIR / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        model_name = meta.get("model_name", args.model)
    else:
        model_name = args.model

    # Data
    _, test_loader, class_names = create_data_loaders()
    num_classes = len(class_names)

    # Model
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = build_model(
        num_classes,
        model_name=model_name,
        in_channels=config.IN_CHANNELS,
        pretrained=False,
    )
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.to(device)
    print(f"Loaded checkpoint: {checkpoint}")

    # Collect predictions
    labels, preds, probs = collect_predictions(model, test_loader, device)

    # Metrics
    acc = accuracy_score(labels, preds)
    report = classification_report(
        labels, preds, target_names=class_names, digits=4
    )
    print(f"\nTest Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(report)

    # Save report
    report_path = config.LOG_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write(report)
    print(f"Report saved to {report_path}")

    # Confusion matrix
    cm_path = config.OUTPUT_DIR / "confusion_matrix.png"
    plot_confusion_matrix(labels, preds, class_names, cm_path)

    # Training curves
    history_path = config.LOG_DIR / "training_history.json"
    if history_path.exists():
        curves_path = config.OUTPUT_DIR / "training_curves.png"
        plot_training_history(history_path, curves_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--checkpoint", type=str,
        default=str(config.CHECKPOINT_DIR / "best_model.pth"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model", type=str, default=config.MODEL_NAME,
        help="Model architecture (used if meta.json not found)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
