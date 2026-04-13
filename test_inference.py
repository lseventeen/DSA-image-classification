"""
Test-set inference and result analysis script.

Loads a trained model, runs inference on every test-set image, and produces
a comprehensive analysis including:
  - Per-sample predictions with confidence scores
  - Classification report (precision / recall / F1 / accuracy)
  - Confusion matrix visualisation
  - Misclassified-sample error analysis
  - Per-class confidence distribution
  - CSV export of all predictions

Usage:
    python test_inference.py
    python test_inference.py --checkpoint outputs/checkpoints/best_model.pth
    python test_inference.py --save_dir outputs/inference_results
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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


# ── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    """Run inference on the entire test loader.

    Returns:
        labels  – ground-truth label indices  (N,)
        preds   – predicted label indices      (N,)
        probs   – softmax probability vectors  (N, C)
        paths   – image file paths             list[str]
    """
    model.eval()
    all_labels, all_preds, all_probs, all_paths = [], [], [], []

    for batch_data in tqdm(loader, desc="Inference"):
        images = batch_data["image"].to(device)
        labels = batch_data["label"]

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        # Retrieve original file paths from the batch metadata
        if "image_meta_dict" in batch_data:
            meta = batch_data["image_meta_dict"]
            filenames = meta.get("filename_or_obj", [])
            if isinstance(filenames, (list, tuple)):
                all_paths.extend([str(f) for f in filenames])
            else:
                all_paths.append(str(filenames))

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        all_paths,
    )


# ── Analysis helpers ─────────────────────────────────────────────────────────

def print_classification_report(labels, preds, class_names):
    """Print and return the sklearn classification report."""
    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=class_names,
                                   digits=4, zero_division=0)
    print(f"\n{'=' * 60}")
    print(f"  Overall Accuracy: {acc:.4f}  ({int(acc * len(labels))}/{len(labels)})")
    print(f"{'=' * 60}")
    print("\nClassification Report:")
    print(report)
    return acc, report


def analyse_errors(labels, preds, probs, paths, class_names, top_n=20):
    """Print details for misclassified samples.

    Returns:
        list[dict] – error records
    """
    errors = []
    for i in range(len(labels)):
        if labels[i] != preds[i]:
            record = {
                "index": i,
                "image": paths[i] if i < len(paths) else "N/A",
                "true_label": class_names[labels[i]],
                "pred_label": class_names[preds[i]],
                "confidence": float(probs[i, preds[i]]),
                "true_class_prob": float(probs[i, labels[i]]),
            }
            errors.append(record)

    print(f"\n{'=' * 60}")
    print(f"  Error Analysis — {len(errors)} misclassified "
          f"out of {len(labels)} samples")
    print(f"{'=' * 60}")

    if not errors:
        print("  No misclassified samples! 🎉")
        return errors

    # Sort by confidence (descending) — high-confidence errors are most
    # informative
    errors.sort(key=lambda x: x["confidence"], reverse=True)

    shown = errors[:top_n]
    for e in shown:
        img_name = Path(e["image"]).name if e["image"] != "N/A" else "N/A"
        print(f"  [{e['index']:>4d}] {img_name:30s}  "
              f"True: {e['true_label']:15s}  "
              f"Pred: {e['pred_label']:15s}  "
              f"Conf: {e['confidence']:.4f}  "
              f"True-prob: {e['true_class_prob']:.4f}")

    if len(errors) > top_n:
        print(f"  … and {len(errors) - top_n} more")

    return errors


def analyse_confidence(labels, preds, probs, class_names):
    """Print per-class confidence statistics."""
    print(f"\n{'=' * 60}")
    print("  Per-class Confidence Statistics")
    print(f"{'=' * 60}")
    print(f"  {'Class':20s} {'Correct':>8s} {'Mean Conf':>10s} "
          f"{'Min Conf':>10s} {'Max Conf':>10s}")
    print(f"  {'-' * 58}")

    for idx, cls in enumerate(class_names):
        mask = labels == idx
        if mask.sum() == 0:
            continue
        cls_probs = probs[mask, idx]
        correct = (preds[mask] == idx).sum()
        total = mask.sum()
        print(f"  {cls:20s} {correct:>4d}/{total:<4d} "
              f"{cls_probs.mean():>10.4f} "
              f"{cls_probs.min():>10.4f} "
              f"{cls_probs.max():>10.4f}")


# ── Visualisation ────────────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, class_names, save_path):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) * 0.8)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test Set — Confusion Matrix")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nConfusion matrix saved to {save_path}")


def plot_confidence_distribution(probs, labels, preds, class_names, save_path):
    """Plot confidence histograms for correct vs incorrect predictions."""
    max_confs = probs.max(axis=1)
    correct_mask = labels == preds

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Overall histogram
    axes[0].hist(max_confs[correct_mask], bins=20, alpha=0.7,
                 label="Correct", color="#4CAF50", edgecolor="white")
    axes[0].hist(max_confs[~correct_mask], bins=20, alpha=0.7,
                 label="Incorrect", color="#F44336", edgecolor="white")
    axes[0].set_xlabel("Prediction Confidence")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Confidence Distribution")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Per-class accuracy bar
    per_class_acc = []
    for idx in range(len(class_names)):
        mask = labels == idx
        if mask.sum() > 0:
            per_class_acc.append((preds[mask] == idx).mean())
        else:
            per_class_acc.append(0.0)

    x = np.arange(len(class_names))
    bars = axes[1].bar(x, per_class_acc, color="#2196F3", edgecolor="white")
    for bar, acc in zip(bars, per_class_acc):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{acc:.2f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=30, ha="right")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Per-class Accuracy")
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confidence distribution saved to {save_path}")


# ── CSV export ───────────────────────────────────────────────────────────────

def save_predictions_csv(labels, preds, probs, paths, class_names, save_path):
    """Save per-sample predictions to a CSV file."""
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["index", "image", "true_label", "pred_label", "correct",
                   "confidence"] + [f"prob_{c}" for c in class_names]
        writer.writerow(header)

        for i in range(len(labels)):
            img = paths[i] if i < len(paths) else ""
            true_cls = class_names[labels[i]]
            pred_cls = class_names[preds[i]]
            correct = labels[i] == preds[i]
            conf = float(probs[i, preds[i]])
            row = [i, img, true_cls, pred_cls, correct, f"{conf:.6f}"]
            row += [f"{p:.6f}" for p in probs[i]]
            writer.writerow(row)

    print(f"Predictions CSV saved to {save_path}")


def save_summary_report(acc, report, errors, class_names, save_path):
    """Save a text summary report."""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("Test Set Inference — Summary Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write(f"Total misclassified: {len(errors)}\n\n")
        if errors:
            f.write("Top misclassified samples (by confidence):\n")
            for e in errors[:30]:
                img_name = Path(e["image"]).name if e["image"] != "N/A" else "N/A"
                f.write(f"  [{e['index']}] {img_name}  "
                        f"True: {e['true_label']}  "
                        f"Pred: {e['pred_label']}  "
                        f"Conf: {e['confidence']:.4f}\n")
    print(f"Summary report saved to {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Load metadata ────────────────────────────────────────────────────
    meta_path = config.CHECKPOINT_DIR / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        model_name = meta.get("model_name", args.model)
    else:
        model_name = args.model

    # ── Data ─────────────────────────────────────────────────────────────
    _, test_loader, class_names = create_data_loaders()
    num_classes = len(class_names)
    print(f"Test set size : {len(test_loader.dataset)}")
    print(f"Classes ({num_classes}): {class_names}")

    # ── Model ────────────────────────────────────────────────────────────
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

    # ── Inference ────────────────────────────────────────────────────────
    labels, preds, probs, paths = run_inference(model, test_loader, device)

    # ── Analysis ─────────────────────────────────────────────────────────
    acc, report = print_classification_report(labels, preds, class_names)
    errors = analyse_errors(labels, preds, probs, paths, class_names)
    analyse_confidence(labels, preds, probs, class_names)

    # ── Visualisation ────────────────────────────────────────────────────
    plot_confusion_matrix(labels, preds, class_names,
                          save_dir / "confusion_matrix.png")
    plot_confidence_distribution(probs, labels, preds, class_names,
                                 save_dir / "confidence_distribution.png")

    # ── Export ────────────────────────────────────────────────────────────
    save_predictions_csv(labels, preds, probs, paths, class_names,
                         save_dir / "predictions.csv")
    save_summary_report(acc, report, errors, class_names,
                        save_dir / "summary_report.txt")

    print(f"\n{'=' * 60}")
    print(f"  All results saved to {save_dir}/")
    print(f"{'=' * 60}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on the test set and analyse results"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=str(config.CHECKPOINT_DIR / "best_model.pth"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model", type=str, default=config.MODEL_NAME,
        help="Model architecture (used if meta.json not found)",
    )
    parser.add_argument(
        "--save_dir", type=str,
        default=str(config.OUTPUT_DIR / "inference_results"),
        help="Directory to save inference results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
