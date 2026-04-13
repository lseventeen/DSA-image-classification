"""
Training script for medical image classification using MONAI.

Usage:
    python train.py
    python train.py --epochs 100 --batch_size 32 --lr 0.0001
    python train.py --model densenet169 --freeze_backbone
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from monai.utils import set_determinism
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import config
from dataset import create_data_loaders
from model import SUPPORTED_MODELS, build_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch and return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_data in tqdm(loader, desc="  Train", leave=False):
        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate and return average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_data in tqdm(loader, desc="  Val  ", leave=False):
        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device, dtype=torch.long)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train(args):
    """Main training loop with early stopping and model checkpointing."""
    set_determinism(seed=args.seed)
    config.setup_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        batch_size=args.batch_size,
        seed=args.seed,
    )
    num_classes = len(class_names)

    # --- Model ---
    model = build_model(
        num_classes=num_classes,
        model_name=args.model,
        in_channels=config.IN_CHANNELS,
        pretrained=config.PRETRAINED,
        freeze_backbone=args.freeze_backbone,
        dropout_prob=config.DROPOUT_PROB,
    )
    model.to(device)

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | "
          f"Trainable params: {trainable_params:,} / {total_params:,}")

    # --- Optimizer & Scheduler ---
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = StepLR(
        optimizer, step_size=args.scheduler_step,
        gamma=args.scheduler_gamma,
    )

    # --- Training loop ---
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }

    print(f"\n{'=' * 60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch:3d}/{args.epochs}]  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  |  "
            f"LR: {lr:.6f}"
        )

        # Checkpoint best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            ckpt_path = config.CHECKPOINT_DIR / "best_model.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(no improvement for {args.patience} epochs)"
                )
                break

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed / 60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # --- Save final model and training history ---
    torch.save(model.state_dict(), config.CHECKPOINT_DIR / "last_model.pth")

    history_path = config.LOG_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    # --- Save class names for inference ---
    meta = {"class_names": class_names, "model_name": args.model}
    with open(config.CHECKPOINT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # --- Evaluate on test set ---
    print(f"\n{'=' * 60}")
    print("Evaluating on test set...")
    print(f"{'=' * 60}")

    best_model = build_model(
        num_classes,
        model_name=args.model,
        in_channels=config.IN_CHANNELS,
        pretrained=False,
    )
    best_model.load_state_dict(
        torch.load(
            config.CHECKPOINT_DIR / "best_model.pth",
            map_location=device,
            weights_only=True,
        )
    )
    best_model.to(device)

    test_loss, test_acc = validate(best_model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}  |  Test Acc: {test_acc:.4f}")

    return history


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train medical image classifier (MONAI)"
    )
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument(
        "--weight_decay", type=float, default=config.WEIGHT_DECAY
    )
    parser.add_argument(
        "--model", type=str, default=config.MODEL_NAME,
        choices=SUPPORTED_MODELS,
    )
    parser.add_argument(
        "--freeze_backbone", action="store_true",
        default=config.FREEZE_BACKBONE,
    )
    parser.add_argument(
        "--patience", type=int, default=config.EARLY_STOPPING_PATIENCE
    )
    parser.add_argument(
        "--scheduler_step", type=int, default=config.SCHEDULER_STEP
    )
    parser.add_argument(
        "--scheduler_gamma", type=float, default=config.SCHEDULER_GAMMA
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
