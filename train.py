"""
Training script for medical image classification using MONAI.

Features (nnU-Net inspired):
- Wandb experiment tracking
- Automatic mixed-precision (AMP) training
- Gradient clipping
- Cosine-annealing LR scheduler (or StepLR)
- Full checkpoint saving with resume support

Usage:
    python train.py
    python train.py --epochs 100 --batch_size 32 --lr 0.0001
    python train.py --model densenet169 --freeze_backbone
    python train.py --no_wandb          # disable wandb
    python train.py --resume outputs/checkpoints/last_checkpoint.pth
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from monai.utils import set_determinism
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

import config
from dataset import create_data_loaders
from model import SUPPORTED_MODELS, build_model

try:
    import wandb
except ImportError:
    wandb = None


# ---------------------------------------------------------------------------
# Wandb helpers
# ---------------------------------------------------------------------------

def _init_wandb(args, num_classes, class_names):
    """Initialise a wandb run (no-op when disabled or unavailable)."""
    if not args.use_wandb:
        return None
    if wandb is None:
        print("Warning: wandb is not installed. Skipping wandb logging.")
        return None

    run = wandb.init(
        project=config.WANDB_PROJECT,
        entity=config.WANDB_ENTITY,
        config={
            "model": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "scheduler_type": args.scheduler_type,
            "scheduler_step": args.scheduler_step,
            "scheduler_gamma": args.scheduler_gamma,
            "patience": args.patience,
            "freeze_backbone": args.freeze_backbone,
            "dropout_prob": config.DROPOUT_PROB,
            "img_size": config.IMG_SIZE,
            "in_channels": config.IN_CHANNELS,
            "num_classes": num_classes,
            "class_names": class_names,
            "seed": args.seed,
            "use_amp": args.use_amp,
            "grad_clip": args.grad_clip,
        },
        reinit=True,
    )
    return run


def _log_wandb(run, metrics, epoch):
    """Log a dict of metrics to wandb for the given epoch."""
    if run is None:
        return
    wandb.log(metrics, step=epoch)


def _finish_wandb(run):
    """Finish the wandb run gracefully."""
    if run is None:
        return
    wandb.finish()


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device,
                    scaler=None, grad_clip=0.0):
    """Train for one epoch with optional AMP and gradient clipping."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None

    for batch_data in tqdm(loader, desc="  Train", leave=False):
        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(path, model, optimizer, scheduler, scaler, epoch,
                     best_test_acc, history):
    """Save a full checkpoint (model + training state) for resume."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_test_acc": best_test_acc,
        "history": history,
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, path)


def _load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    """Load a checkpoint and restore training state.

    Returns:
        (start_epoch, best_test_acc, history)
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt["epoch"], ckpt["best_test_acc"], ckpt["history"]


# ---------------------------------------------------------------------------
# Main training entry-point
# ---------------------------------------------------------------------------

def train(args):
    """Main training loop with wandb, AMP, gradient clipping, and
    cosine-annealing LR scheduler."""
    set_determinism(seed=args.seed)
    config.setup_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    train_loader, test_loader, class_names = create_data_loaders(
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

    if args.scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    else:
        scheduler = StepLR(
            optimizer, step_size=args.scheduler_step,
            gamma=args.scheduler_gamma,
        )

    # --- AMP scaler ---
    use_amp = args.use_amp and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Automatic Mixed Precision (AMP) enabled")

    # --- Wandb ---
    wb_run = _init_wandb(args, num_classes, class_names)
    if wb_run is not None:
        wandb.watch(model, log="gradients", log_freq=100)

    # --- Resume ---
    start_epoch = 1
    best_test_acc = 0.0
    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": [],
    }

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        last_epoch, best_test_acc, history = _load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device,
        )
        start_epoch = last_epoch + 1
        print(f"  Resumed at epoch {start_epoch}, "
              f"best_test_acc={best_test_acc:.4f}")

    # --- Training loop ---
    patience_counter = 0

    print(f"\n{'=' * 60}")
    print(f"Starting training for {args.epochs} epochs "
          f"(from epoch {start_epoch})")
    print(f"Scheduler: {args.scheduler_type} | "
          f"Grad clip: {args.grad_clip} | AMP: {use_amp}")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, grad_clip=args.grad_clip,
        )
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch:3d}/{args.epochs}]  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
            f"Test Loss: {test_loss:.4f}  Acc: {test_acc:.4f}  |  "
            f"LR: {lr:.6f}"
        )

        # --- Wandb logging ---
        if epoch % config.WANDB_LOG_FREQ == 0:
            _log_wandb(wb_run, {
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": test_loss,
                "val/accuracy": test_acc,
                "lr": lr,
                "epoch": epoch,
            }, epoch)

        # Checkpoint best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
            best_path = config.CHECKPOINT_DIR / "best_model.pth"
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Best model saved (test_acc={test_acc:.4f})")
            if wb_run is not None:
                wandb.run.summary["best_test_acc"] = best_test_acc
                wandb.run.summary["best_epoch"] = epoch
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(no improvement for {args.patience} epochs)"
                )
                break

        # Save resumable checkpoint every epoch
        _save_checkpoint(
            config.CHECKPOINT_DIR / "last_checkpoint.pth",
            model, optimizer, scheduler, scaler,
            epoch, best_test_acc, history,
        )

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed / 60:.1f} minutes")
    print(f"Best test accuracy: {best_test_acc:.4f}")

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

    _finish_wandb(wb_run)
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
        "--scheduler_type", type=str, default=config.SCHEDULER_TYPE,
        choices=["step", "cosine"],
        help="LR scheduler type: 'step' (StepLR) or 'cosine' (CosineAnnealing)",
    )
    parser.add_argument(
        "--scheduler_step", type=int, default=config.SCHEDULER_STEP
    )
    parser.add_argument(
        "--scheduler_gamma", type=float, default=config.SCHEDULER_GAMMA
    )
    parser.add_argument(
        "--grad_clip", type=float, default=config.GRAD_CLIP_MAX_NORM,
        help="Max gradient norm for clipping (0 to disable)",
    )
    parser.add_argument(
        "--use_amp", action="store_true", default=config.USE_AMP,
        help="Enable automatic mixed precision training",
    )
    parser.add_argument(
        "--no_amp", action="store_true",
        help="Disable AMP even if config enables it",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", default=config.WANDB_ENABLED,
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--no_wandb", action="store_true",
        help="Disable wandb logging",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument("--seed", type=int, default=config.SEED)

    args = parser.parse_args()
    # Handle negative flags
    if args.no_amp:
        args.use_amp = False
    if args.no_wandb:
        args.use_wandb = False
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
