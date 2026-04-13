"""
Predict the class of a single X-ray image.

Usage:
    python predict.py path/to/image.tif
    python predict.py path/to/image.tif --checkpoint outputs/checkpoints/best_model.pth
    python predict.py path/to/image_dir/ --batch
"""

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import config
from dataset import XRayDataset
from model import build_model
from transforms import get_val_transforms


def predict_single(image_path, model, transform, class_names, device):
    """Predict class for a single image. Returns (class_name, confidence, probs)."""
    image = XRayDataset._load_image(image_path)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).squeeze()

    confidence, pred_idx = probs.max(0)
    return class_names[pred_idx.item()], confidence.item(), probs.cpu().numpy()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load metadata
    meta_path = config.CHECKPOINT_DIR / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            "meta.json not found. Train a model first with: python train.py"
        )
    with open(meta_path) as f:
        meta = json.load(f)
    class_names = meta["class_names"]
    model_name = meta.get("model_name", args.model)

    # Load model
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = build_model(len(class_names), model_name=model_name, pretrained=False)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    transform = get_val_transforms()
    input_path = Path(args.input)

    if input_path.is_dir():
        # Batch prediction on a directory
        supported = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        files = sorted(
            f for f in input_path.iterdir()
            if f.suffix.lower() in supported
        )
        if not files:
            print(f"No supported images found in {input_path}")
            return

        print(f"Predicting {len(files)} images from {input_path}\n")
        for fpath in tqdm(files, desc="Predicting"):
            cls, conf, _ = predict_single(fpath, model, transform,
                                          class_names, device)
            print(f"  {fpath.name:40s} → {cls:20s} ({conf:.4f})")
    else:
        # Single image
        cls, conf, probs = predict_single(
            input_path, model, transform, class_names, device
        )
        print(f"\nImage  : {input_path}")
        print(f"Class  : {cls}")
        print(f"Confidence: {conf:.4f}")
        print(f"\nAll class probabilities:")
        for name, prob in zip(class_names, probs):
            bar = "█" * int(prob * 30)
            print(f"  {name:20s} {prob:.4f}  {bar}")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict X-ray image class")
    parser.add_argument("input", type=str,
                        help="Path to image file or directory")
    parser.add_argument("--checkpoint", type=str,
                        default=str(config.CHECKPOINT_DIR / "best_model.pth"))
    parser.add_argument("--model", type=str, default=config.MODEL_NAME)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
