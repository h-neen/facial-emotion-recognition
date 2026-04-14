"""
notebooks/grad_cam_visualise.py
================================
Generate Grad-CAM heatmaps using the pytorch-grad-cam library.
Target layer: EfficientNetB0's last conv block (features[-1][0]).

Usage:
    python notebooks/grad_cam_visualise.py \
        --checkpoint checkpoints/fer_best_val_acc.pth \
        --image_dir  data/FER2013_SR/validation/happy \
        --output_dir results/gradcam_happy \
        --n_samples  10
"""

import argparse, random, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import CFG
from preprocessing.transforms import get_val_transforms
from models.ea_net import build_model


def load_model(ckpt_path: Path, device):
    model = build_model(CFG, freeze_backbone=False).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model


def get_target_layer(model):
    """Last conv block of EfficientNetB0 features."""
    return [model.efficient.features[-1][0]]


def generate_gradcam(
    model,
    img_paths: list,
    transform,
    device,
    output_dir: Path,
    class_names=list(CFG.class_names),
):
    target_layers = get_target_layer(model)
    cam = GradCAM(model=model, target_layers=target_layers)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(img_paths):
        pil_img  = Image.open(path).convert("RGB")
        tensor   = transform(pil_img).unsqueeze(0).to(device)  # (1,3,H,W)

        # Run GradCAM
        grayscale_cam = cam(input_tensor=tensor, targets=None)  # None = use predicted class
        grayscale_cam = grayscale_cam[0]   # (H, W)

        # Predicted class
        with torch.no_grad():
            logits = model(tensor)
        pred_idx  = logits.argmax(dim=1).item()
        pred_name = class_names[pred_idx]
        conf      = torch.softmax(logits, dim=1)[0, pred_idx].item() * 100

        # Overlay
        rgb_img  = np.array(pil_img.resize((CFG.image_size, CFG.image_size))) / 255.0
        cam_img  = show_cam_on_image(rgb_img.astype(np.float32), grayscale_cam, use_rgb=True)

        # Plot side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(rgb_img)
        axes[0].set_title(f"Original\n{path.name}", fontsize=9)
        axes[0].axis("off")
        axes[1].imshow(cam_img)
        axes[1].set_title(f"Grad-CAM\nPred: {pred_name} ({conf:.1f}%)", fontsize=9)
        axes[1].axis("off")
        plt.suptitle(f"EA-Net Grad-CAM — Sample {i+1}", fontsize=11, y=1.01)
        plt.tight_layout()

        save_path = output_dir / f"gradcam_{i:03d}_{pred_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [{i+1}/{len(img_paths)}] saved → {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--image_dir",  required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--n_samples",  default=10,    type=int)
    ap.add_argument("--seed",       default=42,    type=int)
    args = ap.parse_args()

    random.seed(args.seed)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_val_transforms(CFG.image_size)
    model     = load_model(args.checkpoint, device)

    img_exts = {".jpg", ".jpeg", ".png"}
    all_imgs = [p for p in args.image_dir.iterdir()
                if p.suffix.lower() in img_exts]

    if not all_imgs:
        print(f"[ERROR] No images found in {args.image_dir}")
        return

    chosen = random.sample(all_imgs, min(args.n_samples, len(all_imgs)))
    print(f"[GradCAM] Generating {len(chosen)} heatmaps → {args.output_dir}")
    generate_gradcam(model, chosen, transform, device, args.output_dir)
    print("[GradCAM] Done.")


if __name__ == "__main__":
    main()
