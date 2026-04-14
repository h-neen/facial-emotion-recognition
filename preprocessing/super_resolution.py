"""
preprocessing/super_resolution.py
===================================
Offline 4x super-resolution using LapSRN via OpenCV DNN module.
Upscales FER2013 images from 48x48 -> 192x192.

Usage:
    python preprocessing/super_resolution.py \
        --input_dir  data/FER2013/train \
        --output_dir data/FER2013_SR/train \
        --model_path preprocessing/LapSRN_x4.pb \
        --scale 4

Download LapSRN_x4.pb first:
    python -c "
    import urllib.request
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/fannymonori/TF-LapSRN/master/export/LapSRN_x4.pb',
        'preprocessing/LapSRN_x4.pb')
    print('Downloaded.')"
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

BATCH_SIZE = 50          # reduce to 20 if RAM spikes above 12 GB
IMG_EXTS   = {".jpg", ".jpeg", ".png"}
TARGET_SIZE = 192        # paper uses 192x192 after 4x SR from 48x48


def load_sr_model(model_path: str, scale: int = 4) -> cv2.dnn_superres.DnnSuperResImpl:
    """Load the LapSRN model into OpenCV DNN super-res backend."""
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    if not Path(model_path).exists():
        print(f"\n[ERROR] Model file not found: {model_path}")
        print("Download it with:")
        print("  python -c \"import urllib.request; urllib.request.urlretrieve(")
        print("  'https://raw.githubusercontent.com/fannymonori/TF-LapSRN/master/export/LapSRN_x4.pb',")
        print("  'preprocessing/LapSRN_x4.pb')\"")
        sys.exit(1)
    sr.readModel(model_path)
    sr.setModel("lapsrn", scale)
    print(f"[SR] Loaded LapSRN x{scale} from {model_path}")
    return sr


def upscale_image(
    sr_model: cv2.dnn_superres.DnnSuperResImpl,
    img_bgr: np.ndarray,
    target_size: int = TARGET_SIZE,
) -> np.ndarray:
    """
    Upscale a single BGR image.
    - If input is 48x48, run LapSRN (4x -> 192x192)
    - Otherwise bicubic-resize to target_size directly
    """
    h, w = img_bgr.shape[:2]
    if h == 48 and w == 48:
        upscaled = sr_model.upsample(img_bgr)           # LapSRN 4x
    else:
        upscaled = cv2.resize(img_bgr, (target_size, target_size),
                              interpolation=cv2.INTER_CUBIC)
    # Ensure exactly target_size x target_size
    if upscaled.shape[0] != target_size or upscaled.shape[1] != target_size:
        upscaled = cv2.resize(upscaled, (target_size, target_size),
                              interpolation=cv2.INTER_CUBIC)
    return upscaled


def upscale_dir(
    input_dir:  Path,
    output_dir: Path,
    sr_model:   cv2.dnn_superres.DnnSuperResImpl,
    target_size: int = TARGET_SIZE,
) -> None:
    """
    Recursively upscale all images in input_dir, preserving subfolder structure.
    Skips images that already exist in output_dir.
    """
    img_paths = sorted([
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in IMG_EXTS
    ])

    if not img_paths:
        print(f"[WARN] No images found in {input_dir}")
        return

    print(f"[SR] Processing {len(img_paths):,} images from {input_dir}")

    skipped = 0
    errors  = 0
    for path in tqdm(img_paths, desc=str(input_dir.name), unit="img"):
        # Preserve relative folder structure
        rel  = path.relative_to(input_dir)
        dest = output_dir / rel.with_suffix(".png")
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            skipped += 1
            continue

        try:
            img_bgr = cv2.imread(str(path))
            if img_bgr is None:
                raise ValueError(f"cv2.imread returned None for {path}")
            # Handle grayscale FER images (single channel)
            if len(img_bgr.shape) == 2:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
            elif img_bgr.shape[2] == 1:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

            out = upscale_image(sr_model, img_bgr, target_size)
            cv2.imwrite(str(dest), out)

        except Exception as e:
            errors += 1
            print(f"\n[ERROR] {path}: {e}")

    print(f"[SR] Done. Processed={len(img_paths)-skipped-errors:,}  "
          f"Skipped={skipped:,}  Errors={errors:,}")


def run_lapsrn(input_dir: Path, output_dir: Path, model_path: str, scale: int = 4):
    """Entry point — load model and run SR on a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sr = load_sr_model(model_path, scale)
    upscale_dir(input_dir, output_dir, sr)
    print(f"\n[SR] All images saved to: {output_dir}")


def main():
    ap = argparse.ArgumentParser(description="LapSRN offline super-resolution")
    ap.add_argument("--input_dir",  required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--model_path", default="preprocessing/LapSRN_x4.pb")
    ap.add_argument("--scale",      default=4, type=int)
    args = ap.parse_args()

    run_lapsrn(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
