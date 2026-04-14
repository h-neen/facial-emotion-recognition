"""
preprocessing/augment_kdef.py
==============================
Offline augmentation for the KDEF dataset.
Expands each class toward target_per_class using Paper Eq. 1, 2a, 2b
plus geometric transforms. Existing images are copied as-is.

Usage:
    python preprocessing/augment_kdef.py \
        --input_dir  data/KDEF \
        --output_dir data/KDEF_AUG \
        --target_per_class 840
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import CFG

CLASS_NAMES = list(CFG.class_names)
IMG_EXTS    = {".jpg", ".jpeg", ".png"}
TARGET_SIZE = CFG.image_size   # 192


# ────────────────────────────────────────────────────────────────────────────
# Paper Equations 2a / 2b — local contrast enhancement
def local_contrast_enhance(
    img: np.ndarray,
    a: float = 1.0, b: float = 0.5, c: float = 1.5, d: float = 0.8,
    ks: int = 5,
) -> np.ndarray:
    f     = img.astype(np.float32) / 255.0
    m     = cv2.GaussianBlur(f, (ks, ks), 0)
    diff  = (f - m) ** 2
    sigma = np.sqrt(cv2.GaussianBlur(diff, (ks, ks), 0) + 1e-8)
    gs    = c * f - a * m + b * sigma                       # Eq. 2a
    L, H  = np.percentile(gs, 1), np.percentile(gs, 99)
    denom = max(H - L, 1e-8)
    gb    = np.clip((gs - L) / denom, 0, 1) ** d            # Eq. 2b
    return (gb * 255).astype(np.uint8)


def augment_one(img_bgr: np.ndarray) -> np.ndarray:
    """Apply a random subset of augmentation transforms to one image."""
    ops = random.sample([
        "flip", "rotate", "brightness", "contrast",
        "scale", "local_enhance", "grayscale",
    ], k=random.randint(2, 4))

    out = img_bgr.copy()
    h, w = out.shape[:2]

    for op in ops:
        if op == "flip":
            out = cv2.flip(out, 1)

        elif op == "rotate":
            angle = random.uniform(-20, 20)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            out = cv2.warpAffine(out, M, (w, h),
                                 borderMode=cv2.BORDER_REFLECT)

        elif op == "brightness":
            beta = random.randint(-40, 40)
            out  = cv2.convertScaleAbs(out, alpha=1.0, beta=beta)

        elif op == "contrast":
            alpha = random.uniform(0.8, 1.4)
            out   = cv2.convertScaleAbs(out, alpha=alpha, beta=0)

        elif op == "scale":
            s  = random.uniform(0.85, 1.15)
            nw, nh = int(w * s), int(h * s)
            tmp = cv2.resize(out, (nw, nh))
            out = cv2.resize(tmp, (w, h))   # back to original size

        elif op == "local_enhance":
            out = local_contrast_enhance(out)

        elif op == "grayscale":
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            out  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return out


# ────────────────────────────────────────────────────────────────────────────
def augment_dataset(
    input_dir:  Path,
    output_dir: Path,
    target_per_class: int = 840,
    seed: int = CFG.random_seed,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    for cls in CLASS_NAMES:
        src_dir = input_dir / cls
        dst_dir = output_dir / cls
        dst_dir.mkdir(parents=True, exist_ok=True)

        src_imgs = sorted([
            p for p in src_dir.iterdir()
            if p.suffix.lower() in IMG_EXTS
        ])
        n_orig = len(src_imgs)
        print(f"  [{cls}] original={n_orig}  target={target_per_class}")

        # Copy originals (resize to TARGET_SIZE)
        for i, p in enumerate(tqdm(src_imgs, desc=f"  copy {cls}", leave=False)):
            out = dst_dir / f"orig_{i:04d}.png"
            if not out.exists():
                img = cv2.imread(str(p))
                if img is None: continue
                img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))
                cv2.imwrite(str(out), img)

        # Generate augmented images to reach target
        n_needed = max(0, target_per_class - n_orig)
        aug_idx  = 0
        for _ in tqdm(range(n_needed), desc=f"  aug  {cls}", leave=False):
            src = random.choice(src_imgs)
            img = cv2.imread(str(src))
            if img is None: continue
            img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))
            aug = augment_one(img)
            out = dst_dir / f"aug_{aug_idx:05d}.png"
            cv2.imwrite(str(out), aug)
            aug_idx += 1

        total = len(list(dst_dir.glob("*.png")))
        print(f"  [{cls}] saved {total} images to {dst_dir}")

    print(f"\n[AUG] Complete. Output: {output_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir",         required=True, type=Path)
    ap.add_argument("--output_dir",        required=True, type=Path)
    ap.add_argument("--target_per_class",  default=840,   type=int)
    ap.add_argument("--seed",              default=CFG.random_seed, type=int)
    args = ap.parse_args()
    augment_dataset(args.input_dir, args.output_dir,
                    args.target_per_class, args.seed)


if __name__ == "__main__":
    main()
