"""
preprocessing/transforms.py
============================
On-the-fly augmentation transforms for training.
Uses albumentations for geometric/colour transforms,
with a custom LocalContrastEnhance implementing Paper Eq. 2a/2b.

get_train_transforms() — used by training DataLoader
get_val_transforms()   — used by validation / test DataLoader (no augmentation)
"""

import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms.functional as TF


# ────────────────────────────────────────────────────────────────────────────
class LocalContrastEnhance(A.ImageOnlyTransform):
    """
    Implements Paper Equations 2a (local sharpening) and 2b (global stretch).

    gs(x,y) = c * f(x,y) - a * m(x,y) + b * sigma(x,y)   [Eq. 2a]
    gb(x,y) = ((gs(x,y) - L) / (H - L)) ^ d               [Eq. 2b]

    where:
        f      = input image
        m      = local mean (Gaussian blur)
        sigma  = local std  (Gaussian blur of squared diff)
        L, H   = 1% / 99% percentiles of gs
        a,b,c,d = tunable constants (defaults match paper behavior)
    """

    def __init__(self, a=1.0, b=0.5, c=1.5, d=0.8,
                 kernel_size=5, p=0.5):
        super().__init__(p=p)
        self.a = a; self.b = b; self.c = c; self.d = d
        self.ks = kernel_size

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        import cv2
        f = img.astype(np.float32) / 255.0
        # Local mean and std via Gaussian blur
        m     = cv2.GaussianBlur(f, (self.ks, self.ks), 0)
        diff  = (f - m) ** 2
        sigma = np.sqrt(cv2.GaussianBlur(diff, (self.ks, self.ks), 0) + 1e-8)
        # Eq. 2a: local sharpening
        gs = self.c * f - self.a * m + self.b * sigma
        # Eq. 2b: global stretch with 1% tolerance
        L = np.percentile(gs, 1)
        H = np.percentile(gs, 99)
        denom = H - L if (H - L) > 1e-8 else 1e-8
        gb = np.clip((gs - L) / denom, 0.0, 1.0) ** self.d
        return (gb * 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("a", "b", "c", "d", "kernel_size")


class GrayToRGB(A.ImageOnlyTransform):
    """
    Randomly convert image to grayscale then back to 3-channel RGB.
    Implements the 'de-colorize' augmentation from the paper (Section 2.1.1).
    """
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    def get_transform_init_args_names(self):
        return ()


# ────────────────────────────────────────────────────────────────────────────
def get_train_transforms(image_size: int = 192) -> "callable":
    aug = A.Compose([
        # 1. Apply all geometric "size-changing" augmentations first
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.4),
        A.RandomScale(scale_limit=0.15, p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.0, rotate_limit=0, p=0.3),
        
        # 2. Apply color/contrast
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        LocalContrastEnhance(a=1.0, b=0.5, c=1.5, d=0.8, p=0.3),
        GrayToRGB(p=0.15),

        # 3. CRITICAL: Resize must be here, AFTER all the above
        A.Resize(image_size, image_size), 

        # 4. Final step: Normalize and Tensor
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    def transform(pil_img: Image.Image):
        np_img = np.array(pil_img.convert("RGB"))
        return aug(image=np_img)["image"]

    return transform

    def transform(pil_img: Image.Image):
        np_img = np.array(pil_img.convert("RGB"))
        return aug(image=np_img)["image"]

    return transform


def get_val_transforms(image_size: int = 192) -> "callable":
    """
    Validation / test transform — resize + normalise only (no augmentation).
    """
    aug = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    def transform(pil_img: Image.Image):
        np_img = np.array(pil_img.convert("RGB"))
        return aug(image=np_img)["image"]

    return transform
