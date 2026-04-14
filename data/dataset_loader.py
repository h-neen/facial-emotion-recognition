"""
data/dataset_loader.py — FER2013 + KDEF Dataset + DataLoader factory
"""
import sys
from pathlib import Path
from typing import Literal, Tuple, List, Optional
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import CFG
from preprocessing.transforms import get_train_transforms, get_val_transforms

CLASS_NAMES = list(CFG.class_names)
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}


class FERDataset(Dataset):
    """FER2013 from class-folder layout: root/train|validation/classname/img"""

    def __init__(self, root: Path, split: str, transform=None):
        self.root = Path(root) / split
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        for cls in CLASS_NAMES:
            d = self.root / cls
            if not d.exists():
                raise FileNotFoundError(f"Missing: {d}. Run verify_datasets.py first.")
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((p, CLASS_TO_IDX[cls]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label

    @property
    def targets(self): return [s[1] for s in self.samples]


class KDEFDataset(Dataset):
    """KDEF from flat class folders; auto stratified 70/15/15 split."""

    def __init__(self, root: Path, split: str, transform=None,
                 train_r=0.70, val_r=0.15):
        self.transform = transform
        all_p, all_l = [], []
        for cls in CLASS_NAMES:
            d = Path(root) / cls
            if not d.exists():
                raise FileNotFoundError(f"Missing: {d}")
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    all_p.append(p); all_l.append(CLASS_TO_IDX[cls])

        test_r = 1 - train_r - val_r
        Xtr, Xrst, ytr, yrst = train_test_split(
            all_p, all_l, test_size=(val_r + test_r),
            stratify=all_l, random_state=CFG.random_seed)
        Xv, Xte, yv, yte = train_test_split(
            Xrst, yrst, test_size=(test_r / (val_r + test_r)),
            stratify=yrst, random_state=CFG.random_seed)

        chosen = {"train": (Xtr, ytr), "val": (Xv, yv), "test": (Xte, yte)}[split]
        self.samples = list(zip(*chosen))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label

    @property
    def targets(self): return [s[1] for s in self.samples]


def get_dataloaders(dataset: str, data_root=None,
                    batch_size=CFG.batch_size, num_workers=CFG.num_workers):
    """Returns (train_loader, val_loader, test_loader). Attaches class_weights to train_loader."""
    tr_tf = get_train_transforms(CFG.image_size)
    val_tf = get_val_transforms(CFG.image_size)

    if dataset == "fer":
        root = data_root or CFG.fer_sr_root
        tr_ds  = FERDataset(root, "train",      tr_tf)
        val_ds = FERDataset(root, "validation", val_tf)
        te_ds  = val_ds
    else:
        root = data_root or CFG.kdef_aug_root
        tr_ds  = KDEFDataset(root, "train", tr_tf)
        val_ds = KDEFDataset(root, "val",   val_tf)
        te_ds  = KDEFDataset(root, "test",  val_tf)

    print(f"[DataLoader] {dataset.upper()}: train={len(tr_ds):,} val={len(val_ds):,} test={len(te_ds):,}")

    counts = np.bincount(np.array(tr_ds.targets), minlength=CFG.num_classes).astype(float)
    cw = torch.tensor(counts.sum() / (CFG.num_classes * counts), dtype=torch.float32)
    sw = torch.tensor(1.0 / counts[np.array(tr_ds.targets)], dtype=torch.float32)
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

    mkl = lambda ds, shuf, samp: DataLoader(ds, batch_size=batch_size, sampler=samp,
        shuffle=(shuf and samp is None), num_workers=num_workers,
        pin_memory=CFG.pin_memory, drop_last=(samp is not None))

    tl  = mkl(tr_ds,  True,  sampler)
    vl  = mkl(val_ds, False, None)
    tel = mkl(te_ds,  False, None)
    tl.class_weights = cw  # type: ignore
    return tl, vl, tel
