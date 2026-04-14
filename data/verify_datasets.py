"""
data/verify_datasets.py
Run before preprocessing to catch folder name / image count issues early.
Usage: python data/verify_datasets.py --fer data/FER2013 --kdef data/KDEF
"""
import argparse, sys
from pathlib import Path
from collections import Counter
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import CFG

CLASS_NAMES = list(CFG.class_names)
IMG_EXTS = {".jpg", ".jpeg", ".png"}

EXPECTED_FER = {
    "train":      {"angry":3995,"disgust":436,"fear":4097,"happy":7215,
                   "neutral":4965,"sad":4830,"surprise":3171},
    "validation": {"angry":958,"disgust":111,"fear":1024,"happy":1774,
                   "neutral":1233,"sad":1247,"surprise":831},
}


def check_folder(path: Path, label: str) -> bool:
    ok = True
    found = {}
    for cls in CLASS_NAMES:
        d = path / cls
        if not d.exists():
            print(f"  [FAIL] Missing folder: {d}")
            ok = False
            continue
        imgs = [f for f in d.iterdir() if f.suffix.lower() in IMG_EXTS]
        found[cls] = len(imgs)
        # spot-check first image is readable
        if imgs:
            try:
                img = Image.open(imgs[0]).convert("RGB")
                _ = img.size
            except Exception as e:
                print(f"  [WARN] Cannot open {imgs[0]}: {e}")
    print(f"  [{label}] " + "  ".join(f"{c}={found.get(c,0)}" for c in CLASS_NAMES))
    return ok


def verify_fer(fer_root: Path) -> bool:
    print(f"\n[FER2013] Root: {fer_root}")
    ok = True
    for split in ("train", "validation"):
        sp = fer_root / split
        if not sp.exists():
            print(f"  [FAIL] Missing split folder: {sp}"); ok = False; continue
        ok &= check_folder(sp, f"FER {split}")
    return ok


def verify_kdef(kdef_root: Path) -> bool:
    print(f"\n[KDEF] Root: {kdef_root}")
    return check_folder(kdef_root, "KDEF")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fer",  default="data/FER2013",  help="Path to FER2013 root")
    ap.add_argument("--kdef", default="data/KDEF",     help="Path to KDEF root")
    args = ap.parse_args()

    fer_ok  = verify_fer(Path(args.fer))
    kdef_ok = verify_kdef(Path(args.kdef))

    if fer_ok and kdef_ok:
        print("\n[OK] All dataset checks passed. Safe to run preprocessing.")
    else:
        print("\n[FAIL] Fix the issues above before preprocessing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
