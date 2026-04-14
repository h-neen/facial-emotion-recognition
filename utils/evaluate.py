"""
utils/evaluate.py
==================
Test-set evaluation: classification report, confusion matrix, per-class CSV.

Usage:
    python utils/evaluate.py \
        --checkpoint checkpoints/fer_best_val_acc.pth \
        --dataset fer \
        --data_root data/FER2013_SR
"""

import argparse, sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import CFG
from utils.metrics import compute_metrics, full_classification_report
from data.dataset_loader import get_dataloaders
from models.ea_net import build_model


def evaluate_model(model, loader, device, use_amp=True):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def save_confusion_matrix(y_true, y_pred, class_names, save_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title("EA-Net Confusion Matrix",  fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [saved] {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--dataset",    required=True, choices=["fer", "kdef"])
    ap.add_argument("--data_root",  type=Path, default=None)
    ap.add_argument("--batch_size", type=int,  default=CFG.batch_size)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CFG.make_dirs()

    _, _, test_loader = get_dataloaders(
        args.dataset, data_root=args.data_root,
        batch_size=args.batch_size, num_workers=CFG.num_workers)

    model = build_model(CFG, freeze_backbone=False).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state"])
    print(f"[Eval] Loaded {args.checkpoint}")

    y_true, y_pred = evaluate_model(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred)

    print("\n── EA-Net Results ─────────────────────────────────")
    print(f"  Accuracy  : {metrics['accuracy']:.2f}%   (paper: {'78.60' if args.dataset=='fer' else '99.30'}%)")
    print(f"  Precision : {metrics['precision']:.2f}%   (paper: {'76.10' if args.dataset=='fer' else '99.61'}%)")
    print(f"  Recall    : {metrics['recall']:.2f}%   (paper: {'77.98' if args.dataset=='fer' else '98.65'}%)")
    print(f"  F1-Score  : {metrics['f1']:.2f}%   (paper: {'77.98' if args.dataset=='fer' else '99.13'}%)")
    print("────────────────────────────────────────────────────\n")

    report = full_classification_report(y_true, y_pred, list(CFG.class_names))
    print(report)

    tag = args.dataset
    report_path = CFG.results_dir / f"classification_report_{tag}.txt"
    report_path.write_text(report)

    cm_path = CFG.results_dir / f"confusion_matrix_{tag}.png"
    save_confusion_matrix(y_true, y_pred, list(CFG.class_names), cm_path)

    csv_path = CFG.results_dir / f"summary_metrics_{tag}.csv"
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    print(f"  [saved] {csv_path}")

    print(f"\n[Eval] Results saved to {CFG.results_dir}/")


if __name__ == "__main__":
    main()
