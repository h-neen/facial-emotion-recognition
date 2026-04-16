"""
utils/evaluate.py
==================
Enhanced evaluation: Standard vs TTA accuracy, classification report, 
confusion matrix, and per-class CSV.
"""

import argparse, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
    all_labels = []
    all_preds_std = []  # Standard predictions
    all_preds_tta = []  # TTA (Flipped) predictions
    
    print(f"  [Eval] Running dual inference (Standard + TTA) on {len(loader.dataset)} images...")
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                # 1. Standard Forward Pass
                logits_orig = model(imgs)
                probs_orig = F.softmax(logits_orig, dim=1)
                
                # 2. TTA Forward Pass (Horizontal Flip)
                imgs_flipped = torch.flip(imgs, dims=[3])
                logits_flip = model(imgs_flipped)
                probs_flip = F.softmax(logits_flip, dim=1)
                
                # 3. Average the probabilities
                avg_probs = (probs_orig + probs_flip) / 2
            
            # Record results
            all_preds_std.extend(probs_orig.argmax(dim=1).cpu().numpy())
            all_preds_tta.extend(avg_probs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
            
    return np.array(all_labels), np.array(all_preds_std), np.array(all_preds_tta)

def save_confusion_matrix(y_true, y_pred, class_names, save_path: Path, title_suffix=""):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",       fontsize=11)
    ax.set_title(f"EA-Net Confusion Matrix {title_suffix}", fontsize=13)
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
    print(f"[Eval] Loaded {args.checkpoint} (Epoch: {ckpt.get('epoch', 'N/A')})")

    # Run Dual Evaluation
    y_true, y_std, y_tta = evaluate_model(model, test_loader, device)
    
    metrics_std = compute_metrics(y_true, y_std)
    metrics_tta = compute_metrics(y_true, y_tta)

    print("\n── EA-Net Comparison Results ──────────────────────")
    print(f"  Method      | Accuracy | Precision | Recall | F1-Score")
    print(f"  ────────────|──────────|───────────|────────|─────────")
    print(f"  Standard    | {metrics_std['accuracy']:.2f}%  | {metrics_std['precision']:.2f}%   | {metrics_std['recall']:.2f}% | {metrics_std['f1']:.2f}%")
    print(f"  With TTA    | {metrics_tta['accuracy']:.2f}%  | {metrics_tta['precision']:.2f}%   | {metrics_tta['recall']:.2f}% | {metrics_tta['f1']:.2f}%")
    
    boost = metrics_tta['accuracy'] - metrics_std['accuracy']
    print(f"  ────────────|──────────|───────────|────────|─────────")
    print(f"  TTA Boost   | +{boost:.2f}%")
    print("────────────────────────────────────────────────────\n")

    # Save outputs for the better performing version (usually TTA)
    tag = args.dataset
    report = full_classification_report(y_true, y_tta, list(CFG.class_names))
    report_path = CFG.results_dir / f"classification_report_{tag}_tta.txt"
    report_path.write_text(report)

    cm_path = CFG.results_dir / f"confusion_matrix_{tag}_tta.png"
    save_confusion_matrix(y_true, y_tta, list(CFG.class_names), cm_path, title_suffix="(TTA)")

    csv_path = CFG.results_dir / f"summary_metrics_{tag}.csv"
    results_df = pd.DataFrame([
        {"Method": "Standard", **metrics_std},
        {"Method": "TTA", **metrics_tta}
    ])
    results_df.to_csv(csv_path, index=False)
    
    print(f"[Eval] All reports and matrices saved to {CFG.results_dir}/")

if __name__ == "__main__":
    main()