"""
train.py
=========
Main EA-Net training script.
"""

import argparse
import sys
import time
from pathlib import Path
import requests

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.config import CFG
from utils.metrics import MetricsTracker, print_metrics
from data.dataset_loader import get_dataloaders
from models.ea_net import build_model


def parse_args():
    ap = argparse.ArgumentParser(description="Train EA-Net")
    ap.add_argument("--dataset",         required=True, choices=["fer", "kdef"])
    ap.add_argument("--data_root",       type=Path, default=None)
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--resume",          type=Path, default=None)
    ap.add_argument("--epochs",          type=int,   default=CFG.stage1_epochs)
    ap.add_argument("--batch_size",      type=int,   default=CFG.batch_size)
    ap.add_argument("--lr",              type=float, default=CFG.stage1_lr)
    ap.add_argument("--num_workers",     type=int,   default=CFG.num_workers)
    ap.add_argument("--no_amp",          action="store_true", help="Disable AMP")
    ap.add_argument("--tag",             type=str,   default="",
                    help="Optional run tag appended to checkpoint names")
    return ap.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp) -> dict:
    model.train()
    tracker = MetricsTracker()
    running_loss = 0.0

    for imgs, labels in tqdm(loader, desc="  train", leave=False, unit="batch"):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        tracker.update(logits.detach(), labels.detach())

    metrics = tracker.compute()
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp) -> dict:
    model.eval()
    tracker = MetricsTracker()
    running_loss = 0.0

    for imgs, labels in tqdm(loader, desc="  val  ", leave=False, unit="batch"):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        tracker.update(logits, labels)

    metrics = tracker.compute()
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics


def save_checkpoint(model, optimizer, epoch, metrics, path: Path, extra: dict = None):
    obj = {
        "epoch":     epoch,
        "state":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics":   metrics,
    }
    if extra: obj.update(extra)
    torch.save(obj, path)
    print(f"  [ckpt] Saved → {path}")


def load_checkpoint(path: Path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state"])
    # Only load optimizer if it's provided and exists in checkpoint
    if optimizer and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError:
            print("[Warning] Optimizer shape mismatch. Skipping optimizer state load.")
    print(f"[Resume] Loaded checkpoint from {path} (epoch={ckpt.get('epoch', '?')})")
    return ckpt


def main():
    args = parse_args()
    CFG.make_dirs()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = CFG.use_amp and device.type == "cuda" and not args.no_amp

    print(f"[Train] device={device}  AMP={use_amp}  dataset={args.dataset}")

    train_loader, val_loader, _ = get_dataloaders(
        args.dataset, data_root=args.data_root,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    model = build_model(CFG, freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Initial Optimizer Setup
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=CFG.stage1_momentum,
        weight_decay=CFG.stage1_weight_decay,
        nesterov=True,
    )
    
    start_epoch = 0
    if args.resume and args.resume.exists():
        # CRITICAL CHANGE: Pass None for optimizer during Stage 2 resume to avoid group mismatch
        opt_to_load = None if not args.freeze_backbone else optimizer
        ckpt = load_checkpoint(args.resume, model, opt_to_load, device)
        start_epoch = ckpt.get("epoch", 0)

        if not args.freeze_backbone:
            model.unfreeze_backbones()
            # Rebuild with Differential Learning Rates for Stage 2
            backbone_params = list(model.efficient.parameters()) + list(model.inception.parameters())
            head_params = list(model.cam.parameters()) + list(model.sam.parameters()) + list(model.classifier.parameters())

            optimizer = torch.optim.SGD([
                {'params': backbone_params, 'lr': args.lr * 0.1},
                {'params': head_params,     'lr': args.lr}
            ], momentum=CFG.stage2_momentum, weight_decay=CFG.stage2_weight_decay, nesterov=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=10,        # Restart the learning rate every 10 epochs
                T_mult=1,      # Keep the restart interval constant
                eta_min=1e-6   # Don't let the LR drop below this floor
            )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    writer = SummaryWriter(log_dir=str(CFG.log_dir / f"{args.dataset}_{'frozen' if args.freeze_backbone else 'full'}_{args.tag}"))

    best_val_acc = 0.0
    best_val_f1  = 0.0
    patience_ctr = 0

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{start_epoch + args.epochs}  LR={scheduler.get_last_lr()[0]:.6f}")

        train_m = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_m   = validate(model, val_loader, criterion, device, use_amp)
        scheduler.step()

        print_metrics(train_m, prefix="  [train] ")
        print_metrics(val_m,   prefix="  [val]   ")
        print(f"  time: {time.time()-t0:.1f}s")

        try:
            # Constructing the exact terminal output format
            msg = (
                f"Epoch {epoch}/{start_epoch + args.epochs} | LR={scheduler.get_last_lr()[0]:.6f}\n"
                f"[train] Acc={train_m['accuracy']:.2f}%  Prec={train_m['precision']:.2f}%  Rec={train_m['recall']:.2f}%  F1={train_m['f1']:.2f}%\n"
                f"[val]   Acc={val_m['accuracy']:.2f}%  Prec={val_m['precision']:.2f}%  Rec={val_m['recall']:.2f}%  F1={val_m['f1']:.2f}%\n"
                f"time: {time.time()-t0:.1f}s"
            )
            requests.post("https://ntfy.sh/fer_laptop", data=msg.encode('utf-8'))
        except Exception:
            pass  # Silently fail if wifi drops

        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            save_checkpoint(model, optimizer, epoch, val_m, CFG.checkpoint_dir / f"{args.dataset}_best_val_acc.pth")
            patience_ctr = 0
        else: patience_ctr += 1

        if patience_ctr >= CFG.early_stop_patience:
            print(f"\n[EarlyStop] Stopping.")
            break

    writer.close()

if __name__ == "__main__":
    main()