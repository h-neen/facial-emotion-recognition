"""
train.py
=========
Main EA-Net training script.

Stages:
  Stage 1 (--freeze_backbone): backbone frozen, train attention + head
  Stage 2 (resume from Stage 1): full fine-tuning at lower LR

Usage:
  # Stage 1
  python train.py --dataset fer --data_root data/FER2013_SR \
      --freeze_backbone --epochs 20 --batch_size 32 --lr 0.001

  # Stage 2
  python train.py --dataset fer --data_root data/FER2013_SR \
      --resume checkpoints/fer_best_stage1.pth \
      --epochs 30 --batch_size 32 --lr 0.0001
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


# ────────────────────────────────────────────────────────────────────────────
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


# ────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, use_amp
) -> dict:
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

    n = len(loader.dataset)
    metrics = tracker.compute()
    metrics["loss"] = running_loss / n
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

    n = len(loader.dataset)
    metrics = tracker.compute()
    metrics["loss"] = running_loss / n
    return metrics


# ────────────────────────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch, metrics, path: Path, extra: dict = None):
    obj = {
        "epoch":     epoch,
        "state":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics":   metrics,
    }
    if extra:
        obj.update(extra)
    torch.save(obj, path)
    print(f"  [ckpt] Saved → {path}")


def load_checkpoint(path: Path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[Resume] Loaded checkpoint from {path}  (epoch={ckpt.get('epoch', '?')})")
    return ckpt


# ────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    CFG.make_dirs()

    # ── Device ───────────────────────────────────────────────────────────
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = CFG.use_amp and device.type == "cuda" and not args.no_amp
    print(f"[Train] device={device}  AMP={use_amp}  dataset={args.dataset}")

    if device.type == "cuda":
        print(f"        GPU: {torch.cuda.get_device_name(0)}  "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data ─────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(
        args.dataset, data_root=args.data_root,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    class_weights = train_loader.class_weights.to(device)  # type: ignore

    # ── Model ────────────────────────────────────────────────────────────
    model = build_model(CFG, freeze_backbone=args.freeze_backbone).to(device)

    # ── Loss (class-weighted cross-entropy to handle FER imbalance) ──────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Optimizer ────────────────────────────────────────────────────────
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=CFG.stage1_momentum,
        weight_decay=CFG.stage1_weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=CFG.lr_step_size, gamma=CFG.lr_gamma
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Resume ───────────────────────────────────────────────────────────
    start_epoch  = 0
    if args.resume and args.resume.exists():
        ckpt = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = ckpt.get("epoch", 0)
        # Unfreeze backbones for Stage 2 if resuming
        if not args.freeze_backbone:
            model.unfreeze_backbones()
            # Rebuild optimizer with Differential Learning Rates
            backbone_params = list(model.efficient.parameters()) + \
                              list(model.inception.parameters())
                              
            head_params = list(model.cam.parameters()) + \
                          list(model.sam.parameters()) + \
                          list(model.classifier.parameters())

            optimizer = torch.optim.SGD([
                {'params': backbone_params, 'lr': args.lr * 0.1},  # 1e-5 for backbones
                {'params': head_params,     'lr': args.lr}         # 1e-4 for attention/head
            ], momentum=CFG.stage2_momentum, weight_decay=CFG.stage2_weight_decay, nesterov=True)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=CFG.lr_step_size, gamma=CFG.lr_gamma
            )

    # ── TensorBoard ──────────────────────────────────────────────────────
    run_name  = f"{args.dataset}_{'frozen' if args.freeze_backbone else 'full'}_{args.tag}"
    writer    = SummaryWriter(log_dir=str(CFG.log_dir / run_name))
    ckpt_pref = f"{args.dataset}_{args.tag}" if args.tag else args.dataset

    # ── Training loop ────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_val_f1  = 0.0
    patience_ctr = 0

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{start_epoch + args.epochs}  "
              f"LR={scheduler.get_last_lr()[0]:.6f}")

        train_m = train_one_epoch(model, train_loader, criterion,
                                  optimizer, scaler, device, use_amp)
        val_m   = validate(model, val_loader, criterion, device, use_amp)

        scheduler.step()

        print_metrics(train_m, prefix="  [train] ")
        print_metrics(val_m,   prefix="  [val]   ")
        print(f"  time: {time.time()-t0:.1f}s")

# ── Send Phone Notification ─────────────────────────────────────
        try:
            msg = f"Epoch {epoch} | Val Acc: {val_m['accuracy']:.2f}% | Val F1: {val_m['f1']:.2f}%"
            requests.post("https://ntfy.sh/fer_laptop", data=msg.encode('utf-8'))
        except Exception:
            pass  # If the wifi drops, don't crash the training!

        # ── Log to TensorBoard ──────────────────────────────────────────
        for k, v in train_m.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_m.items():
            writer.add_scalar(f"val/{k}",   v, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        # ── Checkpoint: best val accuracy ───────────────────────────────
        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            save_checkpoint(model, optimizer, epoch, val_m,
                            CFG.checkpoint_dir / f"{ckpt_pref}_best_val_acc.pth")
            patience_ctr = 0
        else:
            patience_ctr += 1

        # ── Checkpoint: best val F1 ─────────────────────────────────────
        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            save_checkpoint(model, optimizer, epoch, val_m,
                            CFG.checkpoint_dir / f"{ckpt_pref}_best_val_f1.pth")

        # ── Periodic checkpoint ─────────────────────────────────────────
        if epoch % CFG.save_every_n_epochs == 0:
            save_checkpoint(model, optimizer, epoch, val_m,
                            CFG.checkpoint_dir / f"{ckpt_pref}_epoch{epoch:03d}.pth")

        # ── Early stopping ──────────────────────────────────────────────
        if patience_ctr >= CFG.early_stop_patience:
            print(f"\n[EarlyStop] No improvement for {CFG.early_stop_patience} epochs. Stopping.")
            break

    writer.close()
    print(f"\n[Done] Best val accuracy: {best_val_acc:.2f}%  Best F1: {best_val_f1:.2f}%")


if __name__ == "__main__":
    main()
