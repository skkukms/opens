"""
Training script for semantic segmentation (Project 1).

Phase 1 – COCO pretraining:
  python src/train.py --voc-root /data/VOC --coco-img-dir /data/coco/train2017 \
      --coco-ann /data/coco/annotations/instances_train2017.json \
      --epochs 30 --run-name mbv3l_aspp128_coco

Phase 2 – VOC fine-tuning:
  python src/train.py --voc-root /data/VOC \
      --epochs 50 --resume checkpoints/best.pth \
      --run-name mbv3l_aspp128_voc
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import build_datasets
from src.loss    import SegLoss
from src.model   import SegModel
from src.utils   import MIoUMeter, load_checkpoint, measure_gflops, save_checkpoint, set_seed


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def poly_factor(step: int, warmup: int, total: int, power: float = 0.9) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return max((1.0 - progress) ** power, 1e-6)


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scheduler,
                    scaler, device, epoch, args):
    model.train()
    total_loss = 0.0

    for step, (imgs, masks) in enumerate(loader):
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp):
            logits = model(imgs)
            loss   = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

        if step % args.log_interval == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"[E{epoch:03d} {step}/{len(loader)}] loss={loss.item():.4f} lr={lr_now:.6f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    meter = MIoUMeter()

    for imgs, masks in loader:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device)
        with autocast():
            logits = model(imgs)
        preds = logits.argmax(dim=1)
        meter.update(preds, masks)

    return meter.compute()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--voc-root",      required=True)
    p.add_argument("--coco-img-dir",  default=None)
    p.add_argument("--coco-ann",      default=None)
    p.add_argument("--no-voc07",      action="store_true")
    p.add_argument("--no-voc12",      action="store_true")
    p.add_argument("--crop-size",     type=int, default=512)
    p.add_argument("--workers",       type=int, default=4)
    # model
    p.add_argument("--num-classes",   type=int, default=21)
    p.add_argument("--aspp-ch",       type=int, default=128)
    p.add_argument("--no-pretrained", action="store_true")
    # training
    p.add_argument("--epochs",        type=int, default=50)
    p.add_argument("--batch-size",    type=int, default=16)
    p.add_argument("--base-lr",       type=float, default=0.01)
    p.add_argument("--backbone-lr-scale", type=float, default=0.1)
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--warmup-iters",  type=int, default=500)
    p.add_argument("--amp",           action="store_true", default=True)
    p.add_argument("--no-amp",        dest="amp", action="store_false")
    # misc
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--resume",        default=None)
    p.add_argument("--ckpt-dir",      default="checkpoints")
    p.add_argument("--log-interval",  type=int, default=50)
    # wandb
    p.add_argument("--run-name",      default=None)
    p.add_argument("--wandb-project", default="skku-semseg-p1")
    p.add_argument("--no-wandb",      action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ---- datasets ----
    train_ds, val_ds = build_datasets(
        voc_root      = args.voc_root,
        crop_size     = args.crop_size,
        coco_img_dir  = args.coco_img_dir,
        coco_ann_file = args.coco_ann,
        use_voc07     = not args.no_voc07,
        use_voc12     = not args.no_voc12,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # ---- model ----
    model = SegModel(num_classes=args.num_classes, pretrained=not args.no_pretrained,
                     aspp_ch=args.aspp_ch).to(device)

    # ---- optimizer (separate LR for backbone) ----
    optimizer = torch.optim.SGD(
        [
            {"params": model.backbone_params(),
             "lr": args.base_lr * args.backbone_lr_scale},
            {"params": model.head_params(),
             "lr": args.base_lr},
        ],
        momentum=0.9, weight_decay=args.weight_decay,
    )

    # ---- poly LR scheduler (per-iter) ----
    total_iters = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: poly_factor(step, args.warmup_iters, total_iters),
    )

    # ---- loss / scaler ----
    criterion = SegLoss(num_classes=args.num_classes)
    scaler    = GradScaler(enabled=args.amp)

    # ---- resume ----
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)
        print(f"Resumed from epoch {start_epoch}")

    # ---- WandB ----
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
        )
        # log GFLOPs once
        gflops = measure_gflops(model, device)
        wandb.config.update({"gflops": gflops})
        print(f"GFLOPs @ 3×480×640: {gflops:.2f}")
        model.to(device)

    # ---- training loop ----
    best_miou = 0.0

    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                   scheduler, scaler, device, epoch, args)

        val_metrics = validate(model, val_loader, device)
        miou = val_metrics["mIoU"]

        lr_now = optimizer.param_groups[1]["lr"]
        print(f"Epoch {epoch:03d} | loss={avg_loss:.4f} | mIoU={miou:.4f} | lr={lr_now:.6f}")

        log_dict = {
            "epoch": epoch,
            "train/loss": avg_loss,
            "val/mIoU": miou,
            "lr": lr_now,
        }
        for cls_name, cls_iou in val_metrics["per_class_iou"].items():
            log_dict[f"val/iou_{cls_name}"] = cls_iou

        if not args.no_wandb:
            wandb.log(log_dict)

        # save latest
        save_checkpoint(
            {"epoch": epoch + 1, "model": model.state_dict(),
             "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(),
             "mIoU": miou},
            os.path.join(args.ckpt_dir, "latest.pth"),
        )

        # save best
        if miou > best_miou:
            best_miou = miou
            save_checkpoint(
                {"epoch": epoch + 1, "model": model.state_dict(), "mIoU": miou},
                os.path.join(args.ckpt_dir, "best.pth"),
            )
            print(f"  *** New best mIoU: {best_miou:.4f} ***")

    if not args.no_wandb:
        wandb.finish()
    print(f"Done. Best mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
