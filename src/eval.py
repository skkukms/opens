"""
Evaluation and inference script.

Validate on VOC2012 val:
  python src/eval.py --mode val --voc-root /data/VOC --ckpt checkpoints/best.pth

Inference on test images (submit/img → submit/pred):
  python src/eval.py --mode infer --ckpt checkpoints/best.pth \
      --img-dir submit/img --pred-dir submit/pred

Measure GFLOPs:
  python src/eval.py --mode flops --ckpt checkpoints/best.pth
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import VOCSegDataset, SegTransform
from src.model   import SegModel
from src.utils   import MIoUMeter, load_checkpoint, measure_gflops


MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def preprocess(img: Image.Image, size: int = 512):
    """Resize (bilinear), to tensor, normalize."""
    img = TF.resize(img, [size, size], interpolation=InterpolationMode.BILINEAR,
                    antialias=True)
    t = TF.to_tensor(img)
    return TF.normalize(t, MEAN, STD).unsqueeze(0)   # (1, 3, H, W)


@torch.no_grad()
def infer_image(model, img: Image.Image, orig_h: int, orig_w: int,
                device, crop_size: int = 512) -> np.ndarray:
    """Return (H, W) uint8 class-index array at original resolution."""
    x = preprocess(img, crop_size).to(device)
    with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
        logit = model(x)                             # (1, C, crop, crop)
    # upsample to original size
    logit = torch.nn.functional.interpolate(
        logit, size=(orig_h, orig_w), mode="bilinear", align_corners=False
    )
    pred = logit.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


# ---------------------------------------------------------------------------
# Val
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_val(model, voc_root: str, crop_size: int, batch_size: int,
            workers: int, device):
    val_tf = SegTransform(train=False, crop_size=crop_size)
    val_ds = VOCSegDataset(voc_root, "2012", "val", val_tf)
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)

    meter = MIoUMeter()
    model.eval()

    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(imgs)
        preds = logits.argmax(dim=1)
        meter.update(preds, masks)

    results = meter.compute()
    print(f"\nmIoU: {results['mIoU']:.4f}")
    print("\nPer-class IoU:")
    for cls, iou in results["per_class_iou"].items():
        print(f"  {cls:<14s}: {iou:.4f}" if not np.isnan(iou) else f"  {cls:<14s}: N/A")
    return results


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_infer(model, img_dir: str, pred_dir: str, device, crop_size: int = 512):
    os.makedirs(pred_dir, exist_ok=True)
    model.eval()

    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".jpg"))
    if not files:
        print("No .jpg files found in", img_dir)
        return

    for fname in files:
        img_path = os.path.join(img_dir, fname)
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        pred = infer_image(model, img, orig_h, orig_w, device, crop_size)

        out_name = os.path.splitext(fname)[0] + ".png"
        Image.fromarray(pred).save(os.path.join(pred_dir, out_name))

    print(f"Saved {len(files)} predictions to {pred_dir}")


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------

def run_flops(model, device):
    gflops = measure_gflops(model, device, input_size=(1, 3, 480, 640))
    print(f"GFLOPs @ 3×480×640: {gflops:.3f}")
    score = 5 if gflops <= 15 else (4 if gflops <= 50 else (3 if gflops <= 200 else 0))
    print(f"FLOPs score: {score}/5")
    return gflops


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",       choices=["val", "infer", "flops"], required=True)
    p.add_argument("--ckpt",       required=True)
    p.add_argument("--voc-root",   default=None)
    p.add_argument("--img-dir",    default="submit/img")
    p.add_argument("--pred-dir",   default="submit/pred")
    p.add_argument("--num-classes",type=int, default=21)
    p.add_argument("--aspp-ch",    type=int, default=128)
    p.add_argument("--crop-size",  type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers",    type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SegModel(num_classes=args.num_classes, pretrained=False,
                     aspp_ch=args.aspp_ch).to(device)
    load_checkpoint(args.ckpt, model)
    model.eval()

    if args.mode == "val":
        assert args.voc_root, "--voc-root required for val mode"
        run_val(model, args.voc_root, args.crop_size,
                args.batch_size, args.workers, device)

    elif args.mode == "infer":
        run_infer(model, args.img_dir, args.pred_dir, device, args.crop_size)

    elif args.mode == "flops":
        run_flops(model, device)


if __name__ == "__main__":
    main()
