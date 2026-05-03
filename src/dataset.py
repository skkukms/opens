import os
import random
import numpy as np
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter, InterpolationMode
from torchvision.datasets import VOCSegmentation, CocoDetection


# COCO category ID -> VOC class index (1-20)
COCO_TO_VOC: dict[int, int] = {
    5: 1,   # airplane    -> aeroplane
    2: 2,   # bicycle
    16: 3,  # bird
    9: 4,   # boat
    44: 5,  # bottle
    6: 6,   # bus
    3: 7,   # car
    17: 8,  # cat
    62: 9,  # chair
    21: 10, # cow
    67: 11, # dining table
    18: 12, # dog
    19: 13, # horse
    4: 14,  # motorcycle  -> motorbike
    1: 15,  # person
    64: 16, # potted plant
    20: 17, # sheep
    63: 18, # couch       -> sofa
    7: 19,  # train
    72: 20, # tv          -> tvmonitor
}


class SegTransform:
    """Synchronized image + mask transforms. Mask uses NEAREST, image uses BILINEAR."""

    def __init__(self, train: bool = True, crop_size: int = 512):
        self.train = train
        self.crop_size = crop_size
        self._jitter = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        self._mean = [0.485, 0.456, 0.406]
        self._std  = [0.229, 0.224, 0.225]

    def __call__(self, img: Image.Image, mask: Image.Image):
        if self.train:
            # --- random scale (aspect ratio 유지) ---
            scale = random.uniform(0.5, 2.0)
            orig_h, orig_w = img.height, img.width
            sh = int(orig_h * scale)
            sw = int(orig_w * scale)
            img  = TF.resize(img,  [sh, sw], interpolation=InterpolationMode.BILINEAR,
                             antialias=True)
            mask = TF.resize(mask, [sh, sw], interpolation=InterpolationMode.NEAREST)

            # --- pad if smaller than crop_size ---
            pw = max(self.crop_size - sw, 0)
            ph = max(self.crop_size - sh, 0)
            if pw > 0 or ph > 0:
                img  = TF.pad(img,  [0, 0, pw, ph], fill=0)
                mask = TF.pad(mask, [0, 0, pw, ph], fill=255)

            # --- random crop ---
            i, j, h, w = torch.randint(0, max(img.height - self.crop_size + 1, 1), (1,)).item(), \
                         torch.randint(0, max(img.width  - self.crop_size + 1, 1), (1,)).item(), \
                         self.crop_size, self.crop_size
            img  = TF.crop(img,  i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            # --- random horizontal flip ---
            if random.random() < 0.5:
                img  = TF.hflip(img)
                mask = TF.hflip(mask)

            # --- color jitter ---
            img = self._jitter(img)

            # --- random grayscale ---
            if random.random() < 0.1:
                img = TF.to_grayscale(img, num_output_channels=3)

            # --- gaussian blur (완화: p=0.2, kernel=11) ---
            if random.random() < 0.2:
                img = TF.gaussian_blur(img, kernel_size=11,
                                       sigma=random.uniform(0.1, 1.0))
        else:
            # --- val: aspect ratio 유지, longer side = crop_size, pad ---
            orig_h, orig_w = img.height, img.width
            scale = self.crop_size / max(orig_h, orig_w)
            new_h = int(orig_h * scale)
            new_w = int(orig_w * scale)
            img  = TF.resize(img,  [new_h, new_w],
                             interpolation=InterpolationMode.BILINEAR, antialias=True)
            mask = TF.resize(mask, [new_h, new_w],
                             interpolation=InterpolationMode.NEAREST)
            pw = self.crop_size - new_w
            ph = self.crop_size - new_h
            if pw > 0 or ph > 0:
                img  = TF.pad(img,  [0, 0, pw, ph], fill=0)
                mask = TF.pad(mask, [0, 0, pw, ph], fill=255)

        # --- to tensor ---
        img = TF.to_tensor(img)
        img = TF.normalize(img, self._mean, self._std)

        mask_np = np.array(mask, dtype=np.int64)
        mask_np[mask_np > 20] = 255   # suppress unknown labels (keep 0-20 and 255)
        mask_t = torch.from_numpy(mask_np)

        return img, mask_t


class VOCSegDataset(Dataset):
    def __init__(self, root: str, year: str, image_set: str, transform=None):
        self.ds = VOCSegmentation(root=root, year=year, image_set=image_set,
                                  download=True)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, mask = self.ds[idx]
        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask


def _ann_to_binary_mask(ann: dict, h: int, w: int) -> np.ndarray:
    """Rasterize one COCO segmentation annotation."""
    seg = ann.get("segmentation", [])
    tmp = Image.new("L", (w, h), 0)

    if isinstance(seg, list):
        draw = ImageDraw.Draw(tmp)
        for poly in seg:
            if len(poly) >= 6:
                pts = list(zip(poly[::2], poly[1::2]))
                draw.polygon(pts, fill=1)
        return np.array(tmp, dtype=bool)

    if isinstance(seg, dict):
        from pycocotools import mask as cm
        rle = seg
        if isinstance(rle.get("counts"), list):
            rle = cm.frPyObjects(rle, h, w)
        return cm.decode(rle).astype(bool)

    return np.zeros((h, w), dtype=bool)


def _coco_seg_mask(anns: list, h: int, w: int) -> Image.Image:
    """Convert COCO annotations to a VOC-style mask.

    VOC classes are mapped to labels 1-20. Annotated non-VOC objects and crowd
    regions are ignored instead of being taught as background.
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    # Draw largest area first so smaller objects appear on top.
    for ann in sorted(anns, key=lambda a: a.get("area", 0), reverse=True):
        try:
            ann_mask = _ann_to_binary_mask(ann, h, w)
        except ImportError:
            continue

        if not ann_mask.any():
            continue

        cat_id = ann["category_id"]
        if ann.get("iscrowd", 0) or cat_id not in COCO_TO_VOC:
            mask[ann_mask] = 255
        else:
            mask[ann_mask] = COCO_TO_VOC[cat_id]

    return Image.fromarray(mask)


class COCOSegDataset(Dataset):
    def __init__(self, img_dir: str, ann_file: str, transform=None):
        self.ds = CocoDetection(root=img_dir, annFile=ann_file)
        self.ds.ids = [
            img_id for img_id in self.ds.ids
            if any(
                ann.get("category_id") in COCO_TO_VOC and not ann.get("iscrowd", 0)
                for ann in self.ds.coco.imgToAnns.get(img_id, [])
            )
        ]
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, anns = self.ds[idx]
        w, h = img.size
        mask = _coco_seg_mask(anns, h, w)
        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask


def build_datasets(
    voc_root: str,
    crop_size: int = 512,
    coco_img_dir: str = None,
    coco_ann_file: str = None,
    use_voc07: bool = True,
    use_voc12: bool = True,
):
    train_tf = SegTransform(train=True,  crop_size=crop_size)
    val_tf   = SegTransform(train=False, crop_size=crop_size)

    train_parts = []
    if use_voc07:
        train_parts.append(VOCSegDataset(voc_root, "2007", "train", train_tf))
    if use_voc12:
        train_parts.append(VOCSegDataset(voc_root, "2012", "train", train_tf))
    if coco_img_dir and coco_ann_file:
        train_parts.append(COCOSegDataset(coco_img_dir, coco_ann_file, train_tf))

    train_ds = ConcatDataset(train_parts)
    val_ds   = VOCSegDataset(voc_root, "2012", "val", val_tf)

    return train_ds, val_ds
