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
            # --- random scale ---
            scale = random.uniform(0.5, 2.0)
            sh = int(self.crop_size * scale)
            sw = int(self.crop_size * scale)
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

            # --- gaussian blur ---
            if random.random() < 0.5:
                img = TF.gaussian_blur(img, kernel_size=23,
                                       sigma=random.uniform(0.1, 2.0))
        else:
            img  = TF.resize(img,  [self.crop_size, self.crop_size],
                             interpolation=InterpolationMode.BILINEAR, antialias=True)
            mask = TF.resize(mask, [self.crop_size, self.crop_size],
                             interpolation=InterpolationMode.NEAREST)

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
                                  download=False)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, mask = self.ds[idx]
        mask = mask.convert("L")
        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask


def _coco_seg_mask(anns: list, h: int, w: int) -> Image.Image:
    """Convert COCO annotations (for one image) to a VOC-style mask."""
    mask = np.zeros((h, w), dtype=np.uint8)

    # Draw largest area first so smaller objects appear on top
    for ann in sorted(anns, key=lambda a: a.get("area", 0), reverse=True):
        if ann.get("iscrowd", 0):
            continue
        cat_id = ann["category_id"]
        if cat_id not in COCO_TO_VOC:
            continue
        voc_cls = COCO_TO_VOC[cat_id]

        seg = ann.get("segmentation", [])
        if isinstance(seg, list):
            tmp = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(tmp)
            for poly in seg:
                if len(poly) >= 6:
                    pts = list(zip(poly[::2], poly[1::2]))
                    draw.polygon(pts, fill=voc_cls)
            tmp_arr = np.array(tmp)
            mask[tmp_arr > 0] = tmp_arr[tmp_arr > 0]
        elif isinstance(seg, dict):
            # RLE (crowd annotations skipped above, but just in case)
            try:
                from pycocotools import mask as cm
                rle = seg
                if isinstance(rle.get("counts"), list):
                    rle = cm.frPyObjects(rle, h, w)
                m = cm.decode(rle)
                mask[m > 0] = voc_cls
            except ImportError:
                pass

    return Image.fromarray(mask)


class COCOSegDataset(Dataset):
    def __init__(self, img_dir: str, ann_file: str, transform=None):
        self.ds = CocoDetection(root=img_dir, annFile=ann_file)
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
