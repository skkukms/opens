import random
import numpy as np
import torch
import torch.profiler


NUM_CLASSES = 21
IGNORE_INDEX = 255

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MIoUMeter:
    """Accumulates confusion matrix and computes mean IoU."""

    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        # pred: (B, H, W) int64 class indices
        # target: (B, H, W) int64, ignore_index pixels excluded
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()

        mask = target != self.ignore_index
        pred = pred[mask]
        target = target[mask]

        valid = (target >= 0) & (target < self.num_classes)
        pred = pred[valid]
        target = target[valid]

        np.add.at(self.confusion, (target, pred), 1)

    def compute(self) -> dict:
        cm = self.confusion.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        iou = np.where((tp + fp + fn) > 0, tp / (tp + fp + fn), np.nan)
        miou = float(np.nanmean(iou))

        per_class = {VOC_CLASSES[i]: float(iou[i]) for i in range(self.num_classes)}
        return {"mIoU": miou, "per_class_iou": per_class}


def measure_gflops(model: torch.nn.Module, device: torch.device,
                   input_size: tuple = (1, 3, 480, 640)) -> float:
    model.eval()
    x = torch.randn(*input_size).to(device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_flops=True,
    ) as prof:
        with torch.no_grad():
            model(x)

    total_flops = sum(e.flops for e in prof.key_averages() if e.flops > 0)
    return total_flops / 1e9


def save_checkpoint(state: dict, path: str) -> None:
    torch.save(state, path)


def load_checkpoint(path: str, model: torch.nn.Module,
                    optimizer=None, scheduler=None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("epoch", 0)
