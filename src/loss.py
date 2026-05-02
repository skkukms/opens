import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int = 21, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W), targets: (B, H, W)
        valid = targets != self.ignore_index
        targets_clamped = targets.clone()
        targets_clamped[~valid] = 0

        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        one_hot = F.one_hot(targets_clamped, self.num_classes)  # (B, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()          # (B, C, H, W)

        # zero-out ignored pixels in both
        mask = valid.unsqueeze(1).float()
        probs = probs * mask
        one_hot = one_hot * mask

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        cardinality = (probs + one_hot).sum(dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class SegLoss(nn.Module):
    def __init__(self, num_classes: int = 21, ignore_index: int = 255,
                 ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.ce_w = ce_weight
        self.dice_w = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_w * self.ce(logits, targets) + self.dice_w * self.dice(logits, targets)
