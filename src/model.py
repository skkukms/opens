import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

def _patch_dilation(module: nn.Module, dilation: int, fix_stride: bool) -> None:
    """Replace stride-2 depthwise conv with dilation and, if fix_stride, set stride=1."""
    for m in module.modules():
        if not (isinstance(m, nn.Conv2d) and m.groups > 1 and m.kernel_size[0] > 1):
            continue
        if fix_stride and m.stride == (2, 2):
            m.stride = (1, 1)
        k = m.kernel_size[0]
        m.dilation = (dilation, dilation)
        m.padding  = (dilation * (k - 1) // 2, dilation * (k - 1) // 2)


def build_backbone(pretrained: bool = True):
    """
    MobileNetV3-Large modified to output stride 16 (OS16).

    Original strides: /2 /4 /8 /16 /32
    After patch:      /2 /4 /8 /16      (features[13] stride→dilation=2)
    Output: (B, 960, H/16, W/16)
    """
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    base = mobilenet_v3_large(weights=weights)
    feats = base.features  # 17 sequential blocks (0-16)

    # features[13]: stride=2 block that causes H/32 — convert to dilation
    _patch_dilation(feats[13], dilation=2, fix_stride=True)
    # features[14,15]: already stride=1 but add dilation for consistent RF
    _patch_dilation(feats[14], dilation=2, fix_stride=False)
    _patch_dilation(feats[15], dilation=2, fix_stride=False)

    return feats  # out_channels = 960


# ---------------------------------------------------------------------------
# Lightweight ASPP (depthwise-separable, 128 channels)
# ---------------------------------------------------------------------------

class _DSConvBNReLU(nn.Module):
    """Depthwise-separable conv → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        k = 3
        self.dw = nn.Conv2d(in_ch, in_ch, k,
                            padding=dilation * (k - 1) // 2,
                            dilation=dilation, groups=in_ch, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_ch)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn_dw(self.dw(x)))
        return self.relu(self.bn_pw(self.pw(x)))


class _ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class LightASPP(nn.Module):
    """
    5-branch ASPP with depthwise-separable convs.
    in_ch=960  out_ch=128  rates=[6,12,18] (OS16)
    """

    def __init__(self, in_ch: int = 960, out_ch: int = 128,
                 rates: tuple = (6, 12, 18)):
        super().__init__()
        self.b0   = _ConvBNReLU(in_ch, out_ch, 1)            # 1×1
        self.b1   = _DSConvBNReLU(in_ch, out_ch, rates[0])
        self.b2   = _DSConvBNReLU(in_ch, out_ch, rates[1])
        self.b3   = _DSConvBNReLU(in_ch, out_ch, rates[2])
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            _ConvBNReLU(in_ch, out_ch, 1),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2:]
        p = F.interpolate(self.pool(x), size=(H, W),
                          mode="bilinear", align_corners=False)
        x = torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x), p], dim=1)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Full segmentation model
# ---------------------------------------------------------------------------

class SegModel(nn.Module):
    def __init__(self, num_classes: int = 21, pretrained: bool = True,
                 aspp_ch: int = 128):
        super().__init__()
        self.backbone = build_backbone(pretrained)
        self.aspp     = LightASPP(in_ch=960, out_ch=aspp_ch)
        self.head     = nn.Conv2d(aspp_ch, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2:]
        x = self.backbone(x)           # (B, 960, H/16, W/16)
        x = self.aspp(x)               # (B, 128, H/16, W/16)
        x = self.head(x)               # (B, C,   H/16, W/16)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x                       # (B, C, H, W)

    def backbone_params(self):
        return self.backbone.parameters()

    def head_params(self):
        return list(self.aspp.parameters()) + list(self.head.parameters())
