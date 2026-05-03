import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


# ---------------------------------------------------------------------------
# Backbone helpers
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


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _DSConvBNReLU(nn.Module):
    """Depthwise-separable conv → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        k = 3
        self.dw    = nn.Conv2d(in_ch, in_ch, k,
                               padding=dilation * (k - 1) // 2,
                               dilation=dilation, groups=in_ch, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_ch)
        self.pw    = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

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


# ---------------------------------------------------------------------------
# Lightweight ASPP (depthwise-separable)
# ---------------------------------------------------------------------------

class LightASPP(nn.Module):
    """
    5-branch ASPP with depthwise-separable convs.
    in_ch=960, out_ch configurable, rates=[6,12,18] (OS16)
    """

    def __init__(self, in_ch: int = 960, out_ch: int = 192,
                 rates: tuple = (6, 12, 18)):
        super().__init__()
        self.b0   = _ConvBNReLU(in_ch, out_ch, 1)
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
# Full segmentation model  (DeepLabV3+-style decoder with /8 skip)
# ---------------------------------------------------------------------------

class SegModel(nn.Module):
    """
    MobileNetV3-Large (OS16) + LightASPP + DeepLabV3+-style skip decoder.

    Encoder split:
      enc_low  : features[0:7]  → (B,  40, H/8,  W/8)
      enc_high : features[7:17] → (B, 960, H/16, W/16)  (dilation patched)

    Decoder (Plan-A):
      ASPP(960 → aspp_ch=192)
      → ×2 upsample
      → concat low_proj(40 → low_proj_ch=48)       [total: 192+48=240 ch]
      → DSConv(240 → 192)  ×2
      → head(192 → C)
      → ×8 upsample  →  (B, C, H, W)
    """

    def __init__(self, num_classes: int = 21, pretrained: bool = True,
                 aspp_ch: int = 192, low_proj_ch: int = 48):
        super().__init__()

        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        base  = mobilenet_v3_large(weights=weights)
        feats = base.features          # 17 sequential blocks (0-16)

        # Patch OS16 dilation before splitting
        _patch_dilation(feats[13], dilation=2, fix_stride=True)
        _patch_dilation(feats[14], dilation=2, fix_stride=False)
        _patch_dilation(feats[15], dilation=2, fix_stride=False)

        self.enc_low  = feats[:7]      # (B,  40, H/8,  W/8)
        self.enc_high = feats[7:]      # (B, 960, H/16, W/16)

        self.aspp = LightASPP(in_ch=960, out_ch=aspp_ch)

        # Low-level skip: 40 → low_proj_ch
        self.low_proj = nn.Sequential(
            nn.Conv2d(40, low_proj_ch, 1, bias=False),
            nn.BatchNorm2d(low_proj_ch),
            nn.ReLU(inplace=True),
        )

        # Decoder: two DSConv blocks for better boundary refinement
        dec_in = aspp_ch + low_proj_ch   # 192 + 48 = 240
        self.decoder = nn.Sequential(
            _DSConvBNReLU(dec_in,  aspp_ch),   # 240 → 192
            _DSConvBNReLU(aspp_ch, aspp_ch),   # 192 → 192
        )

        self.head = nn.Conv2d(aspp_ch, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2:]

        low      = self.enc_low(x)           # (B,  40, H/8,  W/8)
        high     = self.enc_high(low)        # (B, 960, H/16, W/16)

        aspp_out = self.aspp(high)           # (B, 192, H/16, W/16)
        aspp_up  = F.interpolate(aspp_out, size=low.shape[2:],
                                 mode="bilinear", align_corners=False)  # (B, 192, H/8, W/8)

        low_feat = self.low_proj(low)        # (B,  48, H/8,  W/8)

        dec = self.decoder(
            torch.cat([aspp_up, low_feat], dim=1)   # (B, 240, H/8, W/8)
        )                                            # (B, 192, H/8, W/8)

        logit = self.head(dec)               # (B,   C, H/8,  W/8)
        return F.interpolate(logit, size=(H, W),
                             mode="bilinear", align_corners=False)

    def backbone_params(self):
        return list(self.enc_low.parameters()) + list(self.enc_high.parameters())

    def head_params(self):
        return (list(self.aspp.parameters())     +
                list(self.low_proj.parameters()) +
                list(self.decoder.parameters())  +
                list(self.head.parameters()))
