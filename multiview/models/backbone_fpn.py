import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


class DilatedAggregation(nn.Module):
    def __init__(self, in_ch, out_ch, groups=32):
        super().__init__()
        self.b1 = BasicResBlock(in_ch, out_ch, dilation=1, groups=groups)
        self.b2 = BasicResBlock(in_ch, out_ch, dilation=2, groups=groups)
        self.b3 = BasicResBlock(in_ch, out_ch, dilation=4, groups=groups)

        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 3, out_ch, 1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        f1 = self.b1(x)
        f2 = self.b2(x)
        f3 = self.b3(x)
        return self.fuse(torch.cat([f1, f2, f3], dim=1))


class BasicResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1, groups=32):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                               padding=padding, dilation=dilation, bias=False)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.act   = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3,
                               padding=padding, dilation=dilation, bias=False)
        self.norm2 = nn.GroupNorm(groups, out_ch)

        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(groups, out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        return self.act(out + identity)
        

class DilatedCountingBackboneV2(nn.Module):
    """
    Produces a proper pyramid:
    c2 stride 4, c3 stride 8, c4 stride 16, c5 stride 16 (dilated)
    """
    def __init__(self, widths=(64, 128, 256, 256)):
        super().__init__()
        w2, w3, w4, w5 = widths

        # Stem: stride 4
        self.stem = nn.Sequential(
            nn.Conv2d(3, w2, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(32, w2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # c2: stride 4
        self.stage2 = nn.Sequential(
            BasicResBlock(w2, w2, stride=1),
            BasicResBlock(w2, w2, stride=1),
        )

        # c3: stride 8
        self.stage3 = nn.Sequential(
            BasicResBlock(w2, w3, stride=2),
            BasicResBlock(w3, w3, stride=1),
        )

        # c4: stride 16
        self.stage4 = nn.Sequential(
            BasicResBlock(w3, w4, stride=2),
            BasicResBlock(w4, w4, stride=1),
        )

        # c5: keep stride 16 but enlarge RF via dilation
        self.stage5 = DilatedAggregation(w4, w5)

    def forward(self, x):
        x = self.stem(x)
        c2 = self.stage2(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}



class MultiViewNetwork(nn.Module):
    """
    Multi-view regression network with optional FPN and configurable backbone.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        use_fpn: bool = True,
        fusion: str = "sum",
        fpn_channels: int = 256,
        in_size: int = 224,
    ):
        super().__init__()
        self.backbone_name = backbone.lower()
        self.use_fpn = use_fpn
        self.fusion = fusion.lower()
        self.fpn_channels = fpn_channels
        if self.fusion not in {"sum", "concat", "attention"}:
            raise ValueError(f"Unsupported fusion strategy '{fusion}'")

        print(
            f"Model | backbone={self.backbone_name} | "
            f"pretrained={pretrained} | use_fpn={self.use_fpn} | fusion={self.fusion}"
        )


        if self.backbone_name == "dcb":
            self.backbone = DilatedCountingBackboneV2()
        else:
            base = self._build_torchvision_backbone(self.backbone_name, pretrained)
            return_nodes = self._get_return_nodes(self.backbone_name)
            self.backbone = create_feature_extractor(base, return_nodes=return_nodes)


        # Infer channels
        c2_ch, c3_ch, c4_ch, c5_ch = self._infer_channels(in_size=in_size)

        # ---------------------------
        # Optional FPN layers
        # ---------------------------
        if self.use_fpn:
            self.fpn_c2 = nn.Conv2d(c2_ch, fpn_channels, kernel_size=1)
            self.fpn_c3 = nn.Conv2d(c3_ch, fpn_channels, kernel_size=1)
            self.fpn_c4 = nn.Conv2d(c4_ch, fpn_channels, kernel_size=1)
            self.fpn_c5 = nn.Conv2d(c5_ch, fpn_channels, kernel_size=1)

            fc_in = fpn_channels * 4
        else:
            # No FPN → regress from c5 only
            fc_in = c5_ch

        self.regressor = self._make_regressor(fc_in)

        if self.fusion == "concat":
            self.fusion_regressor = self._make_regressor(fc_in * 2)
        elif self.fusion == "attention":
            heads = self._attention_heads(fc_in)
            self.cross_view_attention = nn.MultiheadAttention(
                embed_dim=fc_in,
                num_heads=heads,
                batch_first=True,
            )
            self.fusion_regressor = self._make_regressor(fc_in)

    def _make_regressor(self, in_features: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def _attention_heads(self, in_features: int) -> int:
        for heads in (8, 4, 2, 1):
            if in_features % heads == 0:
                return heads
        return 1

    # ------------------------------------------------------------------
    # Backbone creation helpers (unchanged)
    # ------------------------------------------------------------------
    def _build_torchvision_backbone(self, name: str, pretrained: bool) -> nn.Module:
        if not hasattr(models, name):
            raise ValueError(f"Unsupported backbone '{name}'")

        weights = "IMAGENET1K_V1" if pretrained else None
        try:
            return getattr(models, name)(weights=weights)
        except TypeError:
            return getattr(models, name)(pretrained=pretrained)

    def _get_return_nodes(self, name: str) -> dict:
        if name.startswith(("resnet", "resnext")):
            return {"layer1": "c2", "layer2": "c3", "layer3": "c4", "layer4": "c5"}

        if name.startswith("efficientnet"):
            return {
                "features.2": "c2",
                "features.3": "c3",
                "features.5": "c4",
                "features.8": "c5",
            }

        if name.startswith("vgg"):
            return {
                "features.9": "c2",
                "features.16": "c3",
                "features.23": "c4",
                "features.30": "c5",
            }

        raise ValueError(f"Unsupported backbone family: {name}")

    def _infer_channels(self, in_size: int):
        device = next(self.parameters()).device
        self.backbone.eval()

        with torch.no_grad():
            x = torch.zeros(1, 3, in_size, in_size, device=device)
            feats = self.backbone(x)

        self.backbone.train()
        return (
            feats["c2"].shape[1],
            feats["c3"].shape[1],
            feats["c4"].shape[1],
            feats["c5"].shape[1],
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        c2, c3, c4, c5 = feats["c2"], feats["c3"], feats["c4"], feats["c5"]

        if self.use_fpn:
            p4 = self.fpn_c5(c5)
            p4 = F.interpolate(p4, size=c4.shape[-2:], mode="nearest") + self.fpn_c4(c4)

            p3 = self.fpn_c3(c3)
            p3 = F.interpolate(p4, size=c3.shape[-2:], mode="nearest") + p3

            p2 = self.fpn_c2(c2)
            p2 = F.interpolate(p3, size=c2.shape[-2:], mode="nearest") + p2

            p2v = F.adaptive_avg_pool2d(p2, 1).flatten(1)
            p3v = F.adaptive_avg_pool2d(p3, 1).flatten(1)
            p4v = F.adaptive_avg_pool2d(p4, 1).flatten(1)
            p5v = F.adaptive_avg_pool2d(self.fpn_c5(c5), 1).flatten(1)

            features = torch.cat([p2v, p3v, p4v, p5v], dim=1)
        else:
            # Plain backbone regression
            features = F.adaptive_avg_pool2d(c5, 1).flatten(1)

        return features

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.extract_features(x))

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        features1 = self.extract_features(img1)
        features2 = self.extract_features(img2)

        if self.fusion == "sum":
            return self.regressor(features1) + self.regressor(features2)

        if self.fusion == "concat":
            return self.fusion_regressor(torch.cat([features1, features2], dim=1))

        tokens = torch.stack([features1, features2], dim=1)
        attended, _ = self.cross_view_attention(tokens, tokens, tokens)
        return self.fusion_regressor(attended.mean(dim=1))
