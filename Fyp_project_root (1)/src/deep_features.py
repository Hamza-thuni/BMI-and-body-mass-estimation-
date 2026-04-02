import torch
import torch.nn as nn
from torchvision import models


class ResNetFeatureExtractor(nn.Module):
    """
    Extract global pooled deep features from a ResNet backbone.
    Output shape: (batch, 2048)
    """
    def __init__(self, arch: str = "resnet50", pretrained: bool = True):
        super().__init__()

        if arch == "resnet50":
            base = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            )
        elif arch == "resnet101":
            base = models.resnet101(
                weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            )
        else:
            raise ValueError("Unsupported architecture: choose resnet50 or resnet101")

        # everything except the final FC layer
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # Output: (B,2048,1,1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)           # (B,2048,1,1)
        feats = feats.view(feats.size(0), -1)  # (B,2048)
        return feats

