import torch
import torch.nn as nn
from torchvision import models


class EfficientNetFeatureExtractor(nn.Module):
    """
    EfficientNet-B3 feature extractor.
    Outputs a pooled feature vector (batch, C).
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = (
            models.EfficientNet_B3_Weights.IMAGENET1K_V1
            if pretrained else None
        )

        base = models.efficientnet_b3(weights=weights)

        # Everything except final classifier head:
        self.backbone = nn.Sequential(*list(base.children())[:-1])

    def forward(self, x):
        x = self.backbone(x)     
        x = x.view(x.size(0), -1)
        return x
