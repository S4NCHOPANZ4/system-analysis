import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x) * x
        sa = self.spatial_attention(torch.cat([
            torch.mean(ca, dim=1, keepdim=True),
            torch.max(ca, dim=1, keepdim=True)[0]
        ], dim=1)) * ca
        return sa

class ResNet50_CBAM(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet50_CBAM, self).__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1
        resnet = resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.cbam = CBAM(2048)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
