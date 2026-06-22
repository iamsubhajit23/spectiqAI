"""Model definitions for kidney tumor classification."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class ConvBlock(nn.Module):
    """A small CNN block: convolution, normalization, activation, then pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SimpleKidneyCNN(nn.Module):
    """A beginner-friendly CNN for binary kidney image classification."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_resnet50(num_classes: int = 2, pretrained: bool = False, freeze_features: bool = True) -> nn.Module:
    """Create a ResNet50 classifier for binary kidney image classification."""
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    if freeze_features:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes: int = 2, pretrained: bool = False, freeze_features: bool = True) -> nn.Module:
    """Create an EfficientNet-B0 classifier for binary kidney image classification."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    if freeze_features:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_transfer_model(
    model_name: str,
    num_classes: int = 2,
    pretrained: bool = False,
    freeze_features: bool = True,
) -> nn.Module:
    """Build a named transfer-learning model with a two-class classification head."""
    normalized_name = model_name.lower().replace("-", "_")

    if normalized_name == "resnet50":
        return build_resnet50(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_features=freeze_features,
        )
    if normalized_name in {"efficientnet_b0", "efficientnetb0"}:
        return build_efficientnet_b0(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_features=freeze_features,
        )

    raise ValueError("model_name must be 'resnet50' or 'efficientnet_b0'.")
