from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision.models as tvm


@dataclass(frozen=True)
class CnnConfig:
    backbone: str
    pretrained: bool
    dropout: float
    num_classes: int


def build_model(cfg: CnnConfig) -> nn.Module:
    if cfg.backbone == "resnet18":
        weights = tvm.ResNet18_Weights.DEFAULT if cfg.pretrained else None
        model = tvm.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(in_features, cfg.num_classes))
        return model

    if cfg.backbone == "resnet50":
        weights = tvm.ResNet50_Weights.DEFAULT if cfg.pretrained else None
        model = tvm.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(in_features, cfg.num_classes))
        return model

    if cfg.backbone == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.DEFAULT if cfg.pretrained else None
        model = tvm.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(in_features, cfg.num_classes))
        return model

    if cfg.backbone == "efficientnet_b2":
        weights = tvm.EfficientNet_B2_Weights.DEFAULT if cfg.pretrained else None
        model = tvm.efficientnet_b2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(in_features, cfg.num_classes))
        return model

    raise ValueError(f"Unsupported backbone: {cfg.backbone}")


@torch.no_grad()
def predict_logits(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model(batch)
