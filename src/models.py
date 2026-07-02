"""Fabrika modela: MobileNetV2 / ResNet50 u frozen ili finetune varijanti.

Nova glava (in -> hidden_dim -> num_classes) se uvek trenira. frozen zamrzava
ceo backbone; finetune dodatno odmrzava poslednji blok.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

from src.config import Config


def _make_head(in_features, hidden_dim, num_classes, dropout):
    """Glava klasifikatora: Linear -> ReLU -> Dropout -> Linear."""
    return nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )


def _set_requires_grad(module, value):
    for p in module.parameters():
        p.requires_grad = value


# Napomena: zamrzavamo samo tezine (requires_grad=False). BatchNorm running
# statistike su bufferi, ne parametri, pa se i dalje azuriraju dok je model u
# train() modu - "frozen" backbone tako dobija blagu BN adaptaciju na nas
# dataset. Ostavljeno namerno (standardno, cesto i korisno).


def _build_mobilenet(cfg: Config):
    net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    in_features = net.last_channel  # 1280
    _set_requires_grad(net.features, False)
    if cfg.mode == "finetune":
        _set_requires_grad(net.features[-2:], True)  # poslednja dva bloka
    net.classifier = _make_head(in_features, cfg.hidden_dim, cfg.num_classes, cfg.dropout)
    return net


def _build_resnet(cfg: Config):
    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = net.fc.in_features  # 2048
    _set_requires_grad(net, False)
    if cfg.mode == "finetune":
        _set_requires_grad(net.layer4, True)  # poslednji rezidualni blok
    net.fc = _make_head(in_features, cfg.hidden_dim, cfg.num_classes, cfg.dropout)
    return net


def build_model(cfg: Config):
    """Napravi model prema cfg.model i cfg.mode."""
    if cfg.model == "mobilenet":
        return _build_mobilenet(cfg)
    if cfg.model == "resnet":
        return _build_resnet(cfg)
    raise ValueError(f"Nepoznat model: {cfg.model!r}")


if __name__ == "__main__":
    import torch
    from src.utils import count_parameters, model_size_mb

    dummy = torch.randn(1, 3, 224, 224)
    print(f"{'eksperiment':22s}{'ukupno':>12s}{'trenabilno':>13s}{'udeo':>8s}{'MB':>8s}  izlaz")
    print("-" * 78)
    for model_name in ("mobilenet", "resnet"):
        for mode in ("frozen", "finetune"):
            cfg = Config(model=model_name, mode=mode)
            net = build_model(cfg).eval()
            total, trainable = count_parameters(net)
            out = net(dummy)
            print(f"{cfg.experiment_name:22s}{total:12,}{trainable:13,}"
                  f"{trainable / total * 100:7.1f}%{model_size_mb(net):8.1f}  {tuple(out.shape)}")
