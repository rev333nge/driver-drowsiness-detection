"""Metrike: klasifikacija (iz out-of-fold predikcija) + prakticne (FPS, params, velicina)."""

from __future__ import annotations

import time

import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

from src.utils import count_parameters, model_size_mb


def classification_metrics(y_true, y_pred, pos_label):
    """Accuracy + precision/recall/f1 za pozitivnu klasu (Drowsy)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
    }


@torch.no_grad()
def measure_fps(model, device, image_size=224, n=100, warmup=10):
    """Prosecan broj slika/s za inferenciju jedne slike (relevantno za real-time)."""
    model.eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


def practical_metrics(model, device, image_size=224):
    """Broj parametara, velicina (MB) i brzina inferencije (FPS)."""
    total, trainable = count_parameters(model)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "size_mb": round(model_size_mb(model), 2),
        "fps": round(measure_fps(model, device, image_size), 1),
    }
