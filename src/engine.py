"""Trening i validacija: jedna epoha + early stopping."""

from __future__ import annotations

import torch
from torch.amp import autocast


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    """Jedna epoha treninga (uz mixed precision); vraca (prosecan_loss, tacnost)."""
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        # autocast racuna forward u fp16 gde je bezbedno; scaler cuva gradijente
        # od potkoracenja. Kad je scaler iskljucen (CPU), sve ostaje fp32.
        with autocast(device_type=device.type, enabled=scaler.is_enabled()):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        n += images.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Prolaz kroz skup bez treniranja; vraca (prosecan_loss, tacnost)."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        n += images.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate_collect(model, loader, criterion, device):
    """Jedan prolaz kroz skup: vrati (loss, acc, y_true, y_pred). Za finalni test."""
    model.eval()
    total_loss, n, y_true, y_pred = 0.0, 0, [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * images.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(outputs.argmax(1).cpu().tolist())
        n += images.size(0)
    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / n
    return total_loss / n, acc, y_true, y_pred


class EarlyStopping:
    """Prati validacioni loss; staje ako nema poboljsanja `patience` epoha."""

    def __init__(self, patience=3):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        """Azuriraj stanje; vrati True ako je ovo novi najbolji rezultat."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False
