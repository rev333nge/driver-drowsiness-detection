"""Trening jednog eksperimenta. Pokretanje:
    python -m src.train --model mobilenet --mode frozen
    python -m src.train --config configs/resnet_finetune.yaml --epochs 5
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from src import viz
from src.config import build_config
from src.data import build_dataloaders
from src.engine import EarlyStopping, evaluate, train_one_epoch
from src.models import build_model
from src.utils import get_device, save_checkpoint, save_json, set_seed


def main():
    cfg = build_config()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"Eksperiment: {cfg.experiment_name} | uredjaj: {device}")

    # Test skup se ne dira ovde - cuva se za evaluaciju (Faza 5).
    train_loader, val_loader, _, classes = build_dataloaders(cfg)
    model = build_model(cfg).to(device)

    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    stopper = EarlyStopping(cfg.early_stopping_patience)

    out_dir = cfg.experiment_dir
    cfg.to_yaml(out_dir / "config.yaml")
    history = []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0

        is_best = stopper.step(va_loss)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "time_s": dt})
        print(f"epoha {epoch:2d}/{cfg.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.4f} | {dt:.1f}s"
              f"{'  <- best' if is_best else ''}")

        if is_best:
            save_checkpoint(model, out_dir / "best.pt", epoch=epoch,
                            val_loss=va_loss, val_acc=va_acc, classes=classes)
        if stopper.should_stop:
            print(f"Early stopping: nema poboljsanja {cfg.early_stopping_patience} epoha.")
            break

    save_json({"experiment": cfg.experiment_name, "best_val_loss": stopper.best_loss,
               "history": history}, out_dir / "history.json")
    viz.plot_training_curves(history, out_dir / "training_curves.png")
    print(f"Gotovo. Najbolji val loss: {stopper.best_loss:.4f}. Artefakti: {out_dir}")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
