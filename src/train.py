"""Trening jednog eksperimenta kroz subject-wise k-fold CV. Pokretanje:
    python -m src.train --model mobilenet --mode frozen
    python -m src.train --config configs/resnet_finetune.yaml
"""

from __future__ import annotations

import copy
import statistics
import time

import torch
import torch.nn as nn

from src import viz
from src.config import build_config
from src.data import build_fold_dataloaders
from src.engine import EarlyStopping, evaluate, evaluate_collect, train_one_epoch
from src.models import build_model
from src.utils import get_device, save_json, set_seed


def _train_fold(cfg, fold, device):
    """Istreniraj jedan fold (early stopping na val) i oceni na test osobama."""
    train_loader, val_loader, test_loader, classes = build_fold_dataloaders(cfg, fold)
    model = build_model(cfg).to(device)

    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    stopper = EarlyStopping(cfg.early_stopping_patience)

    best_state, history = None, []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0
        if stopper.step(va_loss):  # novi najbolji -> zapamti tezine (na CPU)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "time_s": dt})
        print(f"  epoha {epoch:2d} | train {tr_loss:.3f}/{tr_acc:.3f} "
              f"| val {va_loss:.3f}/{va_acc:.3f} | {dt:.0f}s")
        if stopper.should_stop:
            break

    # najbolji (po val) model -> ocena na test osobama koje nije video
    if best_state is not None:
        model.load_state_dict(best_state)
    te_loss, te_acc, y_true, y_pred = evaluate_collect(model, test_loader, criterion, device)
    return {"fold": fold, "test_loss": te_loss, "test_acc": te_acc,
            "best_val_loss": stopper.best_loss, "epochs_run": len(history),
            "history": history, "y_true": y_true, "y_pred": y_pred}, classes


def main():
    cfg = build_config()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    torch.backends.cudnn.benchmark = True
    print(f"Eksperiment: {cfg.experiment_name} | uredjaj: {device} | "
          f"AMP: {device.type == 'cuda'} | {cfg.n_folds}-fold CV")

    out_dir = cfg.experiment_dir
    cfg.to_yaml(out_dir / "config.yaml")

    folds, classes = [], None
    for fold in range(cfg.n_folds):
        print(f"\n===== Fold {fold + 1}/{cfg.n_folds} =====")
        res, classes = _train_fold(cfg, fold, device)
        folds.append(res)
        print(f"  -> test acc {res['test_acc']:.4f} | test loss {res['test_loss']:.4f}")

    accs = [f["test_acc"] for f in folds]
    losses = [f["test_loss"] for f in folds]
    mean_acc = statistics.mean(accs)
    std_acc = statistics.pstdev(accs) if len(accs) > 1 else 0.0
    print(f"\n=== {cfg.experiment_name}: test acc {mean_acc:.4f} +/- {std_acc:.4f} "
          f"(min {min(accs):.4f}, max {max(accs):.4f}) preko {cfg.n_folds} foldova ===")

    save_json({"experiment": cfg.experiment_name, "classes": classes, "n_folds": cfg.n_folds,
               "test_acc_mean": mean_acc, "test_acc_std": std_acc,
               "test_loss_mean": statistics.mean(losses), "folds": folds},
              out_dir / "cv_results.json")
    for f in folds:
        viz.plot_training_curves(f["history"], out_dir / f"curves_fold{f['fold']}.png")
    print(f"Artefakti: {out_dir}")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
