"""Istreniraj finalni model na (skoro) svim podacima i snimi tezine za demo.

Za razliku od CV-a ovde nema test skupa: subject-wise se izdvoji mali val samo
za early stopping, ostatak je train, a najbolji (po val) model se snima u
outputs/final_<eksperiment>.pt zajedno sa klasama i konfiguracijom.

Pokretanje:
    python -m src.train_final --config configs/resnet_finetune.yaml
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.config import build_config
from src.data import _bases, _subjects, _val_from_trainval
from src.engine import EarlyStopping, evaluate, train_one_epoch
from src.models import build_model
from src.utils import get_device, set_seed


def main():
    cfg = build_config()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    torch.backends.cudnn.benchmark = True
    print(f"Finalni model: {cfg.experiment_name} | uredjaj: {device} | data: {cfg.data_dir}")

    train_base, eval_base = _bases(cfg)
    subjects = _subjects(train_base)
    all_idx = list(range(len(train_base.targets)))
    train_idx, val_idx = _val_from_trainval(all_idx, train_base.targets, subjects,
                                            cfg.val_split, cfg.seed)
    classes = train_base.classes

    common = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                  pin_memory=True, persistent_workers=cfg.num_workers > 0)
    train_loader = DataLoader(Subset(train_base, train_idx), shuffle=True, **common)
    val_loader = DataLoader(Subset(eval_base, val_idx), shuffle=False, **common)
    print(f"train {len(train_idx)} slika | val {len(val_idx)} slika (za early stopping)")

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    stopper = EarlyStopping(cfg.early_stopping_patience)

    best_state = None
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        if stopper.step(va_loss):  # novi najbolji -> zapamti tezine (CPU)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  epoha {epoch:2d} | train {tr_loss:.3f}/{tr_acc:.3f} "
              f"| val {va_loss:.3f}/{va_acc:.3f} | {time.time() - t0:.0f}s")
        if stopper.should_stop:
            break

    out_path = Path(cfg.output_dir) / f"final_{cfg.experiment_name}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "classes": classes, "config": asdict(cfg)}, out_path)
    print(f"\nSnimljen model: {out_path}  (best val loss {stopper.best_loss:.3f})")
    print(f"Demo: python -m src.webcam --model {out_path}")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
