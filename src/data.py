"""Data pipeline: ucitavanje DDD slika i stratifikovan 70/15/15 split.

Napomena za izvestaj: per-image split, pa isti vozac moze zavrsiti i u train
i u test skupu (moguce blago naduvane metrike). Pokreni proveru i EDA grafike:
    python -m src.data
"""

from __future__ import annotations

from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.config import Config, build_config

# ImageNet statistike (oba modela su pretrenirana sa ovim vrednostima).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_transforms(image_size, horizontal_flip):
    """Train transformacija (uz flip) i deterministicka eval transformacija."""
    train_steps = [transforms.Resize((image_size, image_size))]
    if horizontal_flip:
        train_steps.append(transforms.RandomHorizontalFlip())
    train_steps += [transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transforms.Compose(train_steps), eval_tf


def stratified_split(targets, val_split, test_split, seed):
    """Vrati (train_idx, val_idx, test_idx) uz ocuvan odnos klasa."""
    indices = list(range(len(targets)))
    train_idx, temp_idx = train_test_split(
        indices, test_size=val_split + test_split,
        stratify=targets, random_state=seed)
    temp_targets = [targets[i] for i in temp_idx]
    rel_test = test_split / (val_split + test_split)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=rel_test,
        stratify=temp_targets, random_state=seed)
    return train_idx, val_idx, test_idx


def build_dataloaders(cfg: Config):
    """Vrati train/val/test DataLoader-e i imena klasa."""
    train_tf, eval_tf = _build_transforms(cfg.image_size, cfg.horizontal_flip)
    # Dva ImageFolder-a nad istim folderom: isti redosled fajlova znaci da se
    # indeksi splita poklapaju, a svaki split dobija svoju transformaciju.
    train_base = datasets.ImageFolder(cfg.data_dir, transform=train_tf)
    eval_base = datasets.ImageFolder(cfg.data_dir, transform=eval_tf)
    train_idx, val_idx, test_idx = stratified_split(
        train_base.targets, cfg.val_split, cfg.test_split, cfg.seed)

    # persistent_workers izbegava ponovno pokretanje procesa svake epohe
    # (bitno na Windows-u gde je spawn skup).
    common = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                  pin_memory=True, persistent_workers=cfg.num_workers > 0)
    train_loader = DataLoader(Subset(train_base, train_idx), shuffle=True, **common)
    val_loader = DataLoader(Subset(eval_base, val_idx), shuffle=False, **common)
    test_loader = DataLoader(Subset(eval_base, test_idx), shuffle=False, **common)
    return train_loader, val_loader, test_loader, train_base.classes


def _counts(indices, targets, n_classes):
    counts = [0] * n_classes
    for i in indices:
        counts[targets[i]] += 1
    return counts


def _report():
    """Proveri split i sacuvaj EDA grafike (CLI modula)."""
    from src.utils import set_seed
    from src import viz

    cfg = build_config()
    set_seed(cfg.seed)

    base = datasets.ImageFolder(cfg.data_dir)  # bez transformacije, trebaju samo labele
    targets, classes = base.targets, base.classes
    train_idx, val_idx, test_idx = stratified_split(
        targets, cfg.val_split, cfg.test_split, cfg.seed)

    total = len(targets)
    print("Klase: " + ", ".join(f"{c}={i}" for i, c in enumerate(classes)))
    print("-" * 62)
    for name, idx in [("Ukupno", range(total)), ("Train", train_idx),
                      ("Val", val_idx), ("Test", test_idx)]:
        idx = list(idx)
        c = _counts(idx, targets, len(classes))
        detail = ", ".join(f"{classes[i]} {c[i]}" for i in range(len(classes)))
        print(f"{name:7s}{len(idx):7d}  ({len(idx) / total * 100:4.1f}%)   {detail}")
    print("-" * 62)

    assert len(train_idx) + len(val_idx) + len(test_idx) == total
    assert not (set(train_idx) & set(val_idx))
    assert not (set(train_idx) & set(test_idx))
    assert not (set(val_idx) & set(test_idx))
    print("OK: splitovi se ne preklapaju i pokrivaju svih", total, "slika.")

    train_loader, _, _, _ = build_dataloaders(cfg)
    imgs, labels = next(iter(train_loader))
    print(f"Batch: images={tuple(imgs.shape)}, labels={tuple(labels.shape)}, "
          f"dtype={imgs.dtype}, vrednosti=[{imgs.min():.2f}, {imgs.max():.2f}]")

    out = Path(cfg.output_dir) / "eda"
    overall = _counts(list(range(total)), targets, len(classes))
    print("Sacuvano:", viz.plot_class_distribution(classes, overall, out / "class_distribution.png"))
    print("Sacuvano:", viz.plot_sample_images(cfg.data_dir, classes, out / "sample_images.png", seed=cfg.seed))


if __name__ == "__main__":
    import sys
    try:  # Windows konzola je cp1252, pa moze da pukne na non-ASCII znakovima.
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _report()
