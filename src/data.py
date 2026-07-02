"""Data pipeline: ucitavanje DDD slika i subject-wise podela (bez curenja).

Osoba je kodirana u imenu fajla (A0001.png -> osoba A; malo/veliko slovo
razlikuje klase ali oznacava istu osobu). Sve slike jedne osobe idu u isti
skup, pa model uvek testira na osobama koje nije video:
- subject_split: jedan train/val/test (osobe disjunktne)
- cv_folds:      GroupKFold po osobi za k-fold cross-validaciju

Pokreni proveru splita i EDA grafike:
    python -m src.data
"""

from __future__ import annotations

import re
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.config import Config, build_config

# ImageNet statistike (oba modela su pretrenirana sa ovim vrednostima).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def subject_of(path) -> str:
    """Osoba iz imena fajla: vodeca slova, velika (A0001 -> A, za12 -> ZA)."""
    m = re.match(r"[A-Za-z]+", Path(path).name)
    return m.group().upper() if m else Path(path).name


def _build_transforms(image_size, horizontal_flip, strong_augment=False):
    """Train transformacija (opc. jaka augmentacija) i deterministicka eval transformacija."""
    if strong_augment:
        # jaka augmentacija: otezava memorisanje (crop/rotacija/boja/brisanje),
        # da model manje pamti lice a vise koristi oci/usta.
        train_steps = [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        ]
    else:
        train_steps = [transforms.Resize((image_size, image_size))]
    if horizontal_flip:
        train_steps.append(transforms.RandomHorizontalFlip())
    train_steps += [transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if strong_augment:
        train_steps.append(transforms.RandomErasing(p=0.25))

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transforms.Compose(train_steps), eval_tf


def _bases(cfg: Config):
    """Dva ImageFolder-a nad istim folderom (isti redosled -> indeksi se poklapaju)."""
    train_tf, eval_tf = _build_transforms(cfg.image_size, cfg.horizontal_flip, cfg.strong_augment)
    train_base = datasets.ImageFolder(cfg.data_dir, transform=train_tf)
    eval_base = datasets.ImageFolder(cfg.data_dir, transform=eval_tf)
    return train_base, eval_base


def _subjects(base):
    """Osoba po uzorku, poravnato sa base.samples / base.targets."""
    return [subject_of(p) for p, _ in base.samples]


def subject_split(targets, subjects, val_split, test_split, seed):
    """Jedan subject-wise train/val/test (osobe disjunktne izmedju skupova)."""
    idx = list(range(len(targets)))
    gss1 = GroupShuffleSplit(n_splits=1, test_size=val_split + test_split, random_state=seed)
    tr_rel, tmp_rel = next(gss1.split(idx, targets, subjects))
    train_idx = [idx[i] for i in tr_rel]
    tmp_idx = [idx[i] for i in tmp_rel]

    tmp_sub = [subjects[i] for i in tmp_idx]
    tmp_tgt = [targets[i] for i in tmp_idx]
    rel_test = test_split / (val_split + test_split)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_test, random_state=seed)
    val_rel, te_rel = next(gss2.split(tmp_idx, tmp_tgt, tmp_sub))
    return train_idx, [tmp_idx[i] for i in val_rel], [tmp_idx[i] for i in te_rel]


def cv_folds(targets, subjects, n_folds, seed):
    """Lista (trainval_idx, test_idx) po foldu; svaka osoba je u test tacno jednom."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(list(tv), list(te)) for tv, te in
            sgkf.split(range(len(targets)), targets, subjects)]


def _val_from_trainval(trainval_idx, targets, subjects, val_split, seed):
    """Subject-wise izdvoji val iz trainval osoba (za early stopping)."""
    sub = [subjects[i] for i in trainval_idx]
    tgt = [targets[i] for i in trainval_idx]
    gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    tr_rel, val_rel = next(gss.split(range(len(trainval_idx)), tgt, sub))
    return [trainval_idx[i] for i in tr_rel], [trainval_idx[i] for i in val_rel]


def _loaders(cfg: Config, train_base, eval_base, train_idx, val_idx, test_idx):
    # persistent_workers izbegava ponovno pokretanje procesa svake epohe (Windows).
    common = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                  pin_memory=True, persistent_workers=cfg.num_workers > 0)
    train_loader = DataLoader(Subset(train_base, train_idx), shuffle=True, **common)
    val_loader = DataLoader(Subset(eval_base, val_idx), shuffle=False, **common)
    test_loader = DataLoader(Subset(eval_base, test_idx), shuffle=False, **common)
    return train_loader, val_loader, test_loader


def build_dataloaders(cfg: Config):
    """Jedan subject-wise split -> train/val/test loaderi + imena klasa."""
    train_base, eval_base = _bases(cfg)
    subjects = _subjects(train_base)
    train_idx, val_idx, test_idx = subject_split(
        train_base.targets, subjects, cfg.val_split, cfg.test_split, cfg.seed)
    return (*_loaders(cfg, train_base, eval_base, train_idx, val_idx, test_idx),
            train_base.classes)


def build_fold_dataloaders(cfg: Config, fold: int):
    """Za dati fold: test = osobe tog folda; train/val (nested) iz ostalih osoba."""
    train_base, eval_base = _bases(cfg)
    subjects = _subjects(train_base)
    trainval_idx, test_idx = cv_folds(train_base.targets, subjects, cfg.n_folds, cfg.seed)[fold]
    train_idx, val_idx = _val_from_trainval(
        trainval_idx, train_base.targets, subjects, cfg.val_split, cfg.seed)
    return (*_loaders(cfg, train_base, eval_base, train_idx, val_idx, test_idx),
            train_base.classes)


def _subject_set(indices, subjects):
    return {subjects[i] for i in indices}


def _report():
    """Proveri subject-wise split / foldove i sacuvaj EDA grafike (CLI modula)."""
    from src.utils import set_seed
    from src import viz

    cfg = build_config()
    set_seed(cfg.seed)

    base = datasets.ImageFolder(cfg.data_dir)  # bez transformacije, trebaju labele
    subjects = _subjects(base)
    targets, classes = base.targets, base.classes
    all_subjects = sorted(set(subjects))
    total = len(targets)
    print(f"Osoba: {len(all_subjects)} | slika: {total} | klase: {classes}")

    print(f"\n=== {cfg.n_folds}-fold GroupKFold po osobi ===")
    seen = []
    for k, (tv_idx, te_idx) in enumerate(cv_folds(targets, subjects, cfg.n_folds, cfg.seed)):
        te_sub = _subject_set(te_idx, subjects)
        assert not (te_sub & _subject_set(tv_idx, subjects)), "curenje: osoba u train i test!"
        seen += list(te_sub)
        print(f"fold {k}: test {len(te_sub)} osoba, {len(te_idx):5d} slika  {sorted(te_sub)}")
    assert sorted(seen) == all_subjects, "svaka osoba mora biti u test tacno jednom"
    print("OK: svaka osoba je u test tacno jednom, bez preklapanja train/test.")

    print("\n=== primer: fold 0, nested train/val/test ===")
    tr, va, te, _ = build_fold_dataloaders(cfg, 0)
    for name, loader in [("train", tr), ("val", va), ("test", te)]:
        idxs = loader.dataset.indices
        cnt = [sum(1 for i in idxs if targets[i] == c) for c in range(len(classes))]
        detail = ", ".join(f"{classes[c]} {cnt[c]}" for c in range(len(classes)))
        print(f"{name:5s}: {len(_subject_set(idxs, subjects)):2d} osoba, {len(idxs):5d} slika  ({detail})")
    s = [_subject_set(x.dataset.indices, subjects) for x in (tr, va, te)]
    assert not (s[0] & s[1]) and not (s[0] & s[2]) and not (s[1] & s[2])
    print("OK: train/val/test osobe su disjunktne.")

    imgs, labels = next(iter(tr))
    print(f"\nBatch: {tuple(imgs.shape)}, {imgs.dtype}, [{imgs.min():.2f}, {imgs.max():.2f}]")

    out = Path(cfg.output_dir) / "eda"
    overall = [targets.count(c) for c in range(len(classes))]
    print("Sacuvano:", viz.plot_class_distribution(classes, overall, out / "class_distribution.png"))
    print("Sacuvano:", viz.plot_sample_images(cfg.data_dir, classes, out / "sample_images.png", seed=cfg.seed))


if __name__ == "__main__":
    import sys
    try:  # Windows konzola je cp1252, pa moze da pukne na non-ASCII znakovima.
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _report()
