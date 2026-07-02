"""Konfiguracija eksperimenta: defaults < YAML < CLI."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    # identitet eksperimenta
    model: str = "mobilenet"          # mobilenet | resnet
    mode: str = "frozen"              # frozen | finetune

    # podaci
    data_dir: str = "data"
    image_size: int = 224
    val_split: float = 0.15
    test_split: float = 0.15
    n_folds: int = 5                  # GroupKFold po osobi (subject-wise CV)
    batch_size: int = 64
    num_workers: int = 8

    # glava klasifikatora (in -> hidden_dim -> num_classes)
    hidden_dim: int = 256
    dropout: float = 0.5
    num_classes: int = 2

    # trening
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    early_stopping_patience: int = 5

    # augmentacija
    horizontal_flip: bool = True
    strong_augment: bool = False      # RandomResizedCrop/rotacija/ColorJitter/RandomErasing

    # ostalo
    seed: int = 42
    output_dir: str = "outputs"
    device: str = "cuda"              # cuda | cpu

    def __post_init__(self) -> None:
        if self.model not in ("mobilenet", "resnet"):
            raise ValueError(f"Nepoznat model: {self.model!r}")
        if self.mode not in ("frozen", "finetune"):
            raise ValueError(f"Nepoznat mode: {self.mode!r}")

    @property
    def experiment_name(self) -> str:
        """Npr. 'mobilenet_frozen'."""
        return f"{self.model}_{self.mode}"

    @property
    def experiment_dir(self) -> Path:
        return Path(self.output_dir) / self.experiment_name

    def to_yaml(self, path) -> None:
        """Sacuvaj konfiguraciju pored rezultata runa."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)


_VALID_KEYS = {f.name for f in fields(Config)}


def _load_yaml(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    unknown = set(data) - _VALID_KEYS
    if unknown:
        raise ValueError(f"Nepoznati kljucevi u {path}: {sorted(unknown)}")
    return data


def build_config(argv=None) -> Config:
    """Sklopi Config iz CLI argumenata i opcionog YAML-a (CLI ima prednost)."""
    parser = argparse.ArgumentParser(description="Detekcija pospanosti vozaca")
    parser.add_argument("--config", type=str, default=None,
                        help="Putanja do YAML config fajla (vidi configs/).")
    parser.add_argument("--model", choices=["mobilenet", "resnet"])
    parser.add_argument("--mode", choices=["frozen", "finetune"])
    parser.add_argument("--data-dir", dest="data_dir")
    parser.add_argument("--image-size", dest="image_size", type=int)
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--num-workers", dest="num_workers", type=int)
    parser.add_argument("--n-folds", dest="n_folds", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--strong-augment", dest="strong_augment",
                        action="store_true", default=None,
                        help="Ukljuci jaku augmentaciju (crop/rotacija/boja/brisanje).")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu"])
    parser.add_argument("--output-dir", dest="output_dir")
    args = parser.parse_args(argv)

    values = {}
    if args.config:
        values.update(_load_yaml(args.config))
    for key, val in vars(args).items():
        if key == "config" or val is None:
            continue
        values[key] = val

    return Config(**values)
