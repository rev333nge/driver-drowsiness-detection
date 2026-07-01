"""Single source of truth for experiment configuration.

The whole point of this module: one `Config` object drives all four experiments
(mobilenet/resnet x frozen/finetune). Nothing is copy-pasted per experiment —
the model factory, data pipeline and training loop all read from here.

Precedence (low -> high): dataclass defaults < YAML file < CLI flags.
So per-experiment YAMLs live in configs/, and you can still tweak one value
from the command line without editing a file:

    python -m src.train --config configs/resnet_finetune.yaml --epochs 5
    python -m src.train --model mobilenet --mode frozen      # pure CLI, uses defaults
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    # --- experiment identity (drives the model factory + output paths) ---
    model: str = "mobilenet"          # "mobilenet" | "resnet"
    mode: str = "frozen"              # "frozen" | "finetune"

    # --- data ---
    data_dir: str = "data"
    image_size: int = 224
    val_split: float = 0.15
    test_split: float = 0.15
    batch_size: int = 64
    num_workers: int = 4

    # --- classifier head (in -> hidden_dim -> num_classes, ReLU + Dropout) ---
    hidden_dim: int = 256
    dropout: float = 0.5
    num_classes: int = 2

    # --- training ---
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    early_stopping_patience: int = 3

    # --- augmentation ---
    horizontal_flip: bool = True

    # --- misc ---
    seed: int = 42
    output_dir: str = "outputs"
    device: str = "cuda"              # "cuda" | "cpu"

    def __post_init__(self) -> None:
        if self.model not in ("mobilenet", "resnet"):
            raise ValueError(f"Unknown model {self.model!r} (expected mobilenet|resnet)")
        if self.mode not in ("frozen", "finetune"):
            raise ValueError(f"Unknown mode {self.mode!r} (expected frozen|finetune)")

    @property
    def experiment_name(self) -> str:
        """e.g. 'mobilenet_frozen' — used for output dirs and log labels."""
        return f"{self.model}_{self.mode}"

    @property
    def experiment_dir(self) -> Path:
        return Path(self.output_dir) / self.experiment_name

    def to_yaml(self, path: str | Path) -> None:
        """Persist the resolved config next to the run's artifacts."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)


_VALID_KEYS = {f.name for f in fields(Config)}


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    unknown = set(data) - _VALID_KEYS
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
    return data


def build_config(argv: list[str] | None = None) -> Config:
    """Parse CLI args + optional YAML into one Config (CLI wins over YAML)."""
    parser = argparse.ArgumentParser(description="Driver drowsiness detection")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file (see configs/).")
    parser.add_argument("--model", choices=["mobilenet", "resnet"])
    parser.add_argument("--mode", choices=["frozen", "finetune"])
    parser.add_argument("--data-dir", dest="data_dir")
    parser.add_argument("--image-size", dest="image_size", type=int)
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--num-workers", dest="num_workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu"])
    parser.add_argument("--output-dir", dest="output_dir")
    args = parser.parse_args(argv)

    values: dict = {}
    if args.config:
        values.update(_load_yaml(args.config))
    # Only flags the user actually passed (non-None) override YAML/defaults.
    for key, val in vars(args).items():
        if key == "config" or val is None:
            continue
        values[key] = val

    return Config(**values)
