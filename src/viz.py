"""Crtanje grafika - svaka funkcija cuva PNG (bez prikaza na ekranu)."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402


def plot_class_distribution(class_names, counts, save_path):
    """Bar chart broja slika po klasi."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(counts)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(class_names, counts, color="#2a78d6")
    ax.set_ylabel("Broj slika")
    ax.set_title("Distribucija klasa - DDD")
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, c,
                f"{c:,}\n({c / total:.1%})", ha="center", va="bottom")
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_sample_images(data_dir, class_names, save_path, per_class=8, seed=42):
    """Mreza nasumicnih primera slika po klasi."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows, cols = len(class_names), per_class

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.7))
    for r, cls in enumerate(class_names):
        files = [p for p in (Path(data_dir) / cls).iterdir() if p.is_file()]
        picks = rng.sample(files, min(per_class, len(files)))
        for c in range(cols):
            ax = axes[r][c] if rows > 1 else axes[c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c < len(picks):
                ax.imshow(Image.open(picks[c]))
            if c == 0:
                ax.set_ylabel(cls, fontsize=11)
    fig.suptitle("Primeri slika po klasama")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
