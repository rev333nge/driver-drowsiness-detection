"""Evaluacija iz sacuvanih CV rezultata: metrike po eksperimentu + uporedna tabela.

Za svaki eksperiment spaja out-of-fold predikcije (svaka slika je predvidjena
modelom koji je nije trenirao) i racuna accuracy/precision/recall/f1 + confusion
matricu, plus prakticne metrike (params, velicina, FPS, vreme po epohi).

Pokretanje (posle treninga):
    python -m src.evaluate
"""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import confusion_matrix

from src import viz
from src.config import build_config
from src.metrics import classification_metrics, practical_metrics
from src.models import build_model
from src.utils import get_device, save_json


def _evaluate_experiment(exp_dir, device):
    data = json.loads((exp_dir / "cv_results.json").read_text(encoding="utf-8"))
    classes = data["classes"]
    drowsy = classes.index("Drowsy")  # pozitivna klasa: pospan vozac

    # out-of-fold predikcije preko svih foldova = predikcije za sve slike
    y_true = [y for f in data["folds"] for y in f["y_true"]]
    y_pred = [y for f in data["folds"] for y in f["y_pred"]]

    cls = classification_metrics(y_true, y_pred, pos_label=drowsy)
    cm = confusion_matrix(y_true, y_pred)
    viz.plot_confusion_matrix(cm, classes, exp_dir / "confusion_matrix.png",
                              title=data["experiment"])

    times = [h["time_s"] for f in data["folds"] for h in f["history"]]
    avg_epoch = sum(times) / len(times)

    # svez model (arhitektura) za params/velicinu/FPS - ne trebaju istrenirane tezine
    cfg = build_config(["--config", str(exp_dir / "config.yaml")])
    prac = practical_metrics(build_model(cfg).to(device), device, cfg.image_size)

    result = {
        "experiment": data["experiment"],
        "accuracy": round(cls["accuracy"], 4),
        "precision_drowsy": round(cls["precision"], 4),
        "recall_drowsy": round(cls["recall"], 4),
        "f1_drowsy": round(cls["f1"], 4),
        "test_acc_mean": round(data["test_acc_mean"], 4),
        "test_acc_std": round(data["test_acc_std"], 4),
        "avg_epoch_s": round(avg_epoch, 1),
        **prac,
    }
    save_json(result, exp_dir / "metrics.json")
    return result


def _latest_runs(out):
    """Najnoviji run po eksperimentu (podrzava ravan i timestampovan raspored)."""
    runs = list(out.glob("*/cv_results.json")) + list(out.glob("*/*/cv_results.json"))
    latest = {}
    for p in runs:
        name = json.loads(p.read_text(encoding="utf-8"))["experiment"]
        mtime = p.stat().st_mtime
        if name not in latest or mtime > latest[name][1]:
            latest[name] = (p.parent, mtime)
    return [latest[k][0] for k in sorted(latest)]


def main():
    cfg = build_config()
    device = get_device(cfg.device)
    out = Path(cfg.output_dir)

    exps = _latest_runs(out)
    if not exps:
        print(f"Nema cv_results.json u {out}/. Pokreni trening prvo (run_all.ps1).")
        return

    results = [_evaluate_experiment(d, device) for d in exps]

    header = (f"{'eksperiment':22s}{'acc':>7s}{'prec':>7s}{'rec':>7s}{'f1':>7s}"
              f"{'params':>12s}{'MB':>8s}{'FPS':>8s}{'s/ep':>8s}")
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(f"{r['experiment']:22s}{r['accuracy']:7.3f}{r['precision_drowsy']:7.3f}"
              f"{r['recall_drowsy']:7.3f}{r['f1_drowsy']:7.3f}{r['total_params']:12,}"
              f"{r['size_mb']:8.1f}{r['fps']:8.0f}{r['avg_epoch_s']:8.1f}")
    save_json(results, out / "comparison.json")
    print(f"\nSacuvano: {out / 'comparison.json'} + po eksperimentu metrics.json / confusion_matrix.png")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
