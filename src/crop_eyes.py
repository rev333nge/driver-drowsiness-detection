"""Preprocessing: iseci ROI oko ociju (YuNet) u novi dataset data_eyes/.

Detektuje LICE (radi i sa zatvorenim ocima), uzme tacke levog/desnog oka i
iseca zonu oko oba oka (obrve + podocnjaci). Ako lice nije nadjeno, koristi
geometrijski fallback (gornja-srednja zona slike), da se nijedna slika ne izgubi.
Imena fajlova se cuvaju ista, pa subject-wise CV i dalje radi.

Pokretanje:
    python -m src.crop_eyes
    python -m src.crop_eyes --data-dir data --out-dir data_eyes
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def _make_detector(model_path):
    # (0,0) input size -> postavlja se po slici sa setInputSize
    return cv2.FaceDetectorYN.create(model_path, "", (0, 0), 0.5, 0.3, 5000)


def _clamp(x0, y0, x1, y1, w, h):
    x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    x1, y1 = min(w, int(round(x1))), min(h, int(round(y1)))
    if x1 - x0 < 8 or y1 - y0 < 8:  # degenerisano -> ceo kadar
        return 0, 0, w, h
    return x0, y0, x1, y1


def _eye_box(faces, w, h):
    """Zona oko oba oka iz YuNet tacaka: (x0,y0,x1,y1) ili None."""
    if faces is None or len(faces) == 0:
        return None
    f = faces[np.argmax(faces[:, 14])]        # lice sa najvecim score
    rx, ry, lx, ly = f[4], f[5], f[6], f[7]   # desno oko, levo oko
    cx, cy = (rx + lx) / 2, (ry + ly) / 2
    d = float(np.hypot(lx - rx, ly - ry)) or 1.0  # razmak medju ocima
    # sirina 2d (margina ~d/2 sa strane), visina 1.3d (obrve gore, podocnjaci dole)
    return _clamp(cx - d, cy - 0.6 * d, cx + d, cy + 0.7 * d, w, h)


def _fallback_box(w, h):
    """Bez detekcije: gornja-srednja zona (oci su tu kod portreta iz baze)."""
    return _clamp(0.10 * w, 0.20 * h, 0.90 * w, 0.55 * h, w, h)


def process(data_dir, out_dir, model_path):
    detector = _make_detector(model_path)
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    classes = [d.name for d in sorted(data_dir.iterdir()) if d.is_dir()]
    print(f"Klase: {classes} | model: {model_path}")

    t0 = time.time()
    total = detected = fallback = failed = 0
    for cls in classes:
        src_cls, dst_cls = data_dir / cls, out_dir / cls
        dst_cls.mkdir(parents=True, exist_ok=True)
        files = [p for p in sorted(src_cls.iterdir()) if p.suffix.lower() in IMG_EXT]
        for i, p in enumerate(files, 1):
            img = cv2.imread(str(p))
            if img is None:
                failed += 1
                continue
            h, w = img.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(img)
            box = _eye_box(faces, w, h)
            if box is None:
                box, fallback = _fallback_box(w, h), fallback + 1
            else:
                detected += 1
            x0, y0, x1, y1 = box
            cv2.imwrite(str(dst_cls / p.name), img[y0:y1, x0:x1])
            total += 1
            if i % 2000 == 0:
                print(f"  {cls}: {i}/{len(files)}")
        print(f"{cls}: {len(files)} slika obradjeno")

    dt = time.time() - t0
    det_pct = 100 * detected / max(total, 1)
    print(f"\nGotovo za {dt:.0f}s | ukupno {total} | detektovano lice {detected} "
          f"({det_pct:.1f}%) | fallback {fallback} | neuspelo {failed}")
    print(f"Novi dataset: {out_dir}  (treniraj sa --data-dir {out_dir})")


def main():
    ap = argparse.ArgumentParser(description="Iseci ROI oko ociju u novi dataset.")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="data_eyes")
    ap.add_argument("--model", default="assets/face_detection_yunet_2023mar.onnx")
    args = ap.parse_args()
    process(args.data_dir, args.out_dir, args.model)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
