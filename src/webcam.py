"""Real-time demo: kamera -> YuNet detekcija lica -> CNN -> Drowsy/Awake.

Ucitava model snimljen sa src.train_final, radi isto pretprocesiranje kao trening
(ukljucujuci grayscale ako je model tako treniran) i vremenski uglacava predikcije
da ne trepere. Odluku donosi CNN; uglacavanje je samo debounce izlaza.

Pokretanje:
    python -m src.webcam --model outputs/final_resnet_finetune.pt
    python -m src.webcam --model outputs/final_mobilenet_finetune.pt --roi eyes
    (izlaz: taster 'q')
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch
from PIL import Image

from src.config import Config
from src.crop_eyes import _eye_box, _make_detector
from src.data import _build_transforms
from src.models import build_model
from src.utils import get_device


def _face_box(faces, w, h, margin=0.15):
    """Okvir lica iz YuNet detekcije, sa malom marginom; None ako nema lica."""
    if faces is None or len(faces) == 0:
        return None
    f = faces[np.argmax(faces[:, 14])]
    x, y, bw, bh = f[0], f[1], f[2], f[3]
    mx, my = bw * margin, bh * margin
    x0, y0 = max(0, int(x - mx)), max(0, int(y - my))
    x1, y1 = min(w, int(x + bw + mx)), min(h, int(y + bh + my))
    return (x0, y0, x1, y1) if x1 - x0 > 8 and y1 - y0 > 8 else None


def main():
    ap = argparse.ArgumentParser(description="Real-time demo detekcije pospanosti.")
    ap.add_argument("--model", required=True, help="Putanja do .pt (src.train_final).")
    ap.add_argument("--yunet", default="assets/face_detection_yunet_2023mar.onnx")
    ap.add_argument("--roi", choices=["face", "eyes"], default="face",
                    help="Sta se secem i salje modelu (uskladi sa treningom).")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--smooth", type=int, default=15, help="Broj frejmova za uglacavanje.")
    ap.add_argument("--threshold", type=float, default=0.5, help="Prag za klasu Drowsy.")
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    cfg = Config(**ckpt["config"])
    classes = ckpt["classes"]
    drowsy_idx = classes.index("Drowsy")
    dev_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = get_device(dev_str)

    model = build_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    # isto pretprocesiranje kao na eval strani treninga (grayscale ako je model tako ucen)
    eval_tf = _build_transforms(cfg.image_size, cfg.horizontal_flip, False, cfg.grayscale)[1]
    detector = _make_detector(args.yunet)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Ne mogu da otvorim kameru {args.camera}.")
        return
    print(f"Model {cfg.experiment_name} | ROI {args.roi} | uredjaj {device} | 'q' za izlaz")

    probs = deque(maxlen=args.smooth)
    t_prev, fps = time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)
        box = _eye_box(faces, w, h) if args.roi == "eyes" else _face_box(faces, w, h)

        if box is not None:
            x0, y0, x1, y1 = box
            rgb = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            x = eval_tf(Image.fromarray(rgb)).unsqueeze(0).to(device)
            with torch.no_grad():
                p = torch.softmax(model(x), 1)[0, drowsy_idx].item()
            probs.append(p)
            avg = sum(probs) / len(probs)
            drowsy = avg >= args.threshold
            color = (0, 0, 255) if drowsy else (0, 200, 0)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            cv2.putText(frame, f"{'DROWSY' if drowsy else 'AWAKE'} {avg * 100:.0f}%",
                        (x0, max(24, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if drowsy and len(probs) == probs.maxlen:
                cv2.putText(frame, "! POSPANOST !", (20, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        else:
            probs.clear()
            cv2.putText(frame, "Nema lica", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now
        cv2.putText(frame, f"{fps:.0f} FPS", (w - 110, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Drowsiness demo (q za izlaz)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
