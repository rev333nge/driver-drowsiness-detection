# Detekcija pospanosti vozaca (CNN)

Real-time detekcija pospanosti vozaca sa lica, uz poredjenje dve CNN arhitekture
(MobileNetV2 vs ResNet50) u dva rezima transfer learning-a (frozen vs finetune),
sa postenom evaluacijom (bez curenja podataka) i OpenCV demo-om uzivo.

## Specifikacija

- **Zadatak:** klasifikacija `Drowsy` / `Non Drowsy` sa portreta lica vozaca.
- **Dataset:** DDD - 41.793 slike, 2 klase, **28 razlicitih vozaca** (osoba kodirana u imenu fajla).
- **4 eksperimenta:** {MobileNetV2, ResNet50} x {frozen, finetune}, jedan config-driven kod.
- **Evaluacija:** subject-wise **5-fold cross-validacija** (vozaci u testu su nevidjeni - bez data leakage-a).

## Struktura

```
src/
  config.py       konfiguracija eksperimenta (defaults < YAML < CLI)
  data.py         ucitavanje DDD + subject-wise podela (bez curenja)
  models.py       fabrika modela (MobileNetV2/ResNet50, frozen/finetune)
  engine.py       jedna epoha treninga/validacije + early stopping
  train.py        trening jednog eksperimenta kroz 5-fold CV
  train_final.py  finalni model na (skoro) svim podacima -> .pt za demo
  evaluate.py     metrike + uporedna tabela iz rezultata CV-a
  metrics.py      klasifikacione + prakticne metrike (params, velicina, FPS)
  crop_eyes.py    YuNet detekcija lica + isecanje ROI oko ociju (ablacija)
  webcam.py       real-time demo: kamera -> CNN -> Drowsy/Awake
  viz.py, utils.py  grafici i pomocne funkcije
configs/          4 YAML fajla (po jedan za svaki eksperiment)
scripts/run_all.ps1   pokrece sva 4 eksperimenta redom
assets/           YuNet model (onnx) za detekciju lica
report.ipynb      izvestaj: rezultati, grafici, zakljucci
report_assets/    artefakti koje izvestaj koristi (JSON + grafici)
```

## Rezultati (subject-wise 5-fold CV)

| eksperiment | acc | F1 | parametri | velicina | FPS |
|---|---|---|---|---|---|
| mobilenet_frozen | 0.565 | 0.621 | 2.5M | 9.9 MB | 306 |
| mobilenet_finetune | 0.604 | 0.654 | 2.5M | 9.9 MB | 276 |
| resnet_frozen | 0.596 | **0.674** | 24M | 91.9 MB | 274 |
| resnet_finetune | **0.648** | 0.662 | 24M | 91.9 MB | 240 |

Dokaz curenja podataka (isti model, ista slika, razlicita podela):
**per-image (curenje) 99.96%** vs **subject-wise (posteno) 60.55%**.

## Zakljucci

- Posten (subject-independent) rezultat je ~**60-65%**; naduvanih ~99% u literaturi je posledica
  **curenja** (isti vozac zavrsi i u train i u test).
- **Transfer learning nosi glavninu:** frozen je blizu finetune-a.
- **10x manji MobileNetV2** daje uporediv F1 kao ResNet50 -> bolji izbor za real-time.
- Plafon je **broj razlicitih vozaca (28)**, ne arhitektura - potvrdjeno kroz 3 ablacije
  (regularizacija, augmentacija, ROI oko ociju) koje sve stoje oko 0.60.

## Pokretanje

Instalacija (CPU, radi na bilo kom racunaru):

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-cpu.txt
```

(sa NVIDIA GPU koristi `requirements.txt`). Dataset ide u `data/Drowsy` i `data/Non Drowsy`.

Trening i evaluacija:

```
python -m src.train --config configs/resnet_finetune.yaml   # jedan eksperiment
.\scripts\run_all.ps1                                       # sva 4 eksperimenta
python -m src.evaluate                                      # uporedna tabela
```

Real-time demo:

```
python -m src.train_final --config configs/mobilenet_finetune.yaml   # snimi model
python -m src.webcam --model outputs/final_mobilenet_finetune.pt      # kamera (izlaz: 'q')
```

Izvestaj: otvori `report.ipynb` u Jupyter-u ili VS Code i uradi **Run All**.
