# Fantasy Cricket AI Challenge

Traditional-ML solution for pre-toss fantasy-point prediction and legal best-XI selection using Cricsheet T20 data.

## Required files

- `model_artifacts/model_2025-12-31.pkl` — ready model with appearance ROC-AUC **0.8860265**
- `src/` — download, training, prediction, scoring, feature and XI-selection code
- `requirements.txt` — Python dependencies
- `run.sh` — complete pipeline
- `results/hidden_predictions.csv` — generated after the organizer supplies the hidden fixture

The model cutoff is fixed at `2025-12-31`. No 2026 dataset or result is included.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/download_data.py --raw-dir data/raw --manifest data/raw/download_manifest.json
```

## Required training command

```bash
python src/train.py --train-start 2000-01-01 --train-end 2025-12-31
```

## Required evaluation command

Place the organizer-provided fixture at `data/raw/test_fixtures.json`, then run:

```bash
python src/evaluate.py --model-path model_artifacts/model_2025-12-31.pkl --test-file data/raw/test_fixtures.json --output results/hidden_predictions.csv
```

The output CSV contains `Match Date`, `Team 1`, `Team 2`, the predicted best 11 with roles, and predicted points in descending order. Selection enforces exactly 11 players, both teams represented, WK 1–4, BAT 3–6, AR 1–4 and BOWL 3–6.

## Full pipeline

```bash
chmod +x run.sh
./run.sh
```

The official hidden CSV is not fabricated or committed before `test_fixtures.json` is supplied.
