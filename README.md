# Fantasy Cricket AI Challenge

Leakage-safe traditional-ML pipeline for predicting fantasy points from Cricsheet T20 data and selecting a valid pre-toss XI from the two announced squads.

## Mandatory deliverables

- Ready model: `model_artifacts/model_2025-12-31.pkl`
- Training and inference code: `src/`
- Dependencies: `requirements.txt`
- Full pipeline: `run.sh` (or `make all`)
- Validation evidence: `reports/v3_metrics.json`
- Technical report: `reports/technical_report.docx`
- Presentation: `presentation/fantasy_cricket_viva.pptx`
- Hidden predictions after the test file is supplied: `results/hidden_predictions.csv`

No 2026 match data is included. Training is hard-capped at 2025-12-31.

## Setup

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download Cricsheet JSON data:

```bash
python src/download_data.py --raw-dir data/raw --manifest data/raw/download_manifest.json
```

## Required training command

```bash
python src/train.py --train-start 2000-01-01 --train-end 2025-12-31
```

The command writes `model_artifacts/model_2025-12-31.pkl` and `reports/v3_metrics.json`. Dates later than the cutoff are rejected.

## Required hidden-test command

Place the organizer-supplied file at `data/raw/test_fixtures.json`, then run exactly:

```bash
python src/evaluate.py --model-path model_artifacts/model_2025-12-31.pkl --test-file data/raw/test_fixtures.json --output results/hidden_predictions.csv
```

The CSV contains `Match Date`, `Team 1`, `Team 2`, and the best 11 players in descending predicted-points order, including role and predicted points. The optimizer enforces exactly 11 players, both teams represented, WK 1–4, BAT 3–6, AR 1–4, and BOWL 3–6.

## One-command pipeline

```bash
chmod +x run.sh
./run.sh
```

The equivalent Make target is `make all`. Both download data, validate inputs, run tests, train, and create hidden predictions. The official `test_fixtures.json` must be present for the final evaluation stage.

## Model and validation

The submitted model is a deterministic blend of 60% MAE-XGBoost and 40% mean-loss HistGradientBoosting for conditional points, plus an XGBoost appearance classifier. Expected pre-toss points are `P(plays) × conditional points`.

Chronological validation (2023-10-06 to 2025-12-31):

- Point prediction: MAE 25.250, RMSE 34.263, Spearman 0.360
- Appearance model: ROC-AUC **0.886**, Brier 0.132, log loss 0.414
- Validation rows: 85,231 point rows and 206,145 appearance rows

MAE is the primary metric because it is measured directly in fantasy points. Spearman correlation is the complementary ranking metric used for XI selection.

## Leakage controls

- Date-based chronological validation; no random row split.
- Feature state is read before a match and updated only after all matches on that date are featurized.
- Toss, current-match actions, innings results, and future aggregates are excluded.
- Training after 2025-12-31 is rejected in code and tested in `tests/test_leakage.py`.
- Random seed `42` is used for reproducibility.

Run verification with:

```bash
python -m pytest -q
```

The official hidden fixture is not included in this repository. Therefore `results/hidden_predictions.csv` must be generated only from the organizer-provided file and must not be fabricated.
