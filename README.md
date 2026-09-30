# Optimal Fantasy Cricket XI

This repository trains a leakage-safe player fantasy-points model from Cricsheet JSON, predicts every member of two announced squads before the toss, and solves the final XI as an exact constrained optimization problem.

## What the pipeline does

1. Downloads JSON match archives only from `cricsheet.org` and records SHA-256 hashes.
2. Reconstructs the supplied fantasy score from ball-by-ball events, excluding Super Overs.
3. Builds each training row from information available strictly before that match.
4. Uses a chronological holdout to report MAE, RMSE, and Spearman rank correlation.
5. Retrains on all permitted data through the requested training end date.
6. Scores full pre-toss squads and selects exactly 11 players with the required role bounds and representation from both teams.

The final v3 system is a two-stage model. It estimates conditional fantasy points and separately learns the probability that each squad member will play; pre-toss expected points are `P(plays) × conditional points`. Non-playing players therefore receive an actual score of zero in evaluation.

## Final model

The ready artifact is `model_artifacts/fantasy_t20_model_v3_2025-12-31.pkl`. Its training cutoff is 2025-12-31. Post-cutoff evaluation datasets and results are intentionally not distributed in this repository.

## Repository layout

```text
data/raw/          Cricsheet archives, extracted JSON, hidden fixtures
data/processed/    Player-match feature table
model_artifacts/   Serialized model and pre-cutoff feature state
results/           Prediction CSV files
src/               Download, scoring, feature, train, and evaluation code
tests/             Scoring, optimizer, and leakage tests
reports/           Pre-cutoff validation metrics
```

## Setup

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Data

Download every Cricsheet JSON match archive:

```bash
python src/download_data.py \
  --raw-dir data/raw \
  --manifest data/raw/download_manifest.json
```

To use a smaller official archive, repeat `--url`, for example `https://cricsheet.org/downloads/t20s_json.zip`. The downloader rejects non-Cricsheet URLs.

## Train

```bash
python src/train.py \
  --train-start 2000-01-01 \
  --train-end 2025-12-31
```

Training aborts if `--train-end` exceeds `2025-12-31`. The feature builder sorts matches chronologically, creates a row from the current state, then updates that state with the match result. This prevents current or future performance from entering the row.

## Hidden evaluation

Place the released fixture file at `data/raw/test_fixtures.json`, then run:

```bash
python src/evaluate.py \
  --model-path model_artifacts/fantasy_t20_model_v3_2025-12-31.pkl \
  --test-file data/raw/test_fixtures.json \
  --output results/hidden_predictions.csv
```

The parser accepts either a list or an object containing `fixtures`, `matches`, or `data`. Each fixture should contain a date, two team names, and squads. A squad member may be a plain name or an object with `name`, optional `player_id`, and `role`. Supported role aliases include `WK`, `BAT`, `AR`, and `BOWL`.

The output has the required match fields. The selected players are ordered by predicted points. Players and roles are encoded as JSON in one CSV cell so names containing commas remain safe.

An executable schema example is provided at `tests/test_fixture.json`, with its verified output at `results/example_predictions.csv`. This example proves the complete model-to-CSV path; it is not a substitute for the official hidden-test predictions.

### Submission checklist

- `README.md` documents setup, dependencies, training, evaluation, and run commands.
- `requirements.txt` pins the Python dependencies.
- `run.sh` and `Makefile` provide reproducible entry points.
- `model_artifacts/fantasy_t20_model_v3_2025-12-31.pkl` is the ready model.
- `results/example_predictions.csv` verifies the required output schema.
- `results/hidden_predictions.csv` is generated from the official hidden fixture using the command above. It must not be fabricated from the example fixture.

## One-command execution

```bash
chmod +x run.sh
./run.sh
```

Every date and path can be overridden with environment variables documented in `run.sh`. A Makefile exposes the same stages.

## Reproducibility and leakage audit

- The global random seed is `42` and every stochastic estimator receives it.
- Raw download URLs and hashes are recorded.
- Training uses a date-based split, never a random row split.
- `tests/test_leakage.py` verifies that a post-cutoff training end date fails.
- `train.py` ignores matches after the policy cutoff and rejects a later requested end date.
- Feature state updates occur only after every match on the same date has been featurized, preventing same-day leakage.
- Toss results, final innings information, current-match playing actions, and future aggregates are absent from features.

Run checks with `python -m pytest -q`.

## Metric choice

MAE is the primary point-forecast metric because its units are fantasy points and it is less dominated by rare explosive performances than RMSE. Spearman correlation is the complementary selection metric because the downstream decision depends on ranking players correctly. RMSE remains useful for exposing large misses.

## Important assumptions

- Cricsheet is authoritative for match events and participating players.
- A one-fielder run out is treated as direct; multi-fielder run outs are treated as assisted because Cricsheet does not encode a direct-hit flag.
- When the hidden fixture supplies roles, those roles take precedence. Otherwise the pipeline infers roles from prior batting, bowling, and fielding contribution only.
- Unknown players receive conservative cold-start features and remain eligible.

## Deliverables

- Validation metrics after training: `reports/v3_metrics.json`
- Verified output example: `results/example_predictions.csv`
- Hidden predictions after evaluation: `results/hidden_predictions.csv`
