#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_START="${TRAIN_START:-2000-01-01}"
TRAIN_END="${TRAIN_END:-2025-12-31}"
CUTOFF="${CUTOFF:-2025-12-31}"
TEST_FILE="${TEST_FILE:-data/raw/test_fixtures.json}"
MODEL_PATH="${MODEL_PATH:-model_artifacts/fantasy_t20_model_v3_${TRAIN_END}.pkl}"
OUTPUT_PATH="${OUTPUT_PATH:-results/hidden_predictions.csv}"

"$PYTHON_BIN" src/download_data.py --raw-dir data/raw --manifest data/raw/download_manifest.json
"$PYTHON_BIN" src/check_pipeline.py --train-start "$TRAIN_START" --train-end "$TRAIN_END" --cutoff "$CUTOFF" --test-file "$TEST_FILE"
"$PYTHON_BIN" -m pytest -q
"$PYTHON_BIN" src/train.py --train-start "$TRAIN_START" --train-end "$TRAIN_END" --raw-dir data/raw --model-path "$MODEL_PATH" --metrics-path reports/v3_metrics.json
"$PYTHON_BIN" src/evaluate.py --model-path "$MODEL_PATH" --test-file "$TEST_FILE" --output "$OUTPUT_PATH"
