#!/usr/bin/env bash
set -euo pipefail

python src/download_data.py --raw-dir data/raw --manifest data/raw/download_manifest.json
python src/train.py --train-start 2000-01-01 --train-end 2025-12-31
python src/evaluate.py --model-path model_artifacts/model_2025-12-31.pkl --test-file data/raw/test_fixtures.json --output results/hidden_predictions.csv
