PYTHON ?= python
TRAIN_START ?= 2000-01-01
TRAIN_END ?= 2025-12-31
CUTOFF ?= 2025-12-31
TEST_FILE ?= data/raw/test_fixtures.json
MODEL ?= model_artifacts/model_$(TRAIN_END).pkl
OUTPUT ?= results/hidden_predictions.csv

.PHONY: setup download check test train evaluate all
setup:
	$(PYTHON) -m pip install -r requirements.txt

download:
	$(PYTHON) src/download_data.py --raw-dir data/raw --manifest data/raw/download_manifest.json

check:
	$(PYTHON) src/check_pipeline.py --train-start $(TRAIN_START) --train-end $(TRAIN_END) --cutoff $(CUTOFF) --test-file $(TEST_FILE)

test:
	$(PYTHON) -m pytest -q

train:
	$(PYTHON) src/train.py --train-start $(TRAIN_START) --train-end $(TRAIN_END) --raw-dir data/raw --model-path $(MODEL) --metrics-path reports/v3_metrics.json

evaluate:
	$(PYTHON) src/evaluate.py --model-path $(MODEL) --test-file $(TEST_FILE) --output $(OUTPUT)

all: download check test train evaluate
