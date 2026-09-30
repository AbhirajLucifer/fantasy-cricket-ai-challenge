# Prediction outputs

`example_predictions.csv` was generated from `tests/test_fixture.json` and verifies the complete serialized-model, constrained-XI, and CSV-export path.

For the challenge submission, place the official hidden fixture at `data/raw/test_fixtures.json` and run:

```bash
python src/evaluate.py \
  --model-path model_artifacts/fantasy_t20_model_v3_2025-12-31.pkl \
  --test-file data/raw/test_fixtures.json \
  --output results/hidden_predictions.csv
```

The official `hidden_predictions.csv` is intentionally absent until the official hidden fixture is supplied. The example output must not be submitted as the hidden-test prediction file.
