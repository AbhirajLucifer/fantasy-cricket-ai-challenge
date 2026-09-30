# Fantasy Cricket T20 Model

## Ready artifact

- Model: `model_artifacts/fantasy_t20_model_2025-12-31.pkl`
- Training data: Cricsheet `all_json.zip`, filtered to `match_type == "T20"`
- Training cutoff: 2025-12-31
- Player-match rows: 281,663
- Chronological validation: 2023-10-06 through 2025-12-31 (85,231 rows)
- MAE: 26.095 fantasy points
- RMSE: 34.008 fantasy points
- Spearman correlation: 0.3600

The serialized artifact contains the fitted model, leakage-safe feature store,
feature list, training dates, validation metrics, ablations, and permutation
importance. Keep `src` on `PYTHONPATH` when loading it directly with
`joblib.load(...)`; use `src/evaluate.py` to score a fixture and select a
constraint-valid fantasy XI.

## Example

```powershell
python src/evaluate.py --model model_artifacts/fantasy_t20_model_2025-12-31.pkl --fixtures data/test_fixtures.json --output results/submission.csv
```

The Colab notebook contains the complete reproducible download, training,
testing, packaging, and download cells.
