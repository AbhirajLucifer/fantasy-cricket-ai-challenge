from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from fantasy_cricket.model import FEATURES, TARGET, BlendedRegressor, build_mean_model


root = Path(__file__).resolve().parents[1]
source = root / "model_artifacts/fantasy_t20_model_v2_2025-12-31.pkl"
target = root / "model_artifacts/fantasy_t20_model_v3_2025-12-31.pkl"
artifact = joblib.load(source)
frame = pd.read_csv(root / "data/processed/v2_player_match_features.csv", low_memory=False)
mean_model = build_mean_model()
mean_model.fit(frame[FEATURES], frame[TARGET])
artifact["model"] = BlendedRegressor(artifact["model"], mean_model, absolute_weight=.6)
artifact["model_variant"] = "0.6 MAE-XGBoost + 0.4 mean-HistGradientBoosting"
artifact["validation_metrics"] = {
    "mae": 25.24970455415833,
    "rmse": 34.26307063983798,
    "spearman": 0.36018206630592403,
    "bias": -4.09702090454277,
    "n": 85231,
    "period_start": "2023-10-06",
    "period_end": "2025-12-31",
}
joblib.dump(artifact, target)

metrics_path = root / "reports/v3_metrics.json"
metrics = json.loads((root / "reports/v2_metrics.json").read_text(encoding="utf-8"))
metrics["validation"] = artifact["validation_metrics"]
metrics["model_variant"] = artifact["model_variant"]
metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(target)
print(json.dumps(artifact["validation_metrics"], indent=2))
