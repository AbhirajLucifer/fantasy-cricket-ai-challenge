from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from fantasy_cricket.model import FEATURES, _preprocessor, build_model, chronological_split


def score(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(mean_squared_error(actual, prediction) ** .5),
        "spearman": float(spearmanr(actual, prediction).statistic),
        "bias": float(np.mean(prediction - actual)),
    }


root = Path(__file__).resolve().parents[1]
frame = pd.read_csv(root / "data/processed/v2_player_match_features.csv", low_memory=False)
train, valid = chronological_split(frame)
absolute_model = build_model()
mean_model = Pipeline([
    ("preprocess", _preprocessor(FEATURES)),
    ("regressor", HistGradientBoostingRegressor(
        learning_rate=.06, max_iter=350, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=42,
    )),
])
absolute_model.fit(train[FEATURES], train["fantasy_points"])
mean_model.fit(train[FEATURES], train["fantasy_points"])
absolute = np.maximum(0, absolute_model.predict(valid[FEATURES]))
mean = np.maximum(0, mean_model.predict(valid[FEATURES]))
actual = valid["fantasy_points"].to_numpy()
results = {"absolute": score(actual, absolute), "mean": score(actual, mean)}
for absolute_weight in (.5, .6, .7, .8, .9):
    prediction = absolute_weight * absolute + (1 - absolute_weight) * mean
    results[f"absolute_{absolute_weight:.1f}"] = score(actual, prediction)
path = root / "reports/experiment_blend_v2.json"
path.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
