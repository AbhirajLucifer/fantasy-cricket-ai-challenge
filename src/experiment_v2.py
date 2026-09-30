from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRanker, XGBRegressor

from fantasy_cricket.model import CATEGORICAL, FEATURES, NUMERIC, chronological_split


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), CATEGORICAL),
        ("num", SimpleImputer(strategy="median"), NUMERIC),
    ])


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    rho = spearmanr(actual, predicted).statistic
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** .5),
        "spearman": float(rho),
    }


def match_ranking(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    data = frame[["match_id", "fantasy_points"]].copy()
    data["prediction"] = prediction
    rhos, overlaps, captures = [], [], []
    for _, group in data.groupby("match_id"):
        rho = spearmanr(group["fantasy_points"], group["prediction"]).statistic
        rhos.append(0.0 if np.isnan(rho) else rho)
        predicted = group.nlargest(11, "prediction")
        actual = group.nlargest(11, "fantasy_points")
        overlaps.append(len(set(predicted.index) & set(actual.index)))
        captures.append(predicted["fantasy_points"].sum() / actual["fantasy_points"].sum())
    return {
        "match_spearman": float(np.mean(rhos)),
        "top11_overlap": float(np.mean(overlaps)),
        "top11_capture": float(np.mean(captures)),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    frame = pd.read_csv(root / "data/processed/full_t20_player_match_features.csv")
    train, valid = chronological_split(frame)
    pre = make_preprocessor()
    x_train = pre.fit_transform(train[FEATURES])
    x_valid = pre.transform(valid[FEATURES])
    y_train = train["fantasy_points"].to_numpy()
    y_valid = valid["fantasy_points"].to_numpy()
    output = {}

    configs = {
        "xgb_squared": "reg:squarederror",
        "xgb_absolute": "reg:absoluteerror",
    }
    predictions = {}
    for name, objective in configs.items():
        model = XGBRegressor(
            objective=objective, n_estimators=700, learning_rate=.05,
            max_depth=7, min_child_weight=12, subsample=.85,
            colsample_bytree=.9, reg_lambda=5, reg_alpha=.1,
            random_state=42, n_jobs=-1, tree_method="hist",
        )
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
        pred = np.maximum(0.0, model.predict(x_valid))
        predictions[name] = pred
        output[name] = {**metrics(y_valid, pred), **match_ranking(valid, pred)}

    # Ranking labels are within-match relevance, so the ranker focuses on XI ordering.
    order = np.argsort(train["match_id"].astype(str).to_numpy(), kind="stable")
    train_sorted = train.iloc[order]
    x_rank = x_train[order]
    relevance = train_sorted.groupby("match_id")["fantasy_points"].rank(method="average").to_numpy()
    groups = train_sorted.groupby("match_id", sort=False).size().to_numpy()
    ranker = XGBRanker(
        objective="rank:pairwise", n_estimators=500, learning_rate=.05,
        max_depth=6, min_child_weight=10, subsample=.85, colsample_bytree=.9,
        reg_lambda=5, random_state=42, n_jobs=-1, tree_method="hist",
    )
    ranker.fit(x_rank, relevance, group=groups, verbose=False)
    rank_score = ranker.predict(x_valid)
    output["xgb_ranker"] = match_ranking(valid, rank_score)

    # Re-rank the point regressor with the ranker, retaining point-scale calibration.
    base = predictions["xgb_squared"]
    for weight in (0.1, 0.2, 0.3, 0.4):
        blended = np.empty(len(valid))
        for _, idx in valid.groupby("match_id").groups.items():
            loc = valid.index.get_indexer(idx)
            base_group = base[loc]
            rank_group = pd.Series(rank_score[loc]).rank(pct=True).to_numpy()
            base_rank = pd.Series(base_group).rank(pct=True).to_numpy()
            target_rank = (1 - weight) * base_rank + weight * rank_group
            sorted_points = np.sort(base_group)
            blended[loc] = np.interp(target_rank, np.linspace(1 / len(loc), 1, len(loc)), sorted_points)
        output[f"blend_{weight:.1f}"] = {**metrics(y_valid, blended), **match_ranking(valid, blended)}

    path = root / "reports/experiment_v2.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
