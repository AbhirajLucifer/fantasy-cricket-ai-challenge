from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier, XGBRegressor

from . import SEED


TARGET = "fantasy_points"
CATEGORICAL = ["team", "opponent", "venue", "match_type", "gender", "role"]
NUMERIC = [
    "experience", "career_avg", "batting_avg", "bowling_avg", "fielding_avg",
    "recent_3", "recent_5", "recent_10", "recent_std", "venue_avg",
    "opponent_avg", "appearance_rate_10", "rest_days", "month",
    "recent_batting_5", "recent_bowling_5", "recent_fielding_5",
    "recent_peak_10", "recent_floor_10", "form_trend", "team_matches",
    "team_share", "matches_last_30", "matches_last_90",
]
FEATURES = CATEGORICAL + NUMERIC


def set_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)


def _preprocessor(selected: list[str]) -> ColumnTransformer:
    categorical = [name for name in CATEGORICAL if name in selected]
    numeric = [name for name in NUMERIC if name in selected]
    return ColumnTransformer([
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), categorical),
        ("num", SimpleImputer(strategy="median"), numeric),
    ])


def build_model(feature_names: list[str] | None = None) -> Pipeline:
    set_seeds()
    selected = feature_names or FEATURES
    reg = XGBRegressor(
        objective="reg:absoluteerror", n_estimators=700, learning_rate=.05,
        max_depth=7, min_child_weight=12, subsample=.85, colsample_bytree=.9,
        reg_lambda=5, reg_alpha=.1, random_state=SEED, n_jobs=-1, tree_method="hist",
    )
    return Pipeline([("preprocess", _preprocessor(selected)), ("regressor", reg)])


def build_mean_model(feature_names: list[str] | None = None) -> Pipeline:
    set_seeds()
    selected = feature_names or FEATURES
    regressor = HistGradientBoostingRegressor(
        learning_rate=.06, max_iter=350, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=SEED,
    )
    return Pipeline([("preprocess", _preprocessor(selected)), ("regressor", regressor)])


@dataclass
class BlendedRegressor:
    absolute_model: Any
    mean_model: Any
    absolute_weight: float = 0.6

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        absolute = np.asarray(self.absolute_model.predict(frame), dtype=float)
        mean = np.asarray(self.mean_model.predict(frame), dtype=float)
        return self.absolute_weight * absolute + (1.0 - self.absolute_weight) * mean


def build_selection_model(feature_names: list[str] | None = None) -> Pipeline:
    set_seeds()
    selected = feature_names or FEATURES
    classifier = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", n_estimators=400,
        learning_rate=.05, max_depth=6, min_child_weight=8, subsample=.85,
        colsample_bytree=.9, reg_lambda=5, random_state=SEED, n_jobs=-1,
        tree_method="hist",
    )
    return Pipeline([("preprocess", _preprocessor(selected)), ("classifier", classifier)])


def chronological_split(frame: pd.DataFrame, validation_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(frame["match_date"].unique())
    split = max(1, int(len(dates) * (1 - validation_fraction)))
    boundary = dates[min(split, len(dates) - 1)]
    train = frame[frame["match_date"] < boundary].copy()
    valid = frame[frame["match_date"] >= boundary].copy()
    if train.empty or valid.empty:
        raise ValueError("Need matches on at least two dates for chronological validation")
    return train, valid


def metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    rho = spearmanr(y_true, y_pred, nan_policy="omit").statistic
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "spearman": float(0.0 if np.isnan(rho) else rho),
        "n": int(len(y_true)),
    }
