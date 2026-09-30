from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from fantasy_cricket.features import build_training_datasets
from fantasy_cricket.io import load_matches
from fantasy_cricket.model import (
    FEATURES, TARGET, BlendedRegressor, build_mean_model, build_model, build_selection_model,
    chronological_split, metrics, set_seeds,
)


HARD_CUTOFF = date(2025, 12, 31)


def main() -> None:
    set_seeds()
    parser = argparse.ArgumentParser(description="Train the fantasy-points model with chronological validation")
    parser.add_argument("--train-start", type=date.fromisoformat, required=True)
    parser.add_argument("--train-end", type=date.fromisoformat, required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-path", type=Path, default=Path("data/processed/player_match_features.csv"))
    parser.add_argument("--selection-path", type=Path, default=Path("data/processed/player_selection_features.csv"))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--metrics-path", type=Path, default=Path("reports/metrics.json"))
    parser.add_argument(
        "--match-type", action="append", dest="match_types",
        help="Cricsheet match type to include; repeat to include more than one (default: T20)",
    )
    args = parser.parse_args()
    if args.train_start > args.train_end:
        parser.error("--train-start must not be after --train-end")
    if args.train_end > HARD_CUTOFF:
        parser.error(f"--train-end must be on or before {HARD_CUTOFF}")

    match_types = set(args.match_types or ["T20"])
    matches = load_matches(args.raw_dir, match_types=match_types)
    if not matches:
        raise SystemExit(f"No Cricsheet match JSON files found under {args.raw_dir}")
    frame, selection_frame, store = build_training_datasets(
        matches, args.train_start, args.train_end, HARD_CUTOFF, match_types=match_types,
    )
    if len(frame) < 50:
        raise SystemExit(f"Only {len(frame)} player-match rows found; at least 50 are required")
    train, valid = chronological_split(frame)
    absolute_model = build_model()
    mean_model = build_mean_model()
    absolute_model.fit(train[FEATURES], train[TARGET])
    mean_model.fit(train[FEATURES], train[TARGET])
    model = BlendedRegressor(absolute_model, mean_model, absolute_weight=0.6)
    validation_prediction = np.maximum(0.0, model.predict(valid[FEATURES]))
    validation_metrics = metrics(valid[TARGET], validation_prediction)
    validation_metrics["period_start"] = str(valid["match_date"].min())
    validation_metrics["period_end"] = str(valid["match_date"].max())

    selection_train, selection_valid = chronological_split(selection_frame)
    selection_model = build_selection_model()
    selection_model.fit(selection_train[FEATURES], selection_train["selected"])
    selection_probability = selection_model.predict_proba(selection_valid[FEATURES])[:, 1]
    selection_metrics = {
        "roc_auc": float(roc_auc_score(selection_valid["selected"], selection_probability)),
        "brier": float(brier_score_loss(selection_valid["selected"], selection_probability)),
        "log_loss": float(log_loss(selection_valid["selected"], selection_probability)),
        "rows": int(len(selection_valid)),
        "positive_rate": float(selection_valid["selected"].mean()),
        "period_start": str(selection_valid["match_date"].min()),
        "period_end": str(selection_valid["match_date"].max()),
    }

    median_prediction = np.full(len(valid), float(train[TARGET].median()))
    recent_prediction = valid["recent_5"].to_numpy()
    ablations = {
        "constant_median": metrics(valid[TARGET], median_prediction),
        "recent_5_only": metrics(valid[TARGET], recent_prediction),
    }
    no_context = [name for name in FEATURES if name not in {"team", "opponent", "venue", "venue_avg", "opponent_avg"}]
    no_context_model = build_model(no_context)
    no_context_model.fit(train[no_context], train[TARGET])
    ablations["model_without_team_venue_opponent"] = metrics(valid[TARGET], no_context_model.predict(valid[no_context]))

    sample = valid.sample(n=min(2000, len(valid)), random_state=42)
    importance = permutation_importance(
        model, sample[FEATURES], sample[TARGET], scoring="neg_mean_absolute_error",
        n_repeats=3, random_state=42,
    )
    feature_importance = sorted(
        ({"feature": name, "mae_increase": float(value)} for name, value in zip(FEATURES, importance.importances_mean)),
        key=lambda item: item["mae_increase"], reverse=True,
    )

    final_absolute_model = build_model()
    final_mean_model = build_mean_model()
    final_absolute_model.fit(frame[FEATURES], frame[TARGET])
    final_mean_model.fit(frame[FEATURES], frame[TARGET])
    final_model = BlendedRegressor(final_absolute_model, final_mean_model, absolute_weight=0.6)
    final_selection_model = build_selection_model()
    final_selection_model.fit(selection_frame[FEATURES], selection_frame["selected"])
    model_path = args.model_path or Path(f"model_artifacts/fantasy_t20_model_v3_{args.train_end.isoformat()}.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": final_model,
        "selection_model": final_selection_model,
        "feature_store": store,
        "features": FEATURES,
        "train_start": args.train_start.isoformat(),
        "train_end": args.train_end.isoformat(),
        "cutoff": HARD_CUTOFF.isoformat(),
        "match_types": sorted(match_types),
        "validation_metrics": validation_metrics,
        "selection_metrics": selection_metrics,
        "ablations": ablations,
        "permutation_importance": feature_importance,
        "model_variant": "0.6 MAE-XGBoost + 0.4 mean-HistGradientBoosting",
    }
    joblib.dump(artifact, model_path)
    args.processed_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.processed_path, index=False)
    args.selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_frame.to_csv(args.selection_path, index=False)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_report = {
        "dataset": {
            "rows": len(frame), "train_rows": len(train), "validation_rows": len(valid),
            "selection_rows": len(selection_frame), "match_types": sorted(match_types),
        },
        "validation": validation_metrics,
        "selection_validation": selection_metrics,
        "ablations": ablations,
        "permutation_importance": feature_importance,
        "model_variant": "0.6 MAE-XGBoost + 0.4 mean-HistGradientBoosting",
    }
    args.metrics_path.write_text(json.dumps(experiment_report, indent=2), encoding="utf-8")
    print(json.dumps({"model_path": str(model_path), "rows": len(frame), **validation_metrics}, indent=2))


if __name__ == "__main__":
    main()
