from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fantasy_cricket.features import match_date_of
from fantasy_cricket.io import load_matches
from fantasy_cricket.optimizer import select_best_xi
from fantasy_cricket.scoring import score_match


def metric_block(frame: pd.DataFrame, prediction: str = "predicted_points") -> dict[str, Any]:
    actual = frame["actual_points"].to_numpy()
    predicted = frame[prediction].to_numpy()
    errors = predicted - actual
    rho = spearmanr(actual, predicted, nan_policy="omit").statistic
    return {
        "rows": int(len(frame)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "spearman": float(0.0 if np.isnan(rho) else rho),
        "r2": float(r2_score(actual, predicted)),
        "bias": float(errors.mean()),
        "median_absolute_error": float(np.median(np.abs(errors))),
        "within_10_points_pct": float(100 * np.mean(np.abs(errors) <= 10)),
        "within_20_points_pct": float(100 * np.mean(np.abs(errors) <= 20)),
    }


def group_metrics(frame: pd.DataFrame, field: str) -> dict[str, dict[str, Any]]:
    return {str(key): metric_block(group) for key, group in frame.groupby(field, dropna=False)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe frozen-model backtest on 2026 T20 matches")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--rows-path", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    args = parser.parse_args()

    artifact = joblib.load(args.model_path)
    store = copy.deepcopy(artifact["feature_store"])
    matches = [
        match for match in load_matches(args.raw_dir, match_types={"T20"})
        if date(2026, 1, 1) <= match_date_of(match) <= date(2026, 12, 31)
    ]
    matches.sort(key=lambda match: (match_date_of(match), match.get("_match_id", "")))
    by_date: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for match in matches:
        by_date[match_date_of(match)].append(match)

    rows: list[dict[str, Any]] = []
    xi_results: list[dict[str, Any]] = []
    for match_date in sorted(by_date):
        pending_updates: list[tuple[str, str, str, str, date, dict[str, float]]] = []
        pending_lineups: list[tuple[str, set[str]]] = []
        for match in by_date[match_date]:
            info = match["info"]
            teams = info.get("teams", [])
            if len(teams) != 2:
                continue
            team1, team2 = teams
            venue = info.get("venue", "unknown")
            gender = info.get("gender", "unknown")
            ids = info.get("registry", {}).get("people", {})
            player_teams = {
                name: team for team, names in info.get("players", {}).items() for name in names
            }
            actuals = score_match(match)
            match_features: list[dict[str, Any]] = []
            names: list[str] = []
            scores: list[dict[str, float]] = []
            for name, score in actuals.items():
                team = player_teams.get(name)
                if team not in teams:
                    continue
                opponent = team2 if team == team1 else team1
                pid = ids.get(name, store.aliases.get(name, name))
                store.aliases[name] = pid
                feature = store.feature_row(
                    pid, team, opponent, venue, match_date, name=name,
                    match_type="T20", gender=gender,
                )
                match_features.append(feature)
                names.append(name)
                scores.append(score)

            if not match_features:
                continue
            feature_frame = pd.DataFrame(match_features)
            predicted = np.maximum(0.0, artifact["model"].predict(feature_frame[artifact["features"]]))
            candidates: list[dict[str, Any]] = []
            for feature, name, score, prediction in zip(match_features, names, scores, predicted):
                experience = int(feature["experience"])
                bucket = "0 (cold start)" if experience == 0 else "1-5" if experience <= 5 else "6-20" if experience <= 20 else "21+"
                row = {
                    "match_id": match.get("_match_id", "unknown"),
                    "match_date": match_date.isoformat(),
                    "month": match_date.strftime("%Y-%m"),
                    "gender": gender,
                    "team": feature["team"],
                    "opponent": feature["opponent"],
                    "venue": venue,
                    "player_id": feature["player_id"],
                    "player_name": name,
                    "role": feature["role"],
                    "experience": experience,
                    "experience_bucket": bucket,
                    "predicted_points": float(prediction),
                    "actual_points": float(score["fantasy_points"]),
                    "career_avg_baseline": max(0.0, float(feature["career_avg"])),
                    "recent_5_baseline": max(0.0, float(feature["recent_5"])),
                }
                row["error"] = row["predicted_points"] - row["actual_points"]
                row["absolute_error"] = abs(row["error"])
                rows.append(row)
                candidates.append({**row})
                opponent = team2 if feature["team"] == team1 else team1
                pending_updates.append((feature["player_id"], feature["team"], opponent, venue, match_date, score))

            try:
                predicted_xi = select_best_xi(candidates, team1, team2)
                oracle_candidates = [{**candidate, "predicted_points": candidate["actual_points"]} for candidate in candidates]
                oracle_xi = select_best_xi(oracle_candidates, team1, team2)
                predicted_actual = sum(item["actual_points"] for item in predicted_xi)
                oracle_actual = sum(item["actual_points"] for item in oracle_xi)
                xi_results.append({
                    "match_id": match.get("_match_id", "unknown"),
                    "match_date": match_date.isoformat(),
                    "predicted_xi_actual_points": predicted_actual,
                    "oracle_xi_actual_points": oracle_actual,
                    "regret": oracle_actual - predicted_actual,
                    "capture_pct": 100 * predicted_actual / oracle_actual if oracle_actual else 0.0,
                })
            except ValueError:
                pass
            for team in teams:
                lineup = {ids.get(name, store.aliases.get(name, name)) for name in info.get("players", {}).get(team, [])}
                pending_lineups.append((team, lineup))

        # All matches on the same date are predicted before any same-day result is learned.
        for update in pending_updates:
            store.update(*update)
        for team, lineup in pending_lineups:
            store.team_recent[team].append(lineup)

    if not rows:
        raise SystemExit("No 2026 T20 player-match rows found")
    frame = pd.DataFrame(rows).sort_values(["match_date", "match_id", "player_name"])
    xi_frame = pd.DataFrame(xi_results)
    overall = metric_block(frame)
    summary = {
        "model_cutoff": artifact["cutoff"],
        "test_start": str(frame["match_date"].min()),
        "test_end": str(frame["match_date"].max()),
        "matches": int(frame["match_id"].nunique()),
        "players": int(frame["player_id"].nunique()),
        "overall": overall,
        "baselines": {
            "career_average": metric_block(frame, "career_avg_baseline"),
            "recent_5_average": metric_block(frame, "recent_5_baseline"),
        },
        "by_month": group_metrics(frame, "month"),
        "by_gender": group_metrics(frame, "gender"),
        "by_experience": group_metrics(frame, "experience_bucket"),
        "xi_selection": {
            "eligible_matches": int(len(xi_frame)),
            "mean_actual_points": float(xi_frame["predicted_xi_actual_points"].mean()) if len(xi_frame) else None,
            "mean_oracle_points": float(xi_frame["oracle_xi_actual_points"].mean()) if len(xi_frame) else None,
            "mean_regret": float(xi_frame["regret"].mean()) if len(xi_frame) else None,
            "median_regret": float(xi_frame["regret"].median()) if len(xi_frame) else None,
            "mean_capture_pct": float(xi_frame["capture_pct"].mean()) if len(xi_frame) else None,
        },
        "largest_errors": frame.nlargest(15, "absolute_error")[
            ["match_date", "match_id", "player_name", "team", "predicted_points", "actual_points", "error"]
        ].to_dict(orient="records"),
    }
    args.rows_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.rows_path, index=False)
    if len(xi_frame):
        xi_frame.to_csv(args.rows_path.with_name("backtest_2026_xi.csv"), index=False)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
