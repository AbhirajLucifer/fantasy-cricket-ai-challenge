from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score

from fantasy_cricket.features import match_date_of
from fantasy_cricket.io import load_matches
from fantasy_cricket.optimizer import select_best_xi
from fantasy_cricket.scoring import score_match


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-toss pseudo-squad backtest on 2026 T20 matches")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifact = joblib.load(args.model_path)
    store = copy.deepcopy(artifact["feature_store"])
    matches = [m for m in load_matches(args.raw_dir, {"T20"}) if match_date_of(m).year == 2026]
    by_date: dict[date, list[dict]] = defaultdict(list)
    for match in matches:
        by_date[match_date_of(match)].append(match)

    candidate_rows: list[dict] = []
    xi_rows: list[dict] = []
    id_to_name = {pid: name for name, pid in store.aliases.items()}
    for match_date in sorted(by_date):
        pending_updates = []
        pending_lineups = []
        for match in sorted(by_date[match_date], key=lambda item: item.get("_match_id", "")):
            info = match["info"]
            teams = info.get("teams", [])
            if len(teams) != 2:
                continue
            team1, team2 = teams
            venue = info.get("venue", "unknown")
            gender = info.get("gender", "unknown")
            ids = info.get("registry", {}).get("people", {})
            for name, pid in ids.items():
                store.aliases[name] = pid
                id_to_name[pid] = name
            actual_scores = score_match(match)
            current = {
                team: {ids.get(name, name) for name in info.get("players", {}).get(team, [])}
                for team in teams
            }
            actual_by_id = {ids.get(name, name): score for name, score in actual_scores.items()}
            candidates = []
            candidate_features = []
            for team in teams:
                opponent = team2 if team == team1 else team1
                history = list(store.team_recent.get(team, []))[-3:]
                historical = set().union(*history) if history else set()
                # Current XI plus the three previous XIs approximates the announced 15+ squad.
                for pid in historical | current[team]:
                    feature = store.feature_row(
                        pid, team, opponent, venue, match_date, name=id_to_name.get(pid, pid),
                        match_type="T20", gender=gender,
                    )
                    candidate_features.append(feature)
            if not candidate_features:
                continue
            feature_frame = pd.DataFrame(candidate_features)
            conditional = np.maximum(0.0, artifact["model"].predict(feature_frame[artifact["features"]]))
            if "selection_model" in artifact:
                play_probability = artifact["selection_model"].predict_proba(
                    feature_frame[artifact["features"]]
                )[:, 1]
            else:
                play_probability = np.array([
                    .35 + .65 * float(feature["appearance_rate_10"]) for feature in candidate_features
                ])
            for feature, conditional_points, probability in zip(candidate_features, conditional, play_probability):
                pid = feature["player_id"]
                actual = float(actual_by_id.get(pid, {}).get("fantasy_points", 0.0))
                candidate = {
                    "match_id": match.get("_match_id", "unknown"),
                    "match_date": match_date.isoformat(),
                    "player_id": pid,
                    "player_name": feature["player_name"],
                    "team": feature["team"],
                    "role": feature["role"],
                    "plays": int(pid in current[feature["team"]]),
                    "appearance_probability": float(probability),
                    "conditional_points": float(conditional_points),
                    "predicted_points": float(conditional_points * probability),
                    "actual_points": actual,
                }
                candidates.append(candidate)
                candidate_rows.append(candidate)
            try:
                predicted_xi = select_best_xi(candidates, team1, team2)
                oracle = select_best_xi(
                    [{**candidate, "predicted_points": candidate["actual_points"]} for candidate in candidates],
                    team1, team2,
                )
                predicted_actual = sum(player["actual_points"] for player in predicted_xi)
                oracle_actual = sum(player["actual_points"] for player in oracle)
                xi_rows.append({
                    "match_id": match.get("_match_id", "unknown"),
                    "selected_actual_points": predicted_actual,
                    "oracle_actual_points": oracle_actual,
                    "capture_pct": 100 * predicted_actual / oracle_actual if oracle_actual else 0.0,
                    "selected_players_who_played": sum(player["plays"] for player in predicted_xi),
                })
            except ValueError:
                pass
            for team in teams:
                opponent = team2 if team == team1 else team1
                for pid in current[team]:
                    if pid in actual_by_id:
                        pending_updates.append((pid, team, opponent, venue, match_date, actual_by_id[pid]))
                pending_lineups.append((team, current[team]))
        for update in pending_updates:
            store.update(*update)
        for team, lineup in pending_lineups:
            store.team_recent[team].append(lineup)

    frame = pd.DataFrame(candidate_rows)
    xi = pd.DataFrame(xi_rows)
    result = {
        "matches": int(frame["match_id"].nunique()),
        "candidate_rows": int(len(frame)),
        "candidate_mae": float(mean_absolute_error(frame["actual_points"], frame["predicted_points"])),
        "appearance_auc": float(roc_auc_score(frame["plays"], frame["appearance_probability"])),
        "appearance_brier": float(brier_score_loss(frame["plays"], frame["appearance_probability"])),
        "candidate_positive_rate": float(frame["plays"].mean()),
        "eligible_xi_matches": int(len(xi)),
        "mean_selected_actual_points": float(xi["selected_actual_points"].mean()),
        "mean_oracle_actual_points": float(xi["oracle_actual_points"].mean()),
        "mean_capture_pct": float(xi["capture_pct"].mean()),
        "mean_selected_players_who_played": float(xi["selected_players_who_played"].mean()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
