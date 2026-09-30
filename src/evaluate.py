from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from fantasy_cricket.features import normalize_role
from fantasy_cricket.io import load_fixtures
from fantasy_cricket.optimizer import select_best_xi, validate_xi


def _field(obj: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in obj:
            return obj[name]
    return default


def _parse_squad(fixture: dict[str, Any], team: str, index: int) -> list[Any]:
    squads = _field(fixture, "squads", "full_squads", default={})
    if isinstance(squads, dict) and team in squads:
        return squads[team]
    if isinstance(squads, list) and len(squads) >= index:
        return squads[index - 1]
    return _field(
        fixture, f"team{index}_squad", f"team_{index}_squad",
        f"squad{index}", f"squad_{index}", default=[]
    )


def candidates_for_fixture(fixture: dict[str, Any], artifact: dict[str, Any]) -> tuple[str, str, str, list[dict[str, Any]]]:
    team1 = _field(fixture, "team1", "team_1", "Team 1")
    team2 = _field(fixture, "team2", "team_2", "Team 2")
    if not team1 or not team2:
        teams = fixture.get("teams", [])
        if len(teams) == 2:
            team1, team2 = teams
    if not team1 or not team2:
        raise ValueError("Each fixture must identify two teams")
    raw_date = _field(fixture, "match_date", "date", "Match Date")
    match_date = date.fromisoformat(str(raw_date))
    venue = _field(fixture, "venue", "ground", default="unknown")
    match_type = _field(fixture, "match_type", "format", default="T20")
    gender = fixture.get("gender", "unknown")
    store = artifact["feature_store"]
    rows: list[dict[str, Any]] = []
    for idx, team in enumerate((team1, team2), start=1):
        opponent = team2 if team == team1 else team1
        for raw in _parse_squad(fixture, team, idx):
            player = {"name": raw} if isinstance(raw, str) else raw
            name = _field(player, "name", "player_name", "player")
            supplied_id = _field(player, "player_id", "id", "cricsheet_id")
            pid = str(supplied_id or store.aliases.get(name, name))
            role = normalize_role(_field(player, "role", "player_role"))
            row = store.feature_row(pid, team, opponent, venue, match_date, role=role, name=name,
                                    match_type=match_type, gender=gender)
            rows.append(row)
    if len(rows) < 11:
        raise ValueError(f"Fixture {team1} vs {team2} has only {len(rows)} squad members")
    frame = pd.DataFrame(rows)
    predictions = artifact["model"].predict(frame[artifact["features"]])
    if "selection_model" in artifact:
        appearance = artifact["selection_model"].predict_proba(frame[artifact["features"]])[:, 1]
    else:
        appearance = [0.35 + 0.65 * float(row["appearance_rate_10"]) for row in rows]
    candidates = []
    for row, conditional, play_probability in zip(rows, predictions, appearance):
        # Cricsheet contains playing participants, so model output is conditional on appearance.
        # A recent team-selection rate supplies the pre-toss availability discount.
        predicted = max(0.0, float(conditional)) * float(play_probability)
        candidates.append({
            "player_name": row["player_name"], "player_id": row["player_id"],
            "team": row["team"], "role": row["role"], "predicted_points": predicted,
            "conditional_points": max(0.0, float(conditional)),
            "appearance_probability": float(play_probability),
        })
    return match_date.isoformat(), team1, team2, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate constrained fantasy-XI predictions")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = joblib.load(args.model_path)
    fixtures = load_fixtures(args.test_file)
    output_rows = []
    for fixture in fixtures:
        match_date, team1, team2, candidates = candidates_for_fixture(fixture, artifact)
        xi = select_best_xi(candidates, team1, team2)
        validate_xi(xi, team1, team2)
        output_rows.append({
            "Match Date": match_date,
            "Team 1": team1,
            "Team 2": team2,
            "Predicted Best 11 Players": json.dumps([
                {"player": p["player_name"], "role": p["role"], "team": p["team"]} for p in xi
            ], ensure_ascii=False),
            "Predicted Points": json.dumps([round(p["predicted_points"], 3) for p in xi]),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} fixture predictions to {args.output}")


if __name__ == "__main__":
    main()
