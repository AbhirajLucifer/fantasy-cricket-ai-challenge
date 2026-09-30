from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date
from statistics import mean
from typing import Any, Iterable

import pandas as pd

from .scoring import score_match


ROLE_MAP = {
    "wk": "Wicket-Keeper", "wicketkeeper": "Wicket-Keeper", "wicket-keeper": "Wicket-Keeper",
    "bat": "Batsman", "batter": "Batsman", "batsman": "Batsman",
    "ar": "All-Rounder", "allrounder": "All-Rounder", "all-rounder": "All-Rounder",
    "bowl": "Bowler", "bowler": "Bowler",
}


def _int_dict() -> defaultdict[str, int]:
    return defaultdict(int)


def _list_dict() -> defaultdict[str, list[float]]:
    return defaultdict(list)


def _lineup_history_dict() -> defaultdict[str, deque[set[str]]]:
    return defaultdict(_new_lineup_history)


def _new_lineup_history() -> deque[set[str]]:
    return deque(maxlen=10)


def normalize_role(value: str | None) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    key = raw.lower().replace("_", " ")
    if key in ROLE_MAP:
        return ROLE_MAP[key]
    compact = key.replace("-", " ")
    if "wicket" in compact or compact.startswith("wk"):
        return "Wicket-Keeper"
    if "all" in compact and "round" in compact:
        return "All-Rounder"
    if "bowl" in compact:
        return "Bowler"
    if "bat" in compact:
        return "Batsman"
    return raw


@dataclass
class PlayerState:
    matches: int = 0
    total: float = 0.0
    batting: float = 0.0
    bowling: float = 0.0
    fielding: float = 0.0
    recent: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recent_batting: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recent_bowling: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recent_fielding: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    match_dates: deque[date] = field(default_factory=lambda: deque(maxlen=100))
    last_date: date | None = None
    teams: dict[str, int] = field(default_factory=_int_dict)
    venues: dict[str, list[float]] = field(default_factory=_list_dict)
    opponents: dict[str, list[float]] = field(default_factory=_list_dict)


def infer_role(state: PlayerState | None) -> str:
    if not state or state.matches == 0:
        return "Batsman"
    bat = state.batting / state.matches
    bowl = state.bowling / state.matches
    field = state.fielding / state.matches
    if field >= 5 and bowl < 8:
        return "Wicket-Keeper"
    if bat >= 8 and bowl >= 8:
        return "All-Rounder"
    if bowl > bat * 1.15 and bowl >= 6:
        return "Bowler"
    return "Batsman"


def _avg(values: Iterable[float], default: float = 0.0) -> float:
    values = list(values)
    return float(mean(values)) if values else default


class FeatureStore:
    def __init__(self) -> None:
        self.players: dict[str, PlayerState] = {}
        self.team_recent: dict[str, deque[set[str]]] = _lineup_history_dict()
        self.aliases: dict[str, str] = {}

    def selection_rate(self, player_id: str, team: str) -> float:
        history = self.team_recent.get(team, [])
        if not history:
            return 0.5
        return sum(player_id in lineup for lineup in history) / len(history)

    def feature_row(self, player_id: str, team: str, opponent: str, venue: str,
                    match_date: date, role: str | None = None, name: str | None = None,
                    match_type: str = "T20", gender: str = "unknown") -> dict[str, Any]:
        state = self.players.get(player_id)
        role = normalize_role(role) or infer_role(state)
        matches = state.matches if state else 0
        recent = list(state.recent) if state else []
        recent_batting = list(state.recent_batting) if state else []
        recent_bowling = list(state.recent_bowling) if state else []
        recent_fielding = list(state.recent_fielding) if state else []
        rest_days = (match_date - state.last_date).days if state and state.last_date else 365
        return {
            "player_id": player_id,
            "player_name": name or player_id,
            "team": team,
            "opponent": opponent,
            "venue": venue or "unknown",
            "match_type": match_type,
            "gender": gender,
            "role": role,
            "experience": matches,
            "career_avg": state.total / matches if matches else 0.0,
            "batting_avg": state.batting / matches if matches else 0.0,
            "bowling_avg": state.bowling / matches if matches else 0.0,
            "fielding_avg": state.fielding / matches if matches else 0.0,
            "recent_3": _avg(recent[-3:]),
            "recent_5": _avg(recent[-5:]),
            "recent_10": _avg(recent),
            "recent_std": float(pd.Series(recent).std(ddof=0)) if recent else 0.0,
            "recent_batting_5": _avg(recent_batting[-5:]),
            "recent_bowling_5": _avg(recent_bowling[-5:]),
            "recent_fielding_5": _avg(recent_fielding[-5:]),
            "recent_peak_10": max(recent) if recent else 0.0,
            "recent_floor_10": min(recent) if recent else 0.0,
            "form_trend": _avg(recent[-3:]) - _avg(recent[-6:-3]),
            "venue_avg": _avg(state.venues.get(venue, [])[-5:]) if state else 0.0,
            "opponent_avg": _avg(state.opponents.get(opponent, [])[-5:]) if state else 0.0,
            "team_matches": state.teams.get(team, 0) if state else 0,
            "team_share": state.teams.get(team, 0) / matches if state and matches else 0.0,
            "matches_last_30": sum((match_date - played).days <= 30 for played in getattr(state, "match_dates", ())) if state else 0,
            "matches_last_90": sum((match_date - played).days <= 90 for played in getattr(state, "match_dates", ())) if state else 0,
            "appearance_rate_10": self.selection_rate(player_id, team),
            "rest_days": min(max(rest_days, 0), 365),
            "month": match_date.month,
        }

    def update(self, player_id: str, team: str, opponent: str, venue: str,
               match_date: date, score: dict[str, float]) -> None:
        state = self.players.setdefault(player_id, PlayerState())
        state.matches += 1
        state.total += score["fantasy_points"]
        state.batting += score["batting"]
        state.bowling += score["bowling"]
        state.fielding += score["fielding"]
        state.recent.append(score["fantasy_points"])
        state.recent_batting.append(score["batting"])
        state.recent_bowling.append(score["bowling"])
        state.recent_fielding.append(score["fielding"])
        if not hasattr(state, "match_dates"):
            state.match_dates = deque(maxlen=100)
        state.match_dates.append(match_date)
        state.last_date = match_date
        state.teams[team] += 1
        state.venues[venue].append(score["fantasy_points"])
        state.opponents[opponent].append(score["fantasy_points"])


def match_date_of(match: dict[str, Any]) -> date:
    return date.fromisoformat(str(match["info"]["dates"][0]))


def player_id_map(match: dict[str, Any]) -> dict[str, str]:
    registry = match["info"].get("registry", {}).get("people", {})
    return {name: registry.get(name, name) for name in registry}


def build_training_datasets(matches: list[dict[str, Any]], train_start: date, train_end: date,
                            cutoff: date, match_types: set[str] | None = None
                            ) -> tuple[pd.DataFrame, pd.DataFrame, FeatureStore]:
    if train_end > cutoff:
        raise ValueError(f"Training end {train_end} exceeds cutoff {cutoff}")
    store = FeatureStore()
    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    id_to_name: dict[str, str] = {}
    by_date: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for match in matches:
        mdate = match_date_of(match)
        if mdate <= cutoff and (not match_types or match["info"].get("match_type") in match_types):
            by_date[mdate].append(match)

    for mdate in sorted(by_date):
        pending_updates: list[tuple[str, str, str, str, date, dict[str, float]]] = []
        pending_lineups: list[tuple[str, set[str]]] = []
        for match in sorted(by_date[mdate], key=lambda item: item.get("_match_id", "")):
            info = match["info"]
            teams = info.get("teams", [])
            if len(teams) != 2:
                continue
            team1, team2 = teams
            venue = info.get("venue", "unknown")
            ids = info.get("registry", {}).get("people", {})
            for name, pid in ids.items():
                store.aliases[name] = pid
                id_to_name[pid] = name
            scores = score_match(match)
            player_teams = {name: team for team, names in info.get("players", {}).items() for name in names}

            current_lineups = {
                team: {ids.get(name, name) for name in info.get("players", {}).get(team, [])}
                for team in teams
            }
            if train_start <= mdate <= train_end:
                for team in teams:
                    opponent = team2 if team == team1 else team1
                    historical = set().union(*store.team_recent.get(team, [])) if store.team_recent.get(team) else set()
                    for pid in historical | current_lineups[team]:
                        feature = store.feature_row(
                            pid, team, opponent, venue, mdate, name=id_to_name.get(pid, pid),
                            match_type=info.get("match_type", "unknown"), gender=info.get("gender", "unknown"),
                        )
                        feature.update({
                            "match_date": mdate.isoformat(), "match_id": match.get("_match_id", "unknown"),
                            "selected": int(pid in current_lineups[team]),
                        })
                        selection_rows.append(feature)

            for name, score in scores.items():
                team = player_teams.get(name)
                if team not in teams:
                    continue
                opponent = team2 if team == team1 else team1
                pid = ids.get(name, name)
                store.aliases[name] = pid
                id_to_name[pid] = name
                feature = store.feature_row(
                    pid, team, opponent, venue, mdate, name=name,
                    match_type=info.get("match_type", "unknown"), gender=info.get("gender", "unknown"),
                )
                feature.update({
                    "match_date": mdate.isoformat(), "match_id": match.get("_match_id", "unknown"),
                    "fantasy_points": score["fantasy_points"], "actual_batting": score["batting"],
                    "actual_bowling": score["bowling"], "actual_fielding": score["fielding"],
                })
                if train_start <= mdate <= train_end:
                    rows.append(feature)
                pending_updates.append((pid, team, opponent, venue, mdate, score))
            pending_lineups.extend(current_lineups.items())

        for update in pending_updates:
            store.update(*update)
        for team, lineup in pending_lineups:
            store.team_recent[team].append(lineup)
    return pd.DataFrame(rows), pd.DataFrame(selection_rows), store


def build_training_rows(matches: list[dict[str, Any]], train_start: date, train_end: date,
                        cutoff: date, match_types: set[str] | None = None) -> tuple[pd.DataFrame, FeatureStore]:
    rows, _, store = build_training_datasets(matches, train_start, train_end, cutoff, match_types)
    return rows, store
