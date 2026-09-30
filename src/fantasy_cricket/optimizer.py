from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ROLES = ("Wicket-Keeper", "Batsman", "All-Rounder", "Bowler")
ROLE_LIMITS = {
    "Wicket-Keeper": (1, 4), "Batsman": (3, 6),
    "All-Rounder": (1, 4), "Bowler": (3, 6),
}


@dataclass
class Node:
    score: float
    selected: tuple[int, ...]


def select_best_xi(candidates: list[dict[str, Any]], team1: str, team2: str) -> list[dict[str, Any]]:
    """Exact dynamic program over count, role counts and team-presence mask."""
    if len(candidates) < 11:
        raise ValueError("At least 11 candidates are required")
    state: dict[tuple[int, int, int, int, int, int], Node] = {(0, 0, 0, 0, 0, 0): Node(0.0, ())}
    for idx, player in enumerate(candidates):
        role = player["role"]
        if role not in ROLES:
            raise ValueError(f"Unknown role {role!r} for {player.get('player_name')}")
        role_index = ROLES.index(role)
        bit = 1 if player["team"] == team1 else 2 if player["team"] == team2 else 0
        updated = dict(state)
        for key, node in state.items():
            count, wk, bat, ar, bowl, mask = key
            counts = [wk, bat, ar, bowl]
            if count == 11 or counts[role_index] >= ROLE_LIMITS[role][1]:
                continue
            counts[role_index] += 1
            new_key = (count + 1, *counts, mask | bit)
            new_node = Node(node.score + float(player["predicted_points"]), node.selected + (idx,))
            if new_key not in updated or new_node.score > updated[new_key].score:
                updated[new_key] = new_node
        state = updated
    feasible: list[Node] = []
    for key, node in state.items():
        count, wk, bat, ar, bowl, mask = key
        counts = dict(zip(ROLES, (wk, bat, ar, bowl)))
        if count == 11 and mask == 3 and all(ROLE_LIMITS[r][0] <= counts[r] <= ROLE_LIMITS[r][1] for r in ROLES):
            feasible.append(node)
    if not feasible:
        available = {role: sum(p["role"] == role for p in candidates) for role in ROLES}
        raise ValueError(f"No valid XI can be formed; role availability={available}")
    best = max(feasible, key=lambda n: n.score)
    selected = [candidates[i] for i in best.selected]
    return sorted(selected, key=lambda p: (-float(p["predicted_points"]), p["player_name"]))


def validate_xi(players: list[dict[str, Any]], team1: str, team2: str) -> None:
    if len(players) != 11:
        raise AssertionError("XI must contain exactly 11 players")
    counts = {role: sum(p["role"] == role for p in players) for role in ROLES}
    for role, (low, high) in ROLE_LIMITS.items():
        if not low <= counts[role] <= high:
            raise AssertionError(f"Invalid {role} count: {counts[role]}")
    teams = {p["team"] for p in players}
    if team1 not in teams or team2 not in teams:
        raise AssertionError("XI must include at least one player from each team")

