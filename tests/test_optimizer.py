from fantasy_cricket.optimizer import select_best_xi, validate_xi


def test_optimizer_enforces_composition_and_both_teams():
    roles = (["Wicket-Keeper"] * 2 + ["Batsman"] * 7 + ["All-Rounder"] * 4 + ["Bowler"] * 7)
    candidates = [
        {"player_name": f"P{i}", "role": role, "team": "A" if i < 15 else "B", "predicted_points": 100 - i}
        for i, role in enumerate(roles)
    ]
    xi = select_best_xi(candidates, "A", "B")
    validate_xi(xi, "A", "B")
    assert len(xi) == 11
    assert {p["team"] for p in xi} == {"A", "B"}

