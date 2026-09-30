from fantasy_cricket.scoring import score_match


def test_scoring_excludes_super_over_and_applies_core_rules():
    match = {
        "info": {"balls_per_over": 6, "players": {"A": ["Bat", "Mate"], "B": ["Bowl", "Keep"]}},
        "innings": [{"team": "A", "overs": [{"over": 0, "deliveries": [
            {"batter": "Bat", "non_striker": "Mate", "bowler": "Bowl", "runs": {"batter": 4, "extras": 0, "total": 4}},
            {"batter": "Bat", "non_striker": "Mate", "bowler": "Bowl", "runs": {"batter": 0, "extras": 0, "total": 0},
             "wickets": [{"player_out": "Bat", "kind": "caught", "fielders": [{"name": "Keep"}]}]},
        ]}]}, {"team": "A", "super_over": True, "overs": [{"over": 0, "deliveries": [
            {"batter": "Bat", "non_striker": "Mate", "bowler": "Bowl", "runs": {"batter": 6, "extras": 0, "total": 6}}
        ]}]}],
    }
    scored = score_match(match)
    assert scored["Bat"]["batting"] == 5  # four runs + one boundary bonus, no SR bonus
    assert scored["Bowl"]["bowling"] == 26  # dot + wicket
    assert scored["Keep"]["fielding"] == 6
