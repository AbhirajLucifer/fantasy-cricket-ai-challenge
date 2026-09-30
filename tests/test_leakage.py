from datetime import date

import pytest

from fantasy_cricket.features import build_training_rows, normalize_role


def test_training_end_after_cutoff_is_rejected():
    with pytest.raises(ValueError):
        build_training_rows([], date(2020, 1, 1), date(2026, 1, 1), date(2025, 12, 31))


def test_common_fixture_roles_are_normalized():
    assert normalize_role("WK-Batter") == "Wicket-Keeper"
    assert normalize_role("Batting Allrounder") == "All-Rounder"
    assert normalize_role("Right arm Bowler") == "Bowler"
