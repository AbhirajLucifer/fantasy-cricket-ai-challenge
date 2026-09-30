from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any


NON_BOWLER_WICKETS = {
    "run out", "retired hurt", "retired out", "obstructing the field",
    "timed out", "handled the ball",
}


@dataclass
class ScoreLine:
    batting: float = 0.0
    bowling: float = 0.0
    fielding: float = 0.0
    participation: float = 3.0

    @property
    def total(self) -> float:
        return self.batting + self.bowling + self.fielding + self.participation

    def to_dict(self) -> dict[str, float]:
        result = asdict(self)
        result["fantasy_points"] = self.total
        return result


def _milestone(runs: int) -> int:
    if runs >= 100:
        return 20
    if runs >= 75:
        return 15
    if runs >= 50:
        return 10
    if runs >= 25:
        return 5
    return 0


def _strike_rate_points(runs: int, balls: int) -> int:
    if balls < 10:
        return 0
    rate = 100.0 * runs / balls
    if rate > 170:
        return 5
    if rate > 150:
        return 3
    if rate >= 130:
        return 1
    if rate > 70:
        return 0
    if rate >= 60:
        return -1
    if rate >= 50:
        return -3
    return -5


def _economy_points(runs: int, legal_balls: int, balls_per_over: int) -> int:
    if legal_balls < 2 * balls_per_over:
        return 0
    economy = runs / (legal_balls / balls_per_over)
    if economy < 5:
        return 5
    if economy < 6:
        return 3
    if economy <= 7:
        return 1
    if economy <= 9.99:
        return 0
    if economy <= 11:
        return -1
    if economy <= 12:
        return -3
    return -5


def _wicket_bonus(wickets: int) -> int:
    if wickets >= 5:
        return 15
    if wickets == 4:
        return 10
    if wickets == 3:
        return 5
    return 0


def score_match(match: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Score a Cricsheet JSON match, excluding Super Overs."""
    info = match["info"]
    players = {p for squad in info.get("players", {}).values() for p in squad}
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(int))
    dismissed: set[str] = set()
    balls_per_over = int(info.get("balls_per_over", 6))

    for innings in match.get("innings", []):
        if innings.get("super_over"):
            continue
        for over in innings.get("overs", []):
            over_bowler_runs: dict[str, int] = defaultdict(int)
            over_legal: dict[str, int] = defaultdict(int)
            for delivery in over.get("deliveries", []):
                batter = delivery["batter"]
                bowler = delivery["bowler"]
                runs = delivery.get("runs", {})
                extras = delivery.get("extras", {})
                batter_runs = int(runs.get("batter", 0))
                total_runs = int(runs.get("total", 0))

                stats[batter]["runs"] += batter_runs
                if "wides" not in extras:
                    stats[batter]["balls"] += 1
                if batter_runs == 4:
                    stats[batter]["fours"] += 1
                elif batter_runs == 6:
                    stats[batter]["sixes"] += 1

                legal = "wides" not in extras and "noballs" not in extras
                if legal:
                    stats[bowler]["legal_balls"] += 1
                    over_legal[bowler] += 1
                conceded = total_runs - int(extras.get("byes", 0)) - int(extras.get("legbyes", 0)) - int(extras.get("penalty", 0))
                stats[bowler]["runs_conceded"] += conceded
                over_bowler_runs[bowler] += conceded
                if total_runs == 0:
                    stats[bowler]["dots"] += 1

                for wicket in delivery.get("wickets", []):
                    kind = wicket.get("kind", "").lower()
                    out = wicket.get("player_out")
                    if out:
                        dismissed.add(out)
                    if kind not in NON_BOWLER_WICKETS:
                        stats[bowler]["wickets"] += 1
                        if kind in {"bowled", "lbw"}:
                            stats[bowler]["lbw_bowled"] += 1
                    fielders = [f.get("name") for f in wicket.get("fielders", []) if f.get("name")]
                    if kind in {"caught", "caught and bowled"}:
                        if kind == "caught and bowled" and not fielders:
                            fielders = [bowler]
                        for fielder in fielders:
                            stats[fielder]["catches"] += 1
                    elif kind == "stumped":
                        for fielder in fielders:
                            stats[fielder]["stumpings"] += 1
                    elif kind == "run out":
                        key = "direct_runouts" if len(fielders) == 1 else "assisted_runouts"
                        for fielder in fielders:
                            stats[fielder][key] += 1

            for bowler, legal_balls in over_legal.items():
                if legal_balls >= balls_per_over and over_bowler_runs[bowler] == 0:
                    stats[bowler]["maidens"] += 1

    results: dict[str, dict[str, float]] = {}
    for player in players | set(stats):
        s = stats[player]
        batting = (
            s["runs"] + s["fours"] + 2 * s["sixes"] + _milestone(s["runs"])
            + _strike_rate_points(s["runs"], s["balls"])
        )
        if s["runs"] == 0 and player in dismissed:
            batting -= 2
        bowling = (
            s["dots"] + 25 * s["wickets"] + 5 * s["lbw_bowled"]
            + _wicket_bonus(s["wickets"]) + 6 * s["maidens"]
            + _economy_points(s["runs_conceded"], s["legal_balls"], balls_per_over)
        )
        fielding = (
            6 * s["catches"] + (4 if s["catches"] >= 3 else 0)
            + 10 * s["stumpings"] + 10 * s["direct_runouts"]
            + 5 * s["assisted_runouts"]
        )
        results[player] = ScoreLine(batting, bowling, fielding).to_dict()
    return results

