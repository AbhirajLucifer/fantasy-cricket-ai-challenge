from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any


def load_matches(raw_dir: Path, match_types: set[str] | None = None) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for path in sorted(raw_dir.rglob("*.json")):
        if path.name in {"test_fixtures.json", "actuals.json"}:
            continue
        try:
            match = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not (isinstance(match, dict) and "info" in match and "innings" in match):
            continue
        if match_types and match["info"].get("match_type") not in match_types:
            continue
        if isinstance(match, dict):
            match["_match_id"] = path.stem
            matches.append(match)
    return matches


def safe_extract_zip(path: Path, output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            target = (output_dir / member.filename).resolve()
            if output_dir not in target.parents and target != output_dir:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        archive.extractall(output_dir)


def load_fixtures(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("fixtures", "matches", "data"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    if not isinstance(data, list):
        raise ValueError("Fixture file must contain a list or an object with fixtures/matches/data")
    return data
