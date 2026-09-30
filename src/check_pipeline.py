from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate pipeline parameters and fixture schema")
    parser.add_argument("--train-start", type=date.fromisoformat, required=True)
    parser.add_argument("--train-end", type=date.fromisoformat, required=True)
    parser.add_argument("--cutoff", type=date.fromisoformat, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    args = parser.parse_args()
    if not args.train_start <= args.train_end <= args.cutoff:
        raise SystemExit("Expected train-start <= train-end <= cutoff")
    if not args.test_file.exists():
        raise SystemExit(f"Missing test fixture: {args.test_file}")
    json.loads(args.test_file.read_text(encoding="utf-8"))
    print("Parameter and fixture checks passed")


if __name__ == "__main__":
    main()

