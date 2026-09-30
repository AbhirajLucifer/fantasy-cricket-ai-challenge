from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from fantasy_cricket.io import safe_extract_zip


DEFAULT_URL = "https://cricsheet.org/downloads/all_json.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Cricsheet JSON archives")
    parser.add_argument("--url", action="append", default=None, help="Cricsheet zip URL; repeatable")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for url in args.url or [DEFAULT_URL]:
        if not url.startswith("https://cricsheet.org/"):
            raise ValueError("Match archives must come from cricsheet.org")
        name = url.rsplit("/", 1)[-1]
        archive = args.raw_dir / name
        urllib.request.urlretrieve(url, archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        extract_dir = args.raw_dir / archive.stem
        extract_dir.mkdir(exist_ok=True)
        safe_extract_zip(archive, extract_dir)
        records.append({"url": url, "sha256": digest, "archive": str(archive), "extract_dir": str(extract_dir)})
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

