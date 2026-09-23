"""Download the Abt-Buy product matching benchmark into data/raw/.

Source: Leipzig University database group, benchmark datasets for entity resolution.
https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import requests

URL = "https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
EXPECTED = {"Abt.csv", "Buy.csv", "abt_buy_perfectMapping.csv"}


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if EXPECTED.issubset({p.name for p in RAW_DIR.iterdir()}):
        print(f"Data already present in {RAW_DIR}")
        return 0
    print(f"Downloading {URL} ...")
    response = requests.get(URL, timeout=120)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = set(archive.namelist())
        missing = EXPECTED - names
        if missing:
            print(f"Archive is missing files: {missing}", file=sys.stderr)
            return 1
        archive.extractall(RAW_DIR)
    print(f"Extracted {sorted(EXPECTED)} to {RAW_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
