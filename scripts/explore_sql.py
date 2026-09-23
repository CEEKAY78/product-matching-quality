"""Run the exploration queries in sql/ against the cleaned catalogs with DuckDB.

Writes cleaned parquet files to data/processed/ and a markdown report to
reports/01_data_quality.md (tables from SQL + python-side quality checks).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pmq.data import load_catalogs  # noqa: E402
from pmq.validate import mapping_quality, profile_catalog  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW, PROCESSED = ROOT / "data" / "raw", ROOT / "data" / "processed"
SQL_DIR, REPORTS = ROOT / "sql", ROOT / "reports"


def to_markdown(frame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def dict_table(d: dict) -> str:
    return "| metric | value |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in d.items())


def main() -> None:
    abt, buy, mapping = load_catalogs(RAW)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    abt.to_parquet(PROCESSED / "abt.parquet", index=False)
    buy.to_parquet(PROCESSED / "buy.parquet", index=False)
    mapping.to_parquet(PROCESSED / "mapping.parquet", index=False)

    con = duckdb.connect()
    for name in ("abt", "buy", "mapping"):
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM '{PROCESSED / (name + '.parquet')}'")

    parts = ["# Data quality and exploration (Abt-Buy)\n"]
    parts.append("## Catalog profiles (python checks)\n")
    parts.append("### Abt\n" + dict_table(profile_catalog(abt)) + "\n")
    parts.append("### Buy\n" + dict_table(profile_catalog(buy)) + "\n")
    parts.append(
        "## Gold mapping quality\n" + dict_table(mapping_quality(abt, buy, mapping)) + "\n"
    )

    for sql_file in sorted(SQL_DIR.glob("*.sql")):
        sql = sql_file.read_text(encoding="utf-8")
        comment = "\n".join(line[3:] for line in sql.splitlines() if line.startswith("-- "))
        result = con.execute(sql).df()
        parts.append(f"## {sql_file.stem}\n\n{comment}\n\n{to_markdown(result)}\n")
        print(f"== {sql_file.name}\n{result.to_string(index=False)}\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, (label, frame) in zip(axes, [("Abt", abt), ("Buy", buy)], strict=True):
        frame["name"].str.len().plot.hist(bins=30, ax=ax, color="#4c72b0")
        ax.set_title(f"{label}: product name length (chars)")
    fig.tight_layout()
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS / "figures" / "name_length.png", dpi=110)
    parts.append("![name length](figures/name_length.png)\n")

    (REPORTS / "01_data_quality.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"Report written to {REPORTS / '01_data_quality.md'}")


if __name__ == "__main__":
    main()
