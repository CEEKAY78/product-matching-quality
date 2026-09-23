"""Loading and cleaning the two product catalogs (Abt, Buy) and the gold mapping.

Data quality issues handled here (documented in reports/01_data_quality.md):
- files are latin-1 encoded, not UTF-8;
- prices are strings with "$" and thousands separators, and are mostly missing;
- Abt has no manufacturer column: the brand is recovered from the first word of the name;
- descriptions are missing for 40% of Buy rows.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd

COLUMNS = ["id", "name", "description", "manufacturer", "price"]

_PUNCT = re.compile(r"[^a-z0-9]+")
_MODEL_CODE = re.compile(r"^(?=.*[a-z])(?=.*\d)[a-z0-9]{4,}$")
_HYPHENATED = re.compile(r"\b\w+(?:[-/]\w+)+\b")  # LCJ-THC/W, CLI-221, 010-10723-06


def parse_price(value: object) -> float:
    """Convert "$1,299.99" style strings to float; anything unparsable becomes NaN."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return math.nan
    text = str(value).strip().replace("$", "").replace(",", "")
    try:
        return float(text)
    except ValueError:
        return math.nan


def normalize_text(value: object) -> str:
    """Lowercase, drop punctuation, collapse whitespace. None/NaN -> empty string."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return _PUNCT.sub(" ", str(value).lower()).strip()


def extract_model_codes(name: object) -> set[str]:
    """Tokens that mix letters and digits (e.g. 'pslx350h') are model codes.

    Model codes are the strongest matching signal in retail catalogs: two listings sharing a
    code are almost always the same product.
    """
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return set()
    text = str(name)
    codes = {tok for tok in normalize_text(text).split() if _MODEL_CODE.match(tok)}
    # hyphenated / slashed tokens are codes even without digits: LCJ-THC/W -> lcjthcw
    for raw in _HYPHENATED.findall(text):
        joined = re.sub(r"[-/]", "", raw).lower()
        if len(joined) >= 4 and not joined.isdigit():
            codes.add(joined)
    return codes


def extract_brand(name: object) -> str:
    tokens = normalize_text(name).split()
    return tokens[0] if tokens else ""


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "manufacturer" not in out.columns:
        out["manufacturer"] = out["name"].map(extract_brand)
    else:
        missing = out["manufacturer"].isna()
        out["manufacturer"] = out["manufacturer"].map(normalize_text)
        out.loc[missing, "manufacturer"] = out.loc[missing, "name"].map(extract_brand)
    out["price"] = out["price"].map(parse_price)
    out["id"] = out["id"].astype(int)
    out["name"] = out["name"].fillna("").astype(str)
    out["description"] = out["description"].fillna("").astype(str)
    return out[COLUMNS].reset_index(drop=True)


def load_catalogs(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (abt, buy, mapping). mapping has columns idAbt, idBuy with int dtype."""
    raw_dir = Path(raw_dir)
    abt = pd.read_csv(raw_dir / "Abt.csv", encoding="latin-1")
    buy = pd.read_csv(raw_dir / "Buy.csv", encoding="latin-1")
    mapping = pd.read_csv(raw_dir / "abt_buy_perfectMapping.csv", encoding="latin-1")
    mapping = mapping.astype({"idAbt": int, "idBuy": int})
    return _clean(abt), _clean(buy), mapping
