"""Pairwise features for candidate (Abt, Buy) pairs.

Every feature is a plain number computed from the two records only, so the same code runs
for training pairs and for scoring new mappings in production. Missing prices are encoded
explicitly (price_both_present flag + sentinel -1) instead of being imputed, because
"price unknown" carries information of its own in this data (61% missing on Abt).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

from pmq.data import extract_model_codes, normalize_text

FEATURE_COLUMNS = [
    "name_char_tfidf_cos",
    "name_word_tfidf_cos",
    "text_word_tfidf_cos",
    "name_token_jaccard",
    "name_token_set_ratio",
    "name_partial_ratio",
    "shared_model_codes",
    "n_shared_codes",
    "code_prefix_match",
    "abt_code_in_buy_text",
    "buy_code_in_abt_text",
    "brand_match",
    "brand_in_other_name",
    "price_both_present",
    "price_rel_gap",
    "name_len_ratio",
    "buy_has_description",
]


def _cosine_rows(vec: TfidfVectorizer, left: pd.Series, right: pd.Series) -> np.ndarray:
    matrix = vec.fit_transform(pd.concat([left, right], ignore_index=True))
    a, b = matrix[: len(left)], matrix[len(left) :]
    return np.asarray(a.multiply(b).sum(axis=1)).ravel()


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def _prefix_match(a: set[str], b: set[str], min_len: int = 5) -> bool:
    """True if a code on one side is a prefix of a code on the other side (CLI221 ~ CLI221BLK)."""
    return any(
        len(x) >= min_len and len(y) >= min_len and x != y and (x.startswith(y) or y.startswith(x))
        for x in a
        for y in b
    )


def build_features(cands: pd.DataFrame, abt: pd.DataFrame, buy: pd.DataFrame) -> pd.DataFrame:
    """Return cands with FEATURE_COLUMNS appended (idAbt, idBuy and label preserved)."""
    a = abt.set_index("id").loc[cands["idAbt"]].reset_index(drop=True)
    b = buy.set_index("id").loc[cands["idBuy"]].reset_index(drop=True)

    a_name, b_name = a["name"].map(normalize_text), b["name"].map(normalize_text)
    a_text = (a_name + " " + a["description"].map(normalize_text)).str.strip()
    b_text = (b_name + " " + b["description"].map(normalize_text)).str.strip()
    a_codes = [
        extract_model_codes(n) | extract_model_codes(d)
        for n, d in zip(a["name"], a["description"], strict=True)
    ]
    b_codes = [
        extract_model_codes(n) | extract_model_codes(d)
        for n, d in zip(b["name"], b["description"], strict=True)
    ]

    out = cands.reset_index(drop=True).copy()
    out["name_char_tfidf_cos"] = _cosine_rows(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True), a_name, b_name
    )
    out["name_word_tfidf_cos"] = _cosine_rows(TfidfVectorizer(sublinear_tf=True), a_name, b_name)
    out["text_word_tfidf_cos"] = _cosine_rows(TfidfVectorizer(sublinear_tf=True), a_text, b_text)
    out["name_token_jaccard"] = [
        _jaccard(set(x.split()), set(y.split())) for x, y in zip(a_name, b_name, strict=True)
    ]
    out["name_token_set_ratio"] = [
        fuzz.token_set_ratio(x, y) / 100 for x, y in zip(a_name, b_name, strict=True)
    ]
    out["name_partial_ratio"] = [
        fuzz.partial_ratio(x, y) / 100 for x, y in zip(a_name, b_name, strict=True)
    ]
    shared = [x & y for x, y in zip(a_codes, b_codes, strict=True)]
    out["n_shared_codes"] = [len(s) for s in shared]
    out["shared_model_codes"] = (out["n_shared_codes"] > 0).astype(int)
    out["code_prefix_match"] = [
        int(_prefix_match(x, y)) for x, y in zip(a_codes, b_codes, strict=True)
    ]
    out["abt_code_in_buy_text"] = [
        int(any(c in t for c in codes)) for codes, t in zip(a_codes, b_text, strict=True)
    ]
    out["buy_code_in_abt_text"] = [
        int(any(c in t for c in codes)) for codes, t in zip(b_codes, a_text, strict=True)
    ]
    out["brand_match"] = (
        ((a["manufacturer"] == b["manufacturer"]) & (a["manufacturer"] != ""))
        .astype(int)
        .to_numpy()
    )
    out["brand_in_other_name"] = [
        int((ma != "" and ma in nb) or (mb != "" and mb in na))
        for ma, mb, na, nb in zip(a["manufacturer"], b["manufacturer"], a_name, b_name, strict=True)
    ]
    both = a["price"].notna() & b["price"].notna()
    gap = (a["price"] - b["price"]).abs() / np.maximum(a["price"], b["price"])
    out["price_both_present"] = both.astype(int).to_numpy()
    out["price_rel_gap"] = np.where(both, gap.fillna(0.0), -1.0)
    out["name_len_ratio"] = [
        min(len(x), len(y)) / max(len(x), len(y), 1) for x, y in zip(a_name, b_name, strict=True)
    ]
    out["buy_has_description"] = (b["description"].map(normalize_text) != "").astype(int).to_numpy()
    return out
