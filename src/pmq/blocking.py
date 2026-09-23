"""Candidate pair generation (blocking).

Comparing every Abt product with every Buy product gives ~1.2M pairs for this dataset and
grows quadratically. Blocking keeps only plausible pairs from two cheap sources:

1. shared model code in name or description (exact key block),
2. top-k nearest neighbours by TF-IDF cosine similarity on name + description.

The union is labelled with the gold mapping. `evaluate_blocking` reports how many true pairs
survive (recall) and how much work is saved (reduction ratio) - both must be checked before
any model is trained, otherwise the model is evaluated on a silently truncated problem.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from pmq.data import extract_model_codes, normalize_text


def _text(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["name"].map(normalize_text) + " " + frame["description"].map(normalize_text)
    ).str.strip()


def _code_block(abt: pd.DataFrame, buy: pd.DataFrame) -> set[tuple[int, int]]:
    def codes(frame: pd.DataFrame) -> dict[str, list[int]]:
        index: dict[str, list[int]] = {}
        for pid, name, desc in zip(frame["id"], frame["name"], frame["description"], strict=True):
            for code in extract_model_codes(name) | extract_model_codes(desc):
                index.setdefault(code, []).append(int(pid))
        return index

    abt_idx, buy_idx = codes(abt), codes(buy)
    pairs: set[tuple[int, int]] = set()
    for code, abt_ids in abt_idx.items():
        for a in abt_ids:
            for b in buy_idx.get(code, []):
                pairs.add((a, b))
    return pairs


def _tfidf_block(abt: pd.DataFrame, buy: pd.DataFrame, top_k: int) -> set[tuple[int, int]]:
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    corpus = pd.concat([_text(abt), _text(buy)], ignore_index=True)
    matrix = vec.fit_transform(corpus)
    a_mat, b_mat = matrix[: len(abt)], matrix[len(abt) :]
    sims = (a_mat @ b_mat.T).toarray()
    k = min(top_k, len(buy))
    pairs: set[tuple[int, int]] = set()
    abt_ids, buy_ids = abt["id"].to_numpy(), buy["id"].to_numpy()
    top_for_abt = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    for i, cols in enumerate(top_for_abt):
        for j in cols:
            pairs.add((int(abt_ids[i]), int(buy_ids[j])))
    top_for_buy = np.argpartition(-sims, k - 1, axis=0)[:k, :]
    for j in range(sims.shape[1]):
        for i in top_for_buy[:, j]:
            pairs.add((int(abt_ids[i]), int(buy_ids[j])))
    return pairs


def build_candidates(
    abt: pd.DataFrame, buy: pd.DataFrame, gold: pd.DataFrame, top_k: int = 10
) -> pd.DataFrame:
    """Return candidate pairs with columns idAbt, idBuy, label (1 = true match)."""
    pairs = _code_block(abt, buy) | _tfidf_block(abt, buy, top_k)
    cands = pd.DataFrame(sorted(pairs), columns=["idAbt", "idBuy"])
    gold_set = set(zip(gold["idAbt"].astype(int), gold["idBuy"].astype(int), strict=True))
    cands["label"] = [
        int((a, b) in gold_set) for a, b in zip(cands["idAbt"], cands["idBuy"], strict=True)
    ]
    return cands


def evaluate_blocking(
    cands: pd.DataFrame, gold: pd.DataFrame, n_abt: int, n_buy: int
) -> dict[str, float | int]:
    found = int(cands["label"].sum())
    total = n_abt * n_buy
    return {
        "gold_pairs": int(len(gold)),
        "gold_pairs_in_candidates": found,
        "recall_pct": round(100.0 * found / len(gold), 1),
        "candidates": int(len(cands)),
        "all_pairs": total,
        "reduction_ratio_pct": round(100.0 * (1 - len(cands) / total), 2),
        "positive_rate_pct": round(100.0 * found / len(cands), 2),
    }
