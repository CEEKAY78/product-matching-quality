"""Send the classifier's uncertain test pairs to a local LLM and measure the hybrid.

Requires Ollama running locally with the model pulled (default qwen2.5:7b-instruct).
Answers are cached in data/processed/llm_cache.json.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from pmq.data import load_catalogs
from pmq.llm_review import OllamaClient, combine_hybrid, load_cache, review_pairs, save_cache

ROOT = Path(__file__).resolve().parents[1]
RAW, PROCESSED, REPORTS = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "reports"
LOW, HIGH = 0.3, 0.7


def prf(y, p) -> dict[str, float]:
    return {
        "precision": round(precision_score(y, p, zero_division=0), 4),
        "recall": round(recall_score(y, p), 4),
        "f1": round(f1_score(y, p), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--classifier", default="random_forest")
    args = parser.parse_args()

    abt, buy, _ = load_catalogs(RAW)
    scores = pd.read_parquet(PROCESSED / "test_scores.parquet")
    scores = scores[scores["model"] == args.classifier].reset_index(drop=True)
    band = scores[(scores["proba"] > LOW) & (scores["proba"] < HIGH)].reset_index(drop=True)
    print(f"{len(band)} uncertain pairs out of {len(scores)} test pairs -> {args.model}")

    cache_path = PROCESSED / f"llm_cache_{args.model.replace(':', '_').replace('/', '_')}.json"
    cache = load_cache(cache_path)
    client = OllamaClient(model=args.model)
    t0 = time.time()

    def progress(n, total):
        if n % 10 == 0 or n == total:
            print(f"  {n}/{total} ({time.time() - t0:.0f}s)")
            save_cache(cache_path, cache)

    verdicts = review_pairs(band, abt, buy, client, cache, on_progress=progress)
    save_cache(cache_path, cache)

    band = band.merge(verdicts, on=["idAbt", "idBuy"])
    valid = band["llm_same"].notna()
    band_clf = prf(band["label"], band["pred"])
    band_llm = prf(band.loc[valid, "label"], band.loc[valid, "llm_same"].astype(bool).astype(int))

    llm_full = scores.merge(verdicts, on=["idAbt", "idBuy"], how="left")["llm_same"]
    hybrid_pred = combine_hybrid(scores, llm_full, LOW, HIGH)
    overall = pd.DataFrame(
        [
            {"setting": "classifier only", **prf(scores["label"], scores["pred"])},
            {"setting": "classifier + LLM in uncertain band", **prf(scores["label"], hybrid_pred)},
        ]
    )
    band_tbl = pd.DataFrame(
        [
            {"decider (uncertain band only)": "classifier at tuned threshold", **band_clf},
            {"decider (uncertain band only)": f"LLM {args.model}", **band_llm},
        ]
    )
    print(overall.to_string(index=False))
    print(band_tbl.to_string(index=False))

    def md(frame: pd.DataFrame) -> str:
        cols = list(frame.columns)
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        lines += [
            "| " + " | ".join(str(v) for v in r) + " |" for r in frame.itertuples(index=False)
        ]
        return "\n".join(lines)

    abt_i, buy_i = abt.set_index("id"), buy.set_index("id")
    examples = band.assign(agree=band["llm_same"].astype("boolean") == band["label"].astype(bool))
    wrong = examples[~examples["agree"].fillna(False)].head(6)
    right = examples[
        examples["agree"].fillna(False) & (examples["pred"] != examples["label"])
    ].head(6)

    def line(r) -> str:
        return (
            f"- label={r.label}, clf p={r.proba:.2f}, LLM={r.llm_same} ({r.llm_confidence}): "
            f"`{abt_i.loc[r.idAbt, 'name']}` vs `{buy_i.loc[r.idBuy, 'name']}`  \n"
            f"  reason: {r.llm_reason}"
        )

    parts = [
        "# LLM second opinion on uncertain pairs\n",
        f"Model: `{args.model}` via Ollama, temperature 0, JSON-only output validated against "
        f"a schema (bool / 0..1 float / string), one retry, then fall back to the classifier. "
        f"Uncertain band: {LOW} < p < {HIGH} of `{args.classifier}`. "
        f"{len(band)} pairs, {int(valid.sum())} valid answers, {time.time() - t0:.0f}s wall time "
        f"(cached answers are free).\n",
        "## Whole test set\n" + md(overall) + "\n",
        "## Inside the uncertain band\n" + md(band_tbl) + "\n",
        "## Where the LLM fixed the classifier\n"
        + "\n".join(line(r) for r in right.itertuples())
        + "\n",
        "## Where the LLM was wrong\n" + "\n".join(line(r) for r in wrong.itertuples()) + "\n",
    ]
    (REPORTS / "03_llm_review.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"Report written to {REPORTS / '03_llm_review.md'}")


if __name__ == "__main__":
    main()
