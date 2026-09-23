"""Second-opinion review of uncertain pairs with a local LLM (Ollama).

The classifier is cheap and handles ~98% of pairs confidently. Pairs in the uncertain band
(0.3 < p < 0.7) are sent to a local instruction-tuned model that sees both listings and must
answer in a strict JSON schema. Every answer is validated; an invalid answer is retried once
and then treated as "no opinion" (the classifier's decision stands). Answers are cached on
disk so re-runs cost nothing.

This is intentionally a human-in-the-loop style design: the LLM does not replace the model,
it triages the grey zone, and its reasons are surfaced in the report for a data steward.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import requests

SYSTEM_PROMPT = (
    "You are a product catalog expert. You compare two product listings from different "
    "online shops and decide whether they describe exactly the same product (same model, "
    "same variant such as colour or size). Different colour or capacity variants of the same "
    "model are NOT the same product. Answer only with JSON: "
    '{"same_product": true|false, "confidence": <number 0..1>, "reason": "<one sentence>"}'
)


@dataclass(frozen=True)
class Verdict:
    same_product: bool
    confidence: float
    reason: str


def parse_verdict(text: str) -> Verdict | None:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    same, conf, reason = data.get("same_product"), data.get("confidence"), data.get("reason", "")
    if not isinstance(same, bool) or isinstance(conf, bool):
        return None
    if not isinstance(conf, int | float) or not 0.0 <= float(conf) <= 1.0:
        return None
    return Verdict(same_product=same, confidence=float(conf), reason=str(reason))


class OllamaClient:
    def __init__(self, model: str = "qwen2.5:7b-instruct", url: str = "http://localhost:11434"):
        self.model, self.url = model, url

    def ask(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 200},
        }
        response = requests.post(f"{self.url}/api/chat", json=payload, timeout=180)
        response.raise_for_status()
        return response.json()["message"]["content"]


def _listing(row: pd.Series) -> str:
    price = "unknown" if pd.isna(row["price"]) else f"{row['price']:.2f} USD"
    desc = str(row["description"])[:300] or "(none)"
    return f"name: {row['name']}\nbrand: {row['manufacturer']}\nprice: {price}\ndescription: {desc}"


def build_prompt(a: pd.Series, b: pd.Series) -> str:
    return (
        f"Listing A (shop Abt):\n{_listing(a)}\n\n"
        f"Listing B (shop Buy):\n{_listing(b)}\n\nSame product?"
    )


def review_pairs(
    pairs: pd.DataFrame,
    abt: pd.DataFrame,
    buy: pd.DataFrame,
    client,
    cache: dict[str, dict],
    on_progress=None,
) -> pd.DataFrame:
    """Return pairs with llm_same (bool | None), llm_confidence, llm_reason columns."""
    abt_i, buy_i = abt.set_index("id"), buy.set_index("id")
    rows = []
    for n, (ida, idb) in enumerate(zip(pairs["idAbt"], pairs["idBuy"], strict=True), start=1):
        key = f"{ida}:{idb}"
        if key not in cache:
            prompt = build_prompt(abt_i.loc[ida], buy_i.loc[idb])
            verdict = parse_verdict(client.ask(prompt)) or parse_verdict(client.ask(prompt))
            cache[key] = (
                asdict(verdict)
                if verdict
                else {"same_product": None, "confidence": None, "reason": "invalid answer"}
            )
        v = cache[key]
        rows.append(
            {
                "idAbt": ida,
                "idBuy": idb,
                "llm_same": v["same_product"],
                "llm_confidence": v["confidence"],
                "llm_reason": v["reason"],
            }
        )
        if on_progress:
            on_progress(n, len(pairs))
    return pd.DataFrame(rows)


def combine_hybrid(
    scores: pd.DataFrame, llm_same: pd.Series, low: float = 0.3, high: float = 0.7
) -> pd.Series:
    """Classifier decision everywhere except the uncertain band, where the LLM verdict wins
    (when it gave a valid one)."""
    pred = scores["pred"].astype(int).copy()
    band = (scores["proba"] > low) & (scores["proba"] < high)
    valid = llm_same.notna()
    pred[band & valid] = llm_same[band & valid].astype(bool).astype(int)
    return pred


def load_cache(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_cache(path: Path, cache: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
