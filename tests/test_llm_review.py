import pandas as pd

from pmq.llm_review import (
    Verdict,
    combine_hybrid,
    parse_verdict,
    review_pairs,
)


def test_parse_verdict_accepts_valid_json():
    v = parse_verdict('{"same_product": true, "confidence": 0.9, "reason": "same code"}')
    assert v == Verdict(same_product=True, confidence=0.9, reason="same code")


def test_parse_verdict_rejects_invalid_or_incomplete_json():
    assert parse_verdict("not json") is None
    assert parse_verdict('{"same_product": "yes"}') is None
    assert parse_verdict('{"same_product": true, "confidence": 7, "reason": ""}') is None


class FakeClient:
    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def ask(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.answers.pop(0)


def test_review_pairs_retries_once_and_uses_cache():
    pairs = pd.DataFrame({"idAbt": [1, 2], "idBuy": [10, 20]})
    abt = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["a", "b"],
            "description": ["", ""],
            "manufacturer": ["", ""],
            "price": [1.0, 2.0],
        }
    )
    buy = abt.assign(id=[10, 20])
    client = FakeClient(
        [
            "garbage",
            '{"same_product": true, "confidence": 0.8, "reason": "r"}',
            '{"same_product": false, "confidence": 0.6, "reason": "s"}',
        ]
    )
    cache: dict = {}
    out = review_pairs(pairs, abt, buy, client, cache)
    assert out["llm_same"].tolist() == [True, False]
    assert len(client.calls) == 3  # one retry for the garbage answer
    out2 = review_pairs(pairs, abt, buy, client, cache)
    assert len(client.calls) == 3  # served from cache
    assert out2["llm_same"].tolist() == [True, False]


def test_combine_hybrid_overrides_only_uncertain_band():
    scores = pd.DataFrame({"proba": [0.05, 0.5, 0.95], "pred": [0, 1, 1]})
    llm = pd.Series([True, False, False])
    pred = combine_hybrid(scores, llm, low=0.3, high=0.7)
    assert pred.tolist() == [0, 0, 1]
