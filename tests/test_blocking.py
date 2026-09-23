import pandas as pd

from pmq.blocking import build_candidates, evaluate_blocking


def _cat(rows):
    return pd.DataFrame(rows, columns=["id", "name", "description", "manufacturer", "price"])


ABT = _cat(
    [
        [1, "Sony Turntable - PSLX350H", "belt drive", "sony", 100.0],
        [2, "Canon PowerShot SD1100 IS", "camera", "canon", 200.0],
        [3, "Weber Genesis Grill", "gas grill", "weber", 900.0],
    ]
)
BUY = _cat(
    [
        [10, "Sony PSLX350H Belt Drive Turntable", "", "sony", 110.0],
        [11, "Canon SD1100IS Digital Camera", "powershot sd1100 is", "canon", 190.0],
        [12, "Weber Genesis E-310 Gas Grill", "", "weber", 950.0],
        [13, "Bose Headphones QC15", "", "bose", 300.0],
    ]
)
GOLD = pd.DataFrame({"idAbt": [1, 2, 3], "idBuy": [10, 11, 12]})


def test_candidates_contain_gold_pairs_and_are_labelled():
    cands = build_candidates(ABT, BUY, GOLD, top_k=2)
    assert {"idAbt", "idBuy", "label"} <= set(cands.columns)
    pairs = set(zip(cands["idAbt"], cands["idBuy"], strict=True))
    assert {(1, 10), (2, 11), (3, 12)} <= pairs
    assert cands.set_index(["idAbt", "idBuy"]).loc[(1, 10), "label"] == 1
    assert cands["label"].sum() == 3
    assert not cands.duplicated(["idAbt", "idBuy"]).any()


def test_candidates_share_model_code_even_if_text_differs():
    # code appears in Buy description only; blocking must still find it
    cands = build_candidates(ABT, BUY, GOLD, top_k=1)
    assert (2, 11) in set(zip(cands["idAbt"], cands["idBuy"], strict=True))


def test_evaluate_blocking_reports_recall_and_reduction():
    cands = build_candidates(ABT, BUY, GOLD, top_k=2)
    stats = evaluate_blocking(cands, GOLD, n_abt=len(ABT), n_buy=len(BUY))
    assert stats["gold_pairs"] == 3
    assert stats["recall_pct"] == 100.0
    assert 0 < stats["candidates"] < len(ABT) * len(BUY)
    assert 0 < stats["reduction_ratio_pct"] < 100
