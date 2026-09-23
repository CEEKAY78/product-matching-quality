import math

import pandas as pd

from pmq.features import FEATURE_COLUMNS, build_features


def _cat(rows):
    return pd.DataFrame(rows, columns=["id", "name", "description", "manufacturer", "price"])


ABT = _cat(
    [
        [1, "Sony Turntable - PSLX350H", "belt drive turntable", "sony", 100.0],
        [2, "Canon PowerShot SD1100 IS", "camera", "canon", float("nan")],
    ]
)
BUY = _cat(
    [
        [10, "Sony PSLX350H Belt Drive Turntable", "", "sony", 120.0],
        [11, "Bose Headphones QC15", "noise cancelling", "bose", 300.0],
    ]
)
CANDS = pd.DataFrame({"idAbt": [1, 1, 2], "idBuy": [10, 11, 11], "label": [1, 0, 0]})


def test_build_features_returns_one_row_per_candidate_with_all_columns():
    feats = build_features(CANDS, ABT, BUY)
    assert len(feats) == 3
    assert set(FEATURE_COLUMNS) <= set(feats.columns)
    assert feats[FEATURE_COLUMNS].isna().sum().sum() == 0


def test_true_pair_scores_higher_than_random_pair():
    feats = build_features(CANDS, ABT, BUY).set_index(["idAbt", "idBuy"])
    good, bad = feats.loc[(1, 10)], feats.loc[(1, 11)]
    assert good["shared_model_codes"] == 1 and bad["shared_model_codes"] == 0
    assert good["brand_match"] == 1 and bad["brand_match"] == 0
    assert good["name_token_set_ratio"] > bad["name_token_set_ratio"]
    assert good["name_char_tfidf_cos"] > bad["name_char_tfidf_cos"]


def test_price_features_handle_missing_values():
    feats = build_features(CANDS, ABT, BUY).set_index(["idAbt", "idBuy"])
    assert feats.loc[(1, 10), "price_both_present"] == 1
    assert math.isclose(feats.loc[(1, 10), "price_rel_gap"], 20 / 120)
    assert feats.loc[(2, 11), "price_both_present"] == 0
    assert feats.loc[(2, 11), "price_rel_gap"] == -1.0


def test_code_prefix_match_detects_code_with_suffix():
    abt = _cat([[1, "Canon Printer Black Ink Cartridge - CLI221BLK", "", "canon", float("nan")]])
    buy = _cat([[10, "Canon CLI-221 Black Ink Cartridge - 2946B001", "", "canon", 11.99]])
    cands = pd.DataFrame({"idAbt": [1], "idBuy": [10], "label": [1]})
    feats = build_features(cands, abt, buy)
    assert feats.loc[0, "code_prefix_match"] == 1
