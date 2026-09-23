import pandas as pd

from pmq.validate import mapping_quality, profile_catalog


def _frame(rows):
    return pd.DataFrame(rows, columns=["id", "name", "description", "manufacturer", "price"])


def test_profile_catalog_reports_missing_and_duplicates():
    abt = _frame(
        [
            [1, "Sony A - X1", "", "sony", 10.0],
            [2, "Sony A - X1", "d", "sony", float("nan")],
            [3, "", "d", "", float("nan")],
        ]
    )
    profile = profile_catalog(abt)
    assert profile["rows"] == 3
    assert profile["missing_price_pct"] == 66.7
    assert profile["missing_description_pct"] == 33.3
    assert profile["empty_name_rows"] == 1
    assert profile["duplicate_name_rows"] == 2


def test_mapping_quality_flags_dangling_and_one_to_many():
    abt = _frame([[1, "a", "", "x", 1.0], [2, "b", "", "x", 1.0]])
    buy = _frame([[10, "a", "", "x", 1.0], [11, "b", "", "x", 1.0]])
    mapping = pd.DataFrame({"idAbt": [1, 1, 2, 3], "idBuy": [10, 11, 11, 10]})
    q = mapping_quality(abt, buy, mapping)
    assert q["pairs"] == 4
    assert q["dangling_abt_ids"] == 1  # id 3 does not exist in abt
    assert q["abt_ids_with_many_buy"] == 1  # abt 1 -> buy 10 and 11
    assert q["buy_ids_with_many_abt"] == 2  # buy 10 <- abt 1,3 ; buy 11 <- abt 1,2
    assert q["abt_coverage_pct"] == 100.0
    assert q["buy_coverage_pct"] == 100.0
