import math

import pandas as pd

from pmq.data import (
    extract_brand,
    extract_model_codes,
    load_catalogs,
    normalize_text,
    parse_price,
)


def test_parse_price_handles_dollar_and_commas():
    assert parse_price("$399.00") == 399.0
    assert parse_price("$1,299.99") == 1299.99


def test_parse_price_missing_returns_nan():
    assert math.isnan(parse_price(None))
    assert math.isnan(parse_price(float("nan")))
    assert math.isnan(parse_price("n/a"))


def test_normalize_text_lowercases_and_strips_punctuation():
    assert normalize_text("Sony Turntable - PSLX350H/ Belt") == "sony turntable pslx350h belt"
    assert normalize_text(None) == ""


def test_extract_model_codes_finds_alphanumeric_tokens():
    codes = extract_model_codes("Sony Turntable - PSLX350H")
    assert codes == {"pslx350h"}
    codes = extract_model_codes("Bose Acoustimass 5 Series III Speaker System - AM53BK")
    assert codes == {"am53bk"}


def test_extract_model_codes_ignores_plain_words_and_short_numbers():
    assert extract_model_codes("Black Finish Speaker") == set()
    assert extract_model_codes("Series 5") == set()


def test_extract_brand_is_first_word_lowercased():
    assert extract_brand("Sony Turntable - PSLX350H") == "sony"
    assert extract_brand("") == ""


def test_load_catalogs_returns_clean_frames(tmp_path):
    (tmp_path / "Abt.csv").write_text(
        '"id","name","description","price"\n"1","Sony X - AB12","desc","$10.00"\n',
        encoding="latin-1",
    )
    (tmp_path / "Buy.csv").write_text(
        '"id","name","description","manufacturer","price"\n"9","Sony X AB12","","Sony",\n',
        encoding="latin-1",
    )
    (tmp_path / "abt_buy_perfectMapping.csv").write_text(
        '"idAbt","idBuy"\n"1","9"\n', encoding="latin-1"
    )
    abt, buy, mapping = load_catalogs(tmp_path)
    assert list(abt.columns) == ["id", "name", "description", "manufacturer", "price"]
    assert abt.loc[0, "price"] == 10.0
    assert abt.loc[0, "manufacturer"] == "sony"
    assert pd.isna(buy.loc[0, "price"])
    assert buy.loc[0, "manufacturer"] == "sony"
    assert mapping.iloc[0].tolist() == [1, 9]
