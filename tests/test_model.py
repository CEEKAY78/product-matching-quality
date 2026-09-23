import numpy as np
import pandas as pd

from pmq.model import MODEL_SPECS, group_split, pick_threshold, train_and_evaluate


def _synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, n)
    frame = pd.DataFrame(
        {
            "idAbt": rng.integers(0, 40, n),
            "idBuy": np.arange(n),
            "label": label,
            "f1": label + rng.normal(0, 0.5, n),
            "f2": rng.normal(0, 1, n),
        }
    )
    return frame


def test_group_split_keeps_each_abt_product_on_one_side():
    frame = _synthetic()
    train, test = group_split(frame, test_size=0.3, seed=1)
    assert len(train) + len(test) == len(frame)
    assert set(train["idAbt"]).isdisjoint(set(test["idAbt"]))


def test_pick_threshold_maximises_f1():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    t = pick_threshold(y, p)
    assert 0.4 < t <= 0.6


def test_train_and_evaluate_returns_metrics_for_every_model():
    frame = _synthetic()
    train, test = group_split(frame, test_size=0.3, seed=1)
    result = train_and_evaluate(train, test, ["f1", "f2"], cv_splits=3, fast=True)
    assert set(result.metrics["model"]) == set(MODEL_SPECS)
    for col in ("precision", "recall", "f1", "roc_auc", "pr_auc", "threshold"):
        assert col in result.metrics.columns
    assert result.metrics["roc_auc"].min() > 0.8
    assert len(result.test_scores) == len(test) * len(MODEL_SPECS)
    assert {"model", "proba"} <= set(result.test_scores.columns)
