"""Training and evaluation of pair classifiers.

Design decisions:
- split by Abt product id (GroupShuffleSplit), so every product appears on one side only;
  a random split over pairs would leak near-duplicate pairs between train and test and
  overstate the score;
- hyperparameters via GridSearchCV on GroupKFold, scoring = average precision, because the
  positive rate is ~5% and accuracy would be meaningless;
- decision threshold is tuned on train out-of-fold predictions, never on the test set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_SPECS: dict[str, tuple[object, dict[str, list]]] = {
    "logistic_regression": (
        Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        {"clf__C": [0.1, 1.0, 10.0]},
    ),
    "random_forest": (
        RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", random_state=0, n_jobs=-1
        ),
        {"max_depth": [6, 10, None], "min_samples_leaf": [1, 3]},
    ),
    "hist_gradient_boosting": (
        HistGradientBoostingClassifier(random_state=0),
        {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31], "max_iter": [200]},
    ),
}


@dataclass
class TrainResult:
    metrics: pd.DataFrame
    test_scores: pd.DataFrame  # idAbt, idBuy, label, model, proba
    best_params: dict[str, dict]
    estimators: dict[str, object]


def group_split(frame: pd.DataFrame, test_size: float = 0.3, seed: int = 42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(frame, groups=frame["idAbt"]))
    return frame.iloc[train_idx].reset_index(drop=True), frame.iloc[test_idx].reset_index(drop=True)


def pick_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def train_and_evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    cv_splits: int = 5,
    fast: bool = False,
) -> TrainResult:
    x_tr, y_tr, g_tr = train[feature_columns].to_numpy(), train["label"].to_numpy(), train["idAbt"]
    x_te, y_te = test[feature_columns].to_numpy(), test["label"].to_numpy()
    cv = GroupKFold(n_splits=cv_splits)
    rows, scores, best_params, estimators = [], [], {}, {}
    for name, (estimator, grid) in MODEL_SPECS.items():
        if fast:
            grid = {k: v[:1] for k, v in grid.items()}
        search = GridSearchCV(estimator, grid, scoring="average_precision", cv=cv, n_jobs=-1)
        search.fit(x_tr, y_tr, groups=g_tr)
        best = search.best_estimator_
        oof = cross_val_predict(best, x_tr, y_tr, cv=cv, groups=g_tr, method="predict_proba")[:, 1]
        threshold = pick_threshold(y_tr, oof)
        proba = best.predict_proba(x_te)[:, 1]
        pred = (proba >= threshold).astype(int)
        rows.append(
            {
                "model": name,
                "cv_pr_auc": round(float(search.best_score_), 4),
                "precision": round(precision_score(y_te, pred, zero_division=0), 4),
                "recall": round(recall_score(y_te, pred), 4),
                "f1": round(f1_score(y_te, pred), 4),
                "roc_auc": round(roc_auc_score(y_te, proba), 4),
                "pr_auc": round(average_precision_score(y_te, proba), 4),
                "threshold": round(threshold, 3),
            }
        )
        scores.append(
            pd.DataFrame(
                {
                    "idAbt": test["idAbt"].to_numpy(),
                    "idBuy": test["idBuy"].to_numpy(),
                    "label": y_te,
                    "model": name,
                    "proba": proba,
                    "pred": pred,
                }
            )
        )
        best_params[name] = search.best_params_
        estimators[name] = best
    return TrainResult(
        pd.DataFrame(rows), pd.concat(scores, ignore_index=True), best_params, estimators
    )
