"""End-to-end pipeline: load -> block -> features -> train 3 models -> evaluate -> report.

Outputs:
- data/processed/candidates_features.parquet  (for the LLM review step)
- data/processed/test_scores.parquet
- reports/02_model_results.md + reports/figures/*.png
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve  # noqa: E402
from sklearn.model_selection import GroupKFold, cross_val_predict  # noqa: E402

from pmq.blocking import build_candidates, evaluate_blocking  # noqa: E402
from pmq.data import load_catalogs  # noqa: E402
from pmq.features import FEATURE_COLUMNS, build_features  # noqa: E402
from pmq.model import group_split, train_and_evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW, PROCESSED, REPORTS = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "reports"
FIG = REPORTS / "figures"


def md(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def dict_md(d: dict) -> str:
    return "| metric | value |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in d.items())


def describe_pair(row, abt: pd.DataFrame, buy: pd.DataFrame) -> str:
    a = abt.set_index("id").loc[row.idAbt]
    b = buy.set_index("id").loc[row.idBuy]
    return f"`{a['name']}` ({a['price']}) vs `{b['name']}` ({b['price']})"


def main() -> None:
    t0 = time.time()
    abt, buy, gold = load_catalogs(RAW)
    cands = build_candidates(abt, buy, gold, top_k=10)
    blocking = evaluate_blocking(cands, gold, len(abt), len(buy))
    feats = build_features(cands, abt, buy)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(PROCESSED / "candidates_features.parquet", index=False)

    train, test = group_split(feats, test_size=0.3, seed=42)
    result = train_and_evaluate(train, test, FEATURE_COLUMNS, cv_splits=5)
    result.test_scores.to_parquet(PROCESSED / "test_scores.parquet", index=False)
    print(result.metrics.to_string(index=False))

    best_name = result.metrics.sort_values("pr_auc", ascending=False).iloc[0]["model"]
    best = result.estimators[best_name]
    best_scores = result.test_scores[result.test_scores["model"] == best_name].reset_index(
        drop=True
    )
    threshold = float(result.metrics.set_index("model").loc[best_name, "threshold"])

    # figures: PR curves + permutation importance
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name in result.metrics["model"]:
        s = result.test_scores[result.test_scores["model"] == name]
        p, r, _ = precision_recall_curve(s["label"], s["proba"])
        ax.plot(r, p, label=name)
    ax.set_xlabel("recall"), ax.set_ylabel("precision"), ax.set_title("Precision-recall (test)")
    ax.legend(), fig.tight_layout(), fig.savefig(FIG / "pr_curves.png", dpi=110)

    imp = permutation_importance(
        best,
        test[FEATURE_COLUMNS].to_numpy(),
        test["label"].to_numpy(),
        scoring="average_precision",
        n_repeats=10,
        random_state=0,
        n_jobs=-1,
    )
    imp_df = (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": imp.importances_mean.round(4)})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#4c72b0")
    ax.set_title(f"Permutation importance ({best_name}, PR-AUC drop)")
    fig.tight_layout(), fig.savefig(FIG / "feature_importance.png", dpi=110)

    # error analysis
    cm = confusion_matrix(best_scores["label"], best_scores["pred"])
    fn = best_scores[(best_scores["label"] == 1) & (best_scores["pred"] == 0)].nsmallest(8, "proba")
    fp = best_scores[(best_scores["label"] == 0) & (best_scores["pred"] == 1)].nlargest(8, "proba")
    uncertain = best_scores[(best_scores["proba"] > 0.3) & (best_scores["proba"] < 0.7)]

    # mapping quality monitoring: score EVERY gold pair with out-of-fold probabilities
    # (GroupKFold over all candidates, so no pair is scored by a model that saw it) and
    # write a review queue sorted by ascending confidence for data stewards
    oof = cross_val_predict(
        clone(best),
        feats[FEATURE_COLUMNS].to_numpy(),
        feats["label"].to_numpy(),
        cv=GroupKFold(n_splits=5),
        groups=feats["idAbt"],
        method="predict_proba",
    )[:, 1]
    queue = feats[["idAbt", "idBuy", "label"]].assign(proba=oof.round(4))
    queue = queue[queue["label"] == 1].sort_values("proba").reset_index(drop=True)
    queue["abt_name"] = queue["idAbt"].map(abt.set_index("id")["name"])
    queue["buy_name"] = queue["idBuy"].map(buy.set_index("id")["name"])
    queue.to_csv(REPORTS / "mapping_review_queue.csv", index=False)
    suspicious = queue.head(8)
    n_flagged = int((queue["proba"] < threshold).sum())

    parts = [
        "# Model results\n",
        f"Pipeline run time: {time.time() - t0:.0f}s on CPU. Split: 70/30 by Abt product id, "
        f"{len(train)} train / {len(test)} test pairs.\n",
        "## Blocking\n" + dict_md(blocking) + "\n",
        "## Models (test set, threshold tuned on train out-of-fold predictions)\n"
        + md(result.metrics)
        + "\n",
        "Best hyperparameters:\n\n```json\n"
        + json.dumps(result.best_params, indent=2, default=str)
        + "\n```\n",
        "![PR curves](figures/pr_curves.png)\n",
        f"## Feature importance ({best_name})\n" + md(imp_df.head(10)) + "\n",
        "![feature importance](figures/feature_importance.png)\n",
        f"## Error analysis ({best_name}, threshold {threshold:.2f})\n",
        f"Confusion matrix (rows = true 0/1, cols = predicted 0/1):\n\n```\n{cm}\n```\n",
        f"Uncertain pairs (0.3 < p < 0.7): {len(uncertain)} of {len(best_scores)} test pairs, "
        f"{int(uncertain['label'].sum())} of them true matches.\n",
        "### False negatives (true matches the model rejected)\n",
        "\n".join(f"- p={r.proba:.2f} " + describe_pair(r, abt, buy) for r in fn.itertuples()),
        "\n### False positives (non-matches the model accepted)\n",
        "\n".join(f"- p={r.proba:.2f} " + describe_pair(r, abt, buy) for r in fp.itertuples()),
        "\n### Mapping quality monitoring: gold pairs with the lowest model confidence\n",
        f"All {len(queue)} gold pairs in the candidate set were scored out-of-fold; "
        f"{n_flagged} fall below the decision threshold and are written to "
        "`reports/mapping_review_queue.csv` (sorted by confidence) for a data steward. "
        "Lowest eight:\n",
        "\n".join(
            f"- p={r.proba:.2f} " + describe_pair(r, abt, buy) for r in suspicious.itertuples()
        ),
        "",
    ]
    (REPORTS / "02_model_results.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"Best model: {best_name}. Report written to {REPORTS / '02_model_results.md'}")


if __name__ == "__main__":
    main()
