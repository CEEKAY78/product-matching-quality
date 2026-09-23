# Product matching quality

Detecting correct and incorrect product mappings between two retail catalogs with classical
machine learning, SQL-driven data diagnostics and a local LLM as a second opinion on the
uncertain cases.

**Business question.** Two shops list the same products under different names, codes and
prices. Which pairs of listings are the same product, and which of the mappings we already
have look wrong? The second question matters for any marketplace that builds product
mappings at scale: a model that scores existing mappings gives data stewards a review queue
instead of a blind audit.

**Data.** [Abt-Buy benchmark](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution)
(Leipzig University): 1,081 Abt products, 1,092 Buy products, 1,097 gold pairs. Consumer
electronics, real catalog noise: 61% missing prices on one side, 40% missing descriptions on
the other, brand only in the product name, latin-1 encoding.

## Results

| model | precision | recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| logistic regression | 0.747 | 0.824 | 0.784 | 0.988 | 0.837 |
| **random forest** | 0.852 | 0.800 | 0.825 | 0.991 | 0.895 |
| hist gradient boosting | 0.841 | 0.818 | 0.830 | 0.990 | 0.892 |
| random forest + LLM on uncertain 2% of pairs | 0.906 | 0.818 | **0.860** | | |

Held-out 30% split by Abt product id (6,908 candidate pairs, 330 true matches). Thresholds
tuned on train out-of-fold predictions only. Full tables, PR curves, feature importance and
error analysis: [reports/02_model_results.md](reports/02_model_results.md).

Three findings that shaped the model:

1. **Only 43% of true pairs share a model code in the product name** (SQL query
   `sql/05_shared_model_code.sql`), so an exact-key approach caps recall early; character
   n-gram TF-IDF similarity is the single most important feature.
2. **Error analysis paid off.** False negatives were dominated by hyphenated codes
   (`LCJ-THC/W` vs `LCJTHCW`, `CLI-221` vs `CLI221BLK`). Normalising them and adding a
   code-prefix feature moved random forest from F1 0.79 to 0.83 and PR-AUC from 0.86 to 0.89.
3. **Remaining false positives are colour and size variants** of one model
   (`DVP-FX820` vs `DVP-FX820/R`, `DC50W` vs `DC50B`). Classical string features cannot see
   that one character matters; a local LLM (qwen2.5 7B via Ollama) reviewing only the
   uncertain band lifts overall F1 by 3.5 points at ~2% of the inference cost of reviewing
   everything. Details and the LLM's reasons: [reports/03_llm_review.md](reports/03_llm_review.md).

Mapping quality monitoring: every gold pair is scored out-of-fold and the least plausible
mappings are written to [reports/mapping_review_queue.csv](reports/mapping_review_queue.csv),
sorted by confidence. In production this is the weekly job that feeds a data steward.

## Pipeline

```
data/raw (csv) -> pmq.data (clean, parse prices, recover brand, extract model codes)
              -> sql/*.sql via DuckDB on parquet  -> reports/01_data_quality.md
              -> pmq.blocking (model-code block + TF-IDF top-k; 1.18M pairs -> 20k, recall 99.3%)
              -> pmq.features (17 pairwise features: text similarity, codes, brand, price)
              -> pmq.model (3 classifiers, GroupKFold grid search, OOF threshold)
              -> reports/02_model_results.md + mapping_review_queue.csv
              -> pmq.llm_review (Ollama, JSON schema validation, retry, cache) -> reports/03_llm_review.md
```

Design choices worth defending:

- **Split by product, not by pair.** A random split over pairs leaks near-duplicate pairs of
  the same product into the test set and inflates scores.
- **Blocking recall is measured** before any model is trained. Pairs lost here can never be
  recovered.
- **Metrics for a 5% positive rate.** Grid search optimises average precision; accuracy is
  not reported because it would be 95% for a model that never predicts a match.
- **Missing prices are a feature, not a bug.** Encoded as an explicit flag plus a sentinel
  instead of being imputed.
- **The LLM validates, it does not replace.** Strict JSON schema, temperature 0, one retry,
  fallback to the classifier, on-disk cache. Every verdict comes with a one-sentence reason
  a human can audit.

## Run it

```bash
python -m venv venv && venv/Scripts/activate      # or source venv/bin/activate
pip install -e . && pip install pytest ruff
python scripts/download_data.py                   # Abt-Buy csv -> data/raw
python scripts/explore_sql.py                     # DuckDB profiling -> reports/01_data_quality.md
python scripts/run_pipeline.py                    # ~45 s on CPU -> reports/02_model_results.md
python scripts/llm_review.py --model qwen2.5:7b-instruct   # needs Ollama running locally
pytest                                            # 24 unit tests
ruff check . && ruff format --check .
```

Python 3.11, pandas, scikit-learn, DuckDB, rapidfuzz, matplotlib, requests. Everything runs on
a laptop CPU; the LLM step used a 6 GB GPU.

## Layout

```
src/pmq/        data.py  validate.py  blocking.py  features.py  model.py  llm_review.py
sql/            five exploration queries, each answering one question about the data
scripts/        download_data.py  explore_sql.py  run_pipeline.py  llm_review.py
tests/          pytest, no network, no data files needed
reports/        markdown reports, figures, mapping_review_queue.csv
```

## Limitations and next steps

- Single dataset and single split seed; a confidence interval over seeds would make the
  comparison between random forest and gradient boosting honest (they are within noise).
- Digit-only hyphenated codes (`010-10704-00` vs `0101070400`) are still not normalised; they
  show up at the top of the review queue.
- Random forest probabilities are not calibrated; the 0.3-0.7 uncertain band is a heuristic.
- Sentence embeddings (e.g. bge-m3) as an extra feature should help the false negatives where
  the two names share no tokens at all.
- Transfer to Amazon-Google, the sibling benchmark with the same schema, to check the
  features generalise.

## License

MIT. The Abt-Buy dataset is distributed by Leipzig University for research use and is downloaded, not redistributed, by this repository.
