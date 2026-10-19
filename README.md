# Music Recommendation Baselines (Yandex Yambda-50M)

Recommend songs a user has not heard yet, using implicit listening feedback. This stage builds
the data pipeline, a naive benchmark, and a course-level baseline:

| Model | Idea |
|---|---|
| **MostPopular** (naive) | Recommend the songs with the most distinct listeners in training, skipping songs the user already heard. Same list for everyone. |
| **ItemKNN-cosine** (baseline) | Item-based k-nearest-neighbours. Songs are vectors of their listeners. Each song keeps its *k* most cosine-similar songs (`sklearn.neighbors.NearestNeighbors`, brute force). A candidate's score is the sum of its similarities to the user's songs. *k* is tuned on validation. |

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_baselines.py         # full run: downloads ~370 MB once, then about 10-15 min on a laptop
python scripts/run_baselines.py --user-fraction 0.2 --knn-k 50 --out results/quick   # ~1 min
```

Tested with Python 3.10 on Windows (8 GB RAM). The pinned versions also install on Python 3.13 (Colab).
Results are deterministic. No step is random except the optional user sample, which uses `seed=42`.

## Pipeline

1. **Data**: `flat/50m/listens.parquet` from `yandex/yambda` on Hugging Face, pinned to revision
   `dd6f3a19…`. It has 46.5M listens, 9,238 users, 877k songs, and spans about 300 days. This is the
   listen-only subset of `multi_event.parquet`, which the exploration notebook used.
2. **Cleaning** (DuckDB): keep listens with `played_ratio_pct > 50`. Collapse to one row per
   (user, song), timestamped at the user's **first** such listen. The result is 9.0M pairs, cached
   to `data/processed/`.
3. **Split**: a **global temporal split**. One cutoff applies to everyone, so no future data leaks
   into training.
   ```
   |-------------- inner train --------------|gap|-- val (7d) --|gap|-- test (7d) --|
   ```
   * Tuning: inner train → val, used to choose KNN's *k* by NDCG@10.
   * Final: inner train + val → test. These are the reported numbers.
   * Gap = 30 minutes, so sessions don't cross a cutoff (same idea as the Yambda paper).
   * Because pairs are first listens, every held-out pair is a song **new** to that user.
4. **Catalog**: songs with at least 20 distinct listeners *in the training window* (computed after
   the split to avoid leakage).
5. **Evaluation**: users with at least 1 training song and at least 1 test song. Metrics are
   Precision, Recall, NDCG, and HitRate @10 and @50, plus catalog coverage. Held-out songs outside the
   catalog count as misses. Their share is reported as `eval_pairs_in_catalog_pct`, an upper bound on
   achievable recall.

## Outputs (`results/`)

| File | Contents |
|---|---|
| `baseline_table.md` / `.csv` | Final test metrics for both models (plus validation NDCG/recall) |
| `knn_tuning_validation.csv` | KNN *k* grid on the validation window |
| `data_summary.csv` | Sizes, density and catalog reachability for each split |
| `diagnostics_breakdown.csv`, `diagnostics.json` | Metrics by user activity (light/medium/heavy), organic vs. non-organic targets, popularity bias, overlap between models |
| `sample_recommendations.csv` | Example output in the `uid, rank, item_id` format |
| `run_config.json` | Exact config, best *k*, Python and library versions |
| `run_log.txt` | Console log of the run |

## Layout

```
src/recsys/   config.py  data.py  split.py  models.py  metrics.py
scripts/      run_baselines.py
Machine_Learning_Project.ipynb   # initial exploration / proposal notebook
```
