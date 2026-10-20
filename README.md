# Music Recommendation Baselines (Yandex Yambda-50M)

Recommend songs a user has not heard yet, using implicit listening feedback. This stage builds
the data pipeline, a naive benchmark, and a course-level baseline:

| Model | Idea |
|---|---|
| **MostPopular** (naive) | Recommend the songs with the most distinct listeners in training, skipping songs the user already heard. Same list for everyone. |
| **ItemKNN-cosine** (baseline) | Item-based k-nearest-neighbours. Songs are vectors of their listeners. Each song keeps its *k* most cosine-similar songs (`sklearn.neighbors.NearestNeighbors`, brute force). A candidate's score is the sum of its similarities to the user's songs. *k* is tuned on validation. |

## Quick start

Open `Machine_Learning_Project.ipynb` and run all cells. The first code cell installs the exact
pinned library versions:

```
numpy==2.1.3 pandas==2.2.3 scipy==1.14.1 scikit-learn==1.6.1 duckdb==1.1.3
pyarrow==18.1.0 huggingface_hub==0.26.5
```

Running locally, use a fresh virtual environment (`python -m venv .venv`) so these don't clash with
other projects. The first run downloads about 750 MB (kept in `data/`), and the full pipeline takes about 10-15 min on a laptop.

On Colab, use the badge at the top of the notebook. Restart the runtime once after the install cell if
Colab asks.

Tested with Python 3.10 on Windows (8 GB RAM). The pinned versions also install on Python 3.13 (Colab).
Results are deterministic. The baseline pipeline has no random steps, and the exploration sample uses `REPEATABLE (42)`.

## Pipeline

1. **Data**: `flat/50m/listens.parquet` from `yandex/yambda` on Hugging Face, pinned to revision
   `dd6f3a19…`. It has 46.5M listens, 9,238 users, 877k songs, and spans about 300 days. This is the
   listen-only subset of `multi_event.parquet`, which the exploration notebook used.
2. **Cleaning** (DuckDB): keep listens with `played_ratio_pct > 50`. Collapse to one row per
   (user, song), timestamped at the user's **first** such listen. The result is 9.0M pairs.
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
   catalog count as misses. The split cell prints their share, which is an upper bound on achievable
   recall.

## Layout

Everything is in `Machine_Learning_Project.ipynb`:

1. **setup**: pinned installs
2. **initial exploration (proposal)**: 100k-row sample of `multi_event.parquet` (seeded), event types, cleaning, `validate_recommendations`
3. **baseline pipeline**: data, split, models, metrics and validation, tuning, baseline table, diagnostics
