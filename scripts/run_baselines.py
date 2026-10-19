import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.config import Config  
from recsys.data import load_pairs
from recsys.metrics import evaluate, user_metrics
from recsys.models import ItemKNN, MostPopular 
from recsys.split import build_dataset, temporal_splits


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_model(model, ds, cfg):
    t0 = time.time()
    model.fit(ds.X)
    fit_s = time.time() - t0
    t0 = time.time()
    rec_idx = model.recommend(ds.eval_users, max(cfg.ks))
    rec_s = time.time() - t0
    rec_items = ds.item_ids[rec_idx]
    for r, idx in zip(ds.eval_users[:500], rec_idx[:500]):
        seen = ds.X.indices[ds.X.indptr[r]:ds.X.indptr[r + 1]]
        assert not np.isin(idx, seen).any(), "recommended an already-listened song"
    metrics = evaluate(rec_items, ds.targets, cfg.ks, n_catalog=len(ds.item_ids))
    metrics.update(fit_seconds=round(fit_s, 1), recommend_seconds=round(rec_s, 1))
    return rec_idx, rec_items, metrics


def describe(name, split, ds):
    return {
        "split": name,
        "train_pairs": len(split.train), "train_users": int(split.train["uid"].nunique()),
        "heldout_pairs": len(split.test), "heldout_users": int(split.test["uid"].nunique()),
        "catalog_items": len(ds.item_ids), "matrix_users": ds.X.shape[0], "matrix_nnz": int(ds.X.nnz),
        "density_pct": round(100 * ds.X.nnz / (ds.X.shape[0] * ds.X.shape[1]), 4),
        "eval_users": len(ds.eval_users),
        "eval_pairs": ds.n_test_pairs_total,
        "eval_pairs_in_catalog_pct": round(100 * ds.n_test_pairs_in_catalog / max(ds.n_test_pairs_total, 1), 2),
    }


def diagnostics(ds, results, k=10):
    """extra breakdowns"""
    out = {}
    hist = np.diff(ds.X.indptr)[ds.eval_users]
    buckets = pd.qcut(hist, q=3, labels=["light", "medium", "heavy"])
    edges = {b: [int(hist[buckets == b].min()), int(hist[buckets == b].max())] for b in ["light", "medium", "heavy"]}
    out["activity_bucket_train_items_range"] = edges

    pop = ds.item_popularity
    pop_rank_pct = pd.Series(pop).rank(pct=True).to_numpy()
    head = pop_rank_pct >= 0.9

    rows = []
    for name, (rec_idx, rec_items) in results.items():
        per_user = [user_metrics(r, t, k) for r, t in zip(rec_items, ds.targets)]
        ndcg = np.array([m[f"ndcg@{k}"] for m in per_user])
        hit = np.array([m[f"hitrate@{k}"] for m in per_user])
        row = {"model": name}
        for b in ["light", "medium", "heavy"]:
            row[f"ndcg@{k}_{b}"] = ndcg[buckets == b].mean()
            row[f"hitrate@{k}_{b}"] = hit[buckets == b].mean()
        for flag, label in [(1, "organic"), (0, "nonorganic")]:
            hits = total = 0
            for r, org in zip(rec_items, ds.target_organic):
                tgt = {i for i, o in org.items() if o == flag}
                if tgt:
                    hits += len(tgt.intersection(r[:k]))
                    total += len(tgt)
            row[f"recall@{k}_{label}_targets"] = hits / max(total, 1)
        top = rec_idx[:, :k]
        row[f"mean_pop_percentile@{k}"] = pop_rank_pct[top].mean()
        row[f"head10pct_share@{k}"] = head[top].mean()
        row[f"unique_items@{k}"] = int(len(np.unique(top)))
        rows.append(row)
    out["breakdown"] = rows

    names = list(results)
    if len(names) == 2:
        a, b = (results[n][0][:, :k] for n in names)
        jacc = [len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(a, b)]
        out[f"jaccard@{k}_{names[0]}_vs_{names[1]}"] = float(np.mean(jacc))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--user-fraction", type=float, default=Config.user_fraction)
    p.add_argument("--min-item-users", type=int, default=Config.min_item_users)
    p.add_argument("--knn-k", type=int, nargs="*", help="KNN neighbour grid (default from config)")
    p.add_argument("--out", type=str, default=None, help="results directory")
    args = p.parse_args()

    cfg = Config(user_fraction=args.user_fraction, min_item_users=args.min_item_users)
    if args.knn_k:
        cfg.knn_k_grid = args.knn_k
    if args.out:
        cfg.results_dir = Path(args.out)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.seed)

    log("loading pairs")
    pairs = load_pairs(cfg)
    log(f"{len(pairs):,} pairs, {pairs['uid'].nunique():,} users, {pairs['item_id'].nunique():,} songs")

    tuning, final = temporal_splits(pairs, cfg)
    ds_val = build_dataset(tuning, cfg.min_item_users)
    ds_test = build_dataset(final, cfg.min_item_users)
    data_summary = [describe("tuning (train -> val)", tuning, ds_val), describe("final (train+val -> test)", final, ds_test)]
    for d in data_summary:
        log(d)

    tuning_rows = []
    for k in cfg.knn_k_grid:
        log(f"validation: ItemKNN k={k}")
        _, _, m = run_model(ItemKNN(k=k), ds_val, cfg)
        tuning_rows.append({"model": "ItemKNN-cosine", "k": k, **m})
        log({x: round(float(y), 4) for x, y in m.items() if "@10" in x})
    _, _, m = run_model(MostPopular(), ds_val, cfg)
    tuning_rows.append({"model": "MostPopular", "k": None, **m})
    tuning_df = pd.DataFrame(tuning_rows)
    best_k = int(tuning_df.dropna(subset=["k"]).sort_values(cfg.tune_metric, ascending=False).iloc[0]["k"])
    log(f"best k = {best_k} by {cfg.tune_metric}")

    final_rows, recs = [], {}
    for model in [MostPopular(), ItemKNN(k=best_k)]:
        log(f"test: {model.name}")
        rec_idx, rec_items, m = run_model(model, ds_test, cfg)
        label = model.name + (f" (k={best_k})" if isinstance(model, ItemKNN) else "")
        val_m = tuning_df[(tuning_df["model"] == model.name) &
                          ((tuning_df["k"] == best_k) if isinstance(model, ItemKNN) else tuning_df["k"].isna())].iloc[0]
        final_rows.append({"model": label, **m, "val_ndcg@10": val_m["ndcg@10"], "val_recall@10": val_m["recall@10"]})
        recs[label] = (rec_idx, rec_items)
        log({x: round(float(y), 4) for x, y in m.items() if "@10" in x})

    table = pd.DataFrame(final_rows)
    diag = diagnostics(ds_test, recs)

    out = cfg.results_dir
    table.to_csv(out / "baseline_table.csv", index=False)
    tuning_df.to_csv(out / "knn_tuning_validation.csv", index=False)
    pd.DataFrame(data_summary).to_csv(out / "data_summary.csv", index=False)
    pd.DataFrame(diag["breakdown"]).to_csv(out / "diagnostics_breakdown.csv", index=False)
    with open(out / "diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2, default=float)

    sample_rows = []
    for label, (_, rec_items) in recs.items():
        for u, items in zip(ds_test.user_ids[ds_test.eval_users[:5]], rec_items[:5, :10]):
            sample_rows += [{"model": label, "uid": int(u), "rank": i + 1, "item_id": int(it)} for i, it in enumerate(items)]
    pd.DataFrame(sample_rows).to_csv(out / "sample_recommendations.csv", index=False)

    import sklearn, scipy, duckdb
    with open(out / "run_config.json", "w") as f:
        json.dump({"config": cfg.to_dict(), "best_k": best_k, "python": platform.python_version(),
                   "platform": platform.platform(),
                   "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                                "scikit-learn": sklearn.__version__, "duckdb": duckdb.__version__}}, f, indent=2)

    cols = ["model"] + [f"{m}@{k}" for k in cfg.ks for m in ["precision", "recall", "ndcg", "hitrate", "coverage"]]
    with open(out / "baseline_table.md", "w") as f:
        f.write(table[cols].to_markdown(index=False, floatfmt=".4f"))
    print("\n" + table[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    log(f"results written to {out}")


if __name__ == "__main__":
    main()
