"""evaluation metrics"""
import numpy as np


def user_metrics(recs, targets: set, k: int) -> dict:
    top = list(recs[:k])
    hits = np.array([item in targets for item in top], dtype=float)
    n_hits = hits.sum()
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    ideal = discounts[: min(len(targets), k)].sum()
    return {
        f"precision@{k}": n_hits / k,
        f"recall@{k}": n_hits / len(targets),
        f"ndcg@{k}": (hits * discounts[: len(top)]).sum() / ideal if ideal > 0 else 0.0,
        f"hitrate@{k}": float(n_hits > 0),
    }


def evaluate(rec_item_ids: np.ndarray, targets: list, ks, n_catalog: int) -> dict:
    """average metrics over users"""
    totals = {}
    for recs, tgt in zip(rec_item_ids, targets):
        for k in ks:
            for name, v in user_metrics(recs, tgt, k).items():
                totals[name] = totals.get(name, 0.0) + v
    result = {name: v / len(targets) for name, v in totals.items()}
    for k in ks:
        result[f"coverage@{k}"] = len(np.unique(rec_item_ids[:, :k])) / n_catalog
    return result
