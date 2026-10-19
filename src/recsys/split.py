from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import DAY, Config


@dataclass
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    train_end: int
    test_start: int
    test_end: int


def temporal_splits(pairs: pd.DataFrame, cfg: Config):
    """split by time"""
    t_end = int(pairs["ts"].max()) + 1
    test_start = t_end - cfg.test_days * DAY
    val_start = test_start - cfg.gap_seconds - cfg.val_days * DAY

    def make(train_end, start, end):
        train = pairs[pairs["ts"] < train_end]
        test = pairs[(pairs["ts"] >= start) & (pairs["ts"] < end)]
        return Split(train.reset_index(drop=True), test.reset_index(drop=True), train_end, start, end)

    tuning = make(val_start - cfg.gap_seconds, val_start, test_start - cfg.gap_seconds)
    final = make(test_start - cfg.gap_seconds, test_start, t_end)
    return tuning, final


@dataclass
class Dataset:
    """data for training and eval"""
    X: sp.csr_matrix
    user_ids: np.ndarray
    item_ids: np.ndarray
    item_popularity: np.ndarray
    eval_users: np.ndarray
    targets: list
    target_organic: list
    n_test_pairs_total: int
    n_test_pairs_in_catalog: int


def build_dataset(split: Split, min_item_users: int) -> Dataset:
    train = split.train
    item_users = train.groupby("item_id")["uid"].nunique()
    catalog = np.sort(item_users[item_users >= min_item_users].index.to_numpy())
    train = train[train["item_id"].isin(catalog)]

    user_ids = np.sort(train["uid"].unique())
    rows = np.searchsorted(user_ids, train["uid"].to_numpy())
    cols = np.searchsorted(catalog, train["item_id"].to_numpy())
    X = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=(len(user_ids), len(catalog)))
    X.sum_duplicates()
    X.data[:] = 1.0
    popularity = np.asarray(X.sum(axis=0)).ravel()

    # only eval users that are in train
    test = split.test[split.test["uid"].isin(user_ids)]
    grouped = test.groupby("uid")
    eval_uids = np.array(sorted(grouped.groups.keys()), dtype=np.int64)
    targets, organic = [], []
    for uid in eval_uids:
        g = grouped.get_group(uid)
        targets.append(set(g["item_id"].tolist()))
        organic.append(dict(zip(g["item_id"].tolist(), g["is_organic"].tolist())))
    in_catalog = np.isin(test["item_id"].to_numpy(), catalog).sum()

    return Dataset(
        X=X, user_ids=user_ids, item_ids=catalog, item_popularity=popularity,
        eval_users=np.searchsorted(user_ids, eval_uids), targets=targets, target_organic=organic,
        n_test_pairs_total=len(test), n_test_pairs_in_catalog=int(in_catalog),
    )
