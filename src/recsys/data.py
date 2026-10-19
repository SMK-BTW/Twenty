"""loading and cleaning the data"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from .config import Config


def download_raw(cfg: Config) -> Path:
    """download the listens file"""
    path = hf_hub_download(
        repo_id=cfg.hf_repo,
        filename=cfg.hf_file,
        repo_type="dataset",
        revision=cfg.hf_revision,
        local_dir=cfg.raw_dir,
    )
    return Path(path)


def build_pairs(raw_path: Path, min_played_ratio_pct: int) -> pd.DataFrame:
    """one row per user and song"""
    con = duckdb.connect()
    df = con.execute(
        """
        SELECT uid,
               item_id,
               min(timestamp)                       AS ts,
               arg_min(is_organic, timestamp)       AS is_organic,
               count(*)                             AS n_listens
        FROM read_parquet(?)
        WHERE played_ratio_pct > ?
        GROUP BY uid, item_id
        ORDER BY uid, ts, item_id
        """,
        [str(raw_path), min_played_ratio_pct],
    ).df()
    con.close()
    return df.astype({"uid": "int64", "item_id": "int64", "ts": "int64",
                      "is_organic": "int8", "n_listens": "int32"})


def sample_users(pairs: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """sample some users"""
    if fraction >= 1.0:
        return pairs
    users = np.sort(pairs["uid"].unique())
    rng = np.random.default_rng(seed)
    keep = rng.choice(users, size=max(1, int(round(len(users) * fraction))), replace=False)
    return pairs[pairs["uid"].isin(keep)].reset_index(drop=True)


def load_pairs(cfg: Config) -> pd.DataFrame:
    """load the pairs"""
    cache = cfg.processed_dir / f"pairs_played{cfg.min_played_ratio_pct}.parquet"
    if cache.exists():
        pairs = pd.read_parquet(cache)
    else:
        raw = download_raw(cfg)
        pairs = build_pairs(raw, cfg.min_played_ratio_pct)
        cache.parent.mkdir(parents=True, exist_ok=True)
        pairs.to_parquet(cache, index=False)
    return sample_users(pairs, cfg.user_fraction, cfg.seed)
