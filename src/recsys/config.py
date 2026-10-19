"""Every setting that affects results lives here so a run can be reproduced from its saved config."""
from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DAY = 86_400  # Yambda timestamps are in seconds (5-second granularity)


@dataclass
class Config:
    # Data source: pinned Hugging Face dataset revision so the raw file never changes underneath us.
    hf_repo: str = "yandex/yambda"
    hf_file: str = "flat/50m/listens.parquet"
    hf_revision: str = "dd6f3a19eef5866e346c3270e098baa641a44948"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    results_dir: Path = PROJECT_ROOT / "results"

    # Cleaning: a listen counts as positive feedback if more than this % of the track was played.
    min_played_ratio_pct: int = 50

    # Sampling: fraction of users kept (whole histories, chosen with a fixed seed). 1.0 = all users.
    user_fraction: float = 1.0
    seed: int = 42

    # Global temporal split (Yambda-style). Test = last `test_days`, validation = the window before it,
    # with a `gap_seconds` buffer before each held-out window so sessions don't straddle the cutoff.
    test_days: int = 7
    val_days: int = 7
    gap_seconds: int = 30 * 60

    # Catalog filter, computed on the training window only: songs need this many distinct listeners.
    min_item_users: int = 20

    # Models / evaluation
    knn_k_grid: list = field(default_factory=lambda: [20, 50, 100, 200])
    ks: list = field(default_factory=lambda: [10, 50])
    tune_metric: str = "ndcg@10"

    def to_dict(self):
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}
