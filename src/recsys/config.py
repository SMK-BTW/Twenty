"""settings for the project"""
from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DAY = 86_400  # seconds in a day


@dataclass
class Config:
    # data source
    hf_repo: str = "yandex/yambda"
    hf_file: str = "flat/50m/listens.parquet"
    hf_revision: str = "dd6f3a19eef5866e346c3270e098baa641a44948"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    results_dir: Path = PROJECT_ROOT / "results"

    # min percent of song played to count
    min_played_ratio_pct: int = 50

    # fraction of users to use
    user_fraction: float = 1.0
    seed: int = 42

    # split settings
    test_days: int = 7
    val_days: int = 7
    gap_seconds: int = 30 * 60

    # min listeners per song
    min_item_users: int = 20

    # model and eval settings
    knn_k_grid: list = field(default_factory=lambda: [20, 50, 100, 200])
    ks: list = field(default_factory=lambda: [10, 50])
    tune_metric: str = "ndcg@10"

    def to_dict(self):
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}
