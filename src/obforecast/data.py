"""Data access for the Kaggle "Optiver Realized Volatility Prediction" files.

Expected layout under the data directory (the competition download, unchanged)::

    train.csv
    book_train.parquet/stock_id=<id>/<part>.parquet
    trade_train.parquet/stock_id=<id>/<part>.parquet

Nothing here depends on Google Drive. ``resolve_data_dir`` picks the directory from an
explicit argument, the ``OBF_DATA_DIR`` environment variable, or the config.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

KAGGLE_COMPETITION = "optiver-realized-volatility-prediction"
ENV_VAR = "OBF_DATA_DIR"


def resolve_data_dir(explicit: str | Path | None = None, cfg: dict | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    if os.environ.get(ENV_VAR):
        return Path(os.environ[ENV_VAR]).expanduser()
    if cfg and cfg.get("data", {}).get("dir"):
        return Path(cfg["data"]["dir"]).expanduser()
    return Path("data") / KAGGLE_COMPETITION


def stock_ids_in(folder: Path) -> set[int]:
    """Stock ids that have a ``stock_id=*`` partition folder."""
    return {int(p.name.split("=")[1]) for p in Path(folder).glob("stock_id=*") if p.is_dir()}


def check_layout(data_dir: Path) -> dict:
    """Describe what is present. Raises FileNotFoundError with a helpful message if unusable."""
    data_dir = Path(data_dir)
    if not (data_dir / "train.csv").is_file():
        raise FileNotFoundError(
            f"{data_dir / 'train.csv'} not found. Download the Kaggle competition "
            f"'{KAGGLE_COMPETITION}' (see README, 'Reproducing the analysis') and point "
            f"--data-dir or {ENV_VAR} at the extracted folder.")
    labels = pd.read_csv(data_dir / "train.csv", usecols=["stock_id", "time_id"])
    train_ids = set(labels.stock_id.unique())
    book_ids = stock_ids_in(data_dir / "book_train.parquet")
    trade_ids = stock_ids_in(data_dir / "trade_train.parquet")

    def has_data(kind, sid):
        part = data_dir / f"{kind}_train.parquet" / f"stock_id={sid}"
        return any(p.stat().st_size > 0 for p in part.rglob("*.parquet"))

    missing = {k: sorted(int(s) for s in train_ids if not has_data(k, s)) for k in ("book", "trade")}
    return {"train_rows": len(labels), "train_stocks": len(train_ids),
            "train_time_ids": labels.time_id.nunique(),
            "book_folders": len(book_ids), "trade_folders": len(trade_ids),
            "book_matches_train": book_ids == train_ids, "trade_matches_train": trade_ids == train_ids,
            "missing": missing,
            "book_stocks": sorted(int(s) for s in train_ids if s not in missing["book"])}


class ParquetSource:
    """Reads one stock's book or trade partition from the Kaggle layout."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def load(self, kind: str, stock_id: int) -> pd.DataFrame:
        folder = self.data_dir / f"{kind}_train.parquet" / f"stock_id={stock_id}"
        paths = sorted(folder.rglob("*.parquet"))
        if not paths:
            raise ValueError(f"No {kind} data found for stock_id={stock_id} under {self.data_dir}.")
        return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True).sort_values(
            ["time_id", "seconds_in_bucket"]).reset_index(drop=True)


class InMemorySource:
    """Same interface as ParquetSource for synthetic books (tests, demos)."""

    def __init__(self, books: dict[int, pd.DataFrame], trades: dict[int, pd.DataFrame] | None = None):
        self.books, self.trades = books, trades or {}

    def load(self, kind: str, stock_id: int) -> pd.DataFrame:
        store = self.books if kind == "book" else self.trades
        if stock_id not in store:
            raise ValueError(f"No {kind} data for stock_id={stock_id}.")
        return store[stock_id].sort_values(["time_id", "seconds_in_bucket"]).reset_index(drop=True)


def select_stocks(available: list[int], spec) -> list[int]:
    """Config ``run.stocks``: "all", an integer N (first N ids), or an explicit list."""
    available = sorted(available)
    if spec in (None, "all"):
        return available
    if isinstance(spec, int):
        return available[:spec]
    missing = sorted(set(spec) - set(available))
    if missing:
        raise ValueError(f"requested stocks without book data: {missing}")
    return sorted(int(s) for s in spec)
