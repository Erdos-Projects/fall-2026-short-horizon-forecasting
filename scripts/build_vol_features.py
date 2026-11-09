"""Build the realized-volatility feature tables for notebook 05 (cached, not committed).

    python scripts/build_vol_features.py --data-dir <kaggle folder>

Writes cache/vol_features_cut600.parquet (competition task) and cache/vol_features_cut300.parquet
(this project's five-minute task).
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from obforecast.data import ParquetSource, resolve_data_dir  # noqa: E402
from obforecast.volatility import build_stock  # noqa: E402

STARTS = {600: [0, 150, 300, 450], 300: [0, 100, 200]}


def one(args):
    data_dir, sid, cut = args
    src = ParquetSource(data_dir)
    return build_stock(sid, src.load("book", sid), src.load("trade", sid), cut, STARTS[cut])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir")
    ap.add_argument("--cuts", type=int, nargs="+", default=[600, 300])
    ap.add_argument("--stocks", type=int, default=None, help="first N stocks only (quick test)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    data_dir = resolve_data_dir(a.data_dir)
    sids = sorted(pd.read_csv(data_dir / "train.csv", usecols=["stock_id"]).stock_id.unique())
    if a.stocks:
        sids = sids[: a.stocks]
    out = ROOT / "cache"
    out.mkdir(exist_ok=True)
    for cut in a.cuts:
        t0 = perf_counter()
        with ProcessPoolExecutor(a.workers) as ex:
            parts = list(ex.map(one, [(str(data_dir), s, cut) for s in sids]))
        df = pd.concat(parts, ignore_index=True)
        if cut == 600:
            y = pd.read_csv(data_dir / "train.csv")
            df = df.merge(y, on=["stock_id", "time_id"], how="inner")
        suffix = "" if not a.stocks else f"_first{a.stocks}"
        df.to_parquet(out / f"vol_features_cut{cut}{suffix}.parquet", index=False)
        print(f"cut={cut}: {df.shape} in {perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
