"""Grouped splitting: the legacy benchmark split and nested grouped folds.

Every row of a ``time_id`` (all stocks in one 10-minute window) always lands in the same fold.
``time_id`` is shuffled in the source data, so it carries no time order and is never a feature.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def assign_folds(time_ids, k: int, seed: int) -> np.ndarray:
    """Deterministic grouped folds: every row of a time_id gets the same fold."""
    u = np.random.default_rng(seed).permutation(np.unique(time_ids))
    return pd.Series(time_ids).map(dict(zip(u, np.arange(len(u)) % k))).to_numpy()


def benchmark_split(all_time_ids, seed: int, frac: float, saved_path: Path | None = None):
    """Legacy benchmark ("holdout") time_ids.

    The recipe is ``default_rng(seed).choice(sorted unique time_ids, round(frac * n))``. If a
    saved file exists it is used (and compared with the recipe); otherwise the recipe is used and
    the result is written to ``saved_path`` so later runs and other numpy versions reuse it.
    Returns ``(benchmark_ids, info)``.
    """
    all_tids = np.sort(np.unique(all_time_ids))
    regenerated = np.random.default_rng(seed).choice(all_tids, size=round(frac * len(all_tids)), replace=False)
    info = {"recipe": f"numpy default_rng({seed}).choice(sorted time_ids, {frac:.0%})",
            "n_windows": int(len(all_tids)), "n_benchmark": int(len(regenerated))}
    if saved_path is not None and Path(saved_path).exists():
        saved = pd.read_csv(saved_path).time_id.to_numpy()
        info.update(source=str(saved_path), matches_recipe=bool(set(saved) == set(regenerated)))
        return np.sort(saved), info
    info.update(source="regenerated from recipe", matches_recipe=True)
    if saved_path is not None:
        Path(saved_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"time_id": np.sort(regenerated)}).to_csv(saved_path, index=False)
        info["written_to"] = str(saved_path)
    return np.sort(regenerated), info


def check_grouped(time_ids, folds) -> bool:
    return bool(pd.Series(folds).groupby(np.asarray(time_ids)).nunique().max() == 1)


def inner_folds_for(dev_time_ids, outer_folds, k: int, inner_k: int, seed_offset: int) -> np.ndarray:
    """Inner fold labels for the training part of outer fold k (NaN on the outer validation part)."""
    tr = outer_folds != k
    out = np.full(len(outer_folds), np.nan)
    out[tr] = assign_folds(np.asarray(dev_time_ids)[tr], inner_k, seed=seed_offset + k)
    return out


def fold_table(dev: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Persistable split identifiers: one row per development time_id with its outer fold and,
    for each outer fold k, its inner fold inside outer-training fold k."""
    s = cfg["split"]
    w = dev.drop_duplicates("time_id")[["time_id", "fold"]].sort_values("time_id").reset_index(drop=True)
    for k in range(s["outer_folds"]):
        tr = w.fold != k
        col = np.full(len(w), -1)
        col[tr.to_numpy()] = assign_folds(w.time_id[tr].to_numpy(), s["inner_folds"], s["inner_seed_offset"] + k)
        w[f"inner_fold_in_outer_{k}"] = col
    return w
