"""Grouped splitting and split persistence."""
import numpy as np
import pandas as pd

from obforecast.validation import assign_folds, benchmark_split, check_grouped, fold_table


def test_assign_folds_keeps_windows_together_and_is_deterministic():
    tids = np.repeat(np.arange(100), 7)
    a, b = assign_folds(tids, 5, seed=1), assign_folds(tids, 5, seed=1)
    assert (a == b).all() and check_grouped(tids, a)
    assert sorted(pd.Series(a).groupby(tids).first().value_counts().tolist()) == [20] * 5
    # labels depend only on the set of windows, not on row order or duplication
    c = assign_folds(np.repeat(np.arange(100), 3)[::-1], 5, seed=1)
    assert dict(zip(np.repeat(np.arange(100), 3)[::-1], c)) == dict(zip(tids, a))


def test_inner_folds_are_disjoint_and_inside_outer_training(tables, cfg):
    feats = tables[0]
    tids = feats.time_id.to_numpy()
    outer = assign_folds(tids, 3, seed=1)
    for k in range(3):
        tr = outer != k
        inner = assign_folds(tids[tr], 2, seed=100 + k)
        assert np.intersect1d(tids[tr], tids[~tr]).size == 0
        assert np.intersect1d(tids[tr][inner == 0], tids[tr][inner == 1]).size == 0


def test_benchmark_split_recipe_persistence_and_mismatch(tmp_path):
    tids = np.arange(5, 5005, 3)
    path = tmp_path / "bench.csv"
    ids1, info1 = benchmark_split(tids, 0, 0.2, path)
    assert path.exists() and info1["source"] == "regenerated from recipe" and len(ids1) == round(0.2 * len(tids))
    ids2, info2 = benchmark_split(tids, 0, 0.2, path)
    assert (ids1 == ids2).all() and info2["matches_recipe"]
    pd.DataFrame({"time_id": ids1[:-1]}).to_csv(path, index=False)     # tampered file is used but flagged
    ids3, info3 = benchmark_split(tids, 0, 0.2, path)
    assert len(ids3) == len(ids1) - 1 and not info3["matches_recipe"]


def test_fold_table_matches_pipeline_assignment(tables, cfg):
    feats = tables[0].copy()
    feats["fold"] = assign_folds(feats.time_id.to_numpy(), cfg["split"]["outer_folds"], cfg["split"]["outer_fold_seed"])
    tab = fold_table(feats, cfg)
    assert tab.time_id.is_unique
    k = 1
    tr = feats[feats.fold != k]
    rowwise = assign_folds(tr.time_id.to_numpy(), cfg["split"]["inner_folds"], cfg["split"]["inner_seed_offset"] + k)
    m = dict(zip(tr.time_id, rowwise))
    t = tab[tab.fold != k]
    assert (t[f"inner_fold_in_outer_{k}"].to_numpy() == t.time_id.map(m).to_numpy()).all()
