"""Feature counts and in-fold learning of every data-dependent statistic."""
import numpy as np
import pandas as pd

from obforecast.book import KEYS
from obforecast.extra import extra_stock
from obforecast.features import (BASE, FAMILIES, FEATURES, family, feature_inventory, fold_features,
                                 leave_one_out_mean, upgraded_feature_names, upgraded_features)
from obforecast.validation import assign_folds


def _split(feats, k=0):
    folds = assign_folds(feats.time_id.to_numpy(), 3, seed=1)
    return feats[folds != k].reset_index(drop=True), feats[folds == k].reset_index(drop=True)


def test_baseline_has_47_inputs_in_7_families():
    assert len(BASE) == 30 and len(FEATURES) == 47 and len(set(FEATURES)) == 47
    counts = pd.Series([family(c) for c in FEATURES]).value_counts().to_dict()
    assert counts == {"spread": 14, "price": 9, "volatility": 6, "activity": 6, "depth": 5,
                      "imbalance": 4, "min_spread": 3}       # matches the saved output in the merged notebook
    assert set(FAMILIES) == set(counts)


def test_leave_one_out_mean_excludes_focal_stock():
    d = pd.DataFrame({"time_id": [1, 1, 1, 2, 2], "x": [1.0, 2.0, np.nan, 5.0, 7.0]})
    out = leave_one_out_mean(d, "x").tolist()
    assert out[:2] == [2.0, 1.0] and out[2] == 1.5 and out[3:] == [7.0, 5.0]
    single = pd.DataFrame({"time_id": [1], "x": [3.0]})
    assert np.isnan(leave_one_out_mean(single, "x").iloc[0])


def test_fold_features_learn_stock_means_from_training_only(tables):
    feats = tables[0]
    tr, te = _split(feats)
    X_tr, X_te = fold_features(tr, te)
    te_mod = te.copy()
    te_mod["log_S1"] += 5.0                         # changing evaluation rows...
    X_tr2, X_te2 = fold_features(tr, te_mod)
    pd.testing.assert_frame_equal(X_tr, X_tr2)      # ...never changes training inputs
    mean_tr = tr.groupby("stock_id").log_S1.mean()
    expected = te_mod.log_S1 - te_mod.stock_id.map(mean_tr)
    np.testing.assert_allclose(X_te2.log_S1_vs_stock, expected)


def test_market_features_use_only_same_window_rows(tables):
    feats = tables[0]
    tr, te = _split(feats)
    _, X_te = fold_features(tr, te)
    w = te.time_id.iloc[0]
    sub = te[te.time_id == w]
    i = sub.index[0]
    np.testing.assert_allclose(X_te.loc[i, "mkt_ret_0_299"], sub.ret_0_299.drop(i).mean())


def test_upgraded_has_131_inputs_with_31_extra(synth, tables, timing):
    source, stocks = synth
    feats, outs = tables[0], tables[1]
    extra = pd.concat([extra_stock(s, source.load("book", s), source.load("trade", s), timing) for s in stocks])
    fx = feats.merge(extra, on=KEYS, how="left")
    folds = assign_folds(fx.time_id.to_numpy(), 3, seed=1)
    tr, te = fx[folds != 0].reset_index(drop=True), fx[folds == 0].reset_index(drop=True)
    y_tr = {"spread": outs.y_spread.to_numpy()[folds != 0], "price": outs.y_price.to_numpy()[folds != 0]}
    Xa, Xb = upgraded_features(tr, te, y_tr)
    assert Xa.shape[1] == 131 and list(Xa.columns) == upgraded_feature_names() == list(Xb.columns)
    inv = feature_inventory()
    assert len(inv) == 131 and inv.baseline_47.sum() == 47
    assert (inv.kind == "raw first-half (extra)").sum() == 31
    # stock target statistics come from the training targets only
    y_mod = {k: v.copy() for k, v in y_tr.items()}
    Xa2, Xb2 = upgraded_features(tr, te.assign(log_S1=te.log_S1 + 1), y_mod)
    pd.testing.assert_series_equal(Xb.te_spread, Xb2.te_spread)
    expected = te.stock_id.map(pd.Series(y_tr["spread"]).groupby(tr.stock_id.to_numpy()).mean()).to_numpy()
    np.testing.assert_allclose(Xb.te_spread.to_numpy(), expected)
