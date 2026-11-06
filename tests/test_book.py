"""Data boundaries of the one-second book and of every feature set."""
import numpy as np
import pandas as pd
import pytest

from obforecast.book import BOOK_COLS, build_stock, dense_grid
from obforecast.checks import forward_fill_boundaries, perturb_future, quote, synthetic_book
from obforecast.config import Timing
from obforecast.extra import XCOLS, extra_stock, path_points


def test_forward_fill_never_backfills_or_crosses_windows():
    res = forward_fill_boundaries()
    assert all(res.values()), res


def test_invalid_snapshots_are_dropped_not_carried():
    b = synthetic_book([quote(1, 0, 1.0, 1.001), quote(1, 5, 1.002, 1.001),   # crossed at 5
                        quote(1, 8, 1.0, 1.002, size=0)])                    # zero size at 8
    _, g, reason = dense_grid(b)
    assert list(reason) == ["", "crossed or locked", "nonpositive size"]
    assert (g["bid_price1"][0, :20] == 1.0).all(), "the valid quote at 0 must stand"
    assert g["age"][0, 10] == 10


def test_quote_age_counts_from_last_valid_update():
    b = synthetic_book([quote(1, 3, 1.0, 1.001), quote(1, 50, 1.0, 1.002)])
    _, g, _ = dense_grid(b)
    assert np.isnan(g["age"][0, 2]) and g["age"][0, 3] == 0 and g["age"][0, 49] == 46 and g["age"][0, 50] == 0


def test_inconsistent_level2_is_missing_only_for_level2():
    row = quote(1, 0, 1.0, 1.001)
    row[4] = 1.0005                                  # bid_price2 above bid_price1
    _, g, _ = dense_grid(synthetic_book([row]))
    assert np.isnan(g["bid_price2"]).all() and (g["bid_price1"] == 1.0).all()


@pytest.mark.parametrize("stock_pos", [0, 3, 7])
def test_future_perturbation_leaves_all_features_unchanged(synth, timing, stock_pos):
    source, stocks = synth
    sid = stocks[stock_pos]
    book, trades = source.load("book", sid), source.load("trade", sid)
    bk2, tr2 = perturb_future(book, trades, timing)
    f_a, o_a, _, _ = build_stock(sid, book, timing)
    f_b, o_b, _, _ = build_stock(sid, bk2, timing)
    pd.testing.assert_frame_equal(f_a, f_b)
    assert not o_a.y_price.equals(o_b.y_price)
    x_a, x_b = extra_stock(sid, book, trades, timing), extra_stock(sid, bk2, tr2, timing)
    pd.testing.assert_frame_equal(x_a, x_b)


def test_perturbation_test_is_sensitive(synth, timing):
    """Guard against a vacuous test: perturbing the LAST feature second must change features."""
    source, stocks = synth
    sid = stocks[0]
    book, trades = source.load("book", sid), source.load("trade", sid)
    shifted = Timing(feature_last_second=298, decision_second=298, entry_second=300, exit_second=599)
    bk2, tr2 = perturb_future(book, trades, shifted)          # changes second 299 onward
    assert not build_stock(sid, book, timing)[0].equals(build_stock(sid, bk2, timing)[0])
    assert not extra_stock(sid, book, trades, timing).equals(extra_stock(sid, bk2, tr2, timing))


def test_extra_features_have_31_columns(synth, timing):
    source, stocks = synth
    x = extra_stock(stocks[0], source.load("book", stocks[0]), source.load("trade", stocks[0]), timing)
    assert len(XCOLS) == 31 and list(x.columns) == ["stock_id", "time_id", *XCOLS]


def test_outcomes_are_not_features(tables):
    feats, outs, _, _ = tables
    outcome_only = set(outs.columns) - {"stock_id", "time_id"}
    assert not outcome_only & set(feats.columns)


def test_targets_match_definitions(timing):
    """y_price = (m_599 / m_299 - 1) * 1e4 ; y_spread = log(mean q[300:] / mean q[:300])."""
    rows = [quote(1, 0, 0.999, 1.001), quote(1, 299, 1.009, 1.011), quote(1, 450, 1.0185, 1.0215)]
    f, o, _, _ = build_stock(0, synthetic_book(rows), timing)
    m299, m599 = 1.010, 1.020
    assert o.y_price.iloc[0] == pytest.approx((m599 / m299 - 1) * 1e4)
    q = np.r_[np.full(299, 0.002 / 1.0), [0.002 / 1.010], np.full(150, 0.002 / 1.010), np.full(150, 0.003 / 1.020)]
    assert o.y_spread.iloc[0] == pytest.approx(np.log(q[300:].mean() / q[:300].mean()))
    assert f.q_299.iloc[0] == pytest.approx(0.002 / 1.010)


def test_window_without_initial_state_is_excluded(timing):
    rows = [quote(1, 0, 1.0, 1.001), quote(2, 350, 1.0, 1.001)]
    f, o, q, _ = build_stock(0, synthetic_book(rows), timing)
    assert list(f.time_id) == [1] and list(o.time_id) == [1] and q.no_initial_state.tolist() == [False, True]


def test_path_points_columns(synth, timing):
    source, stocks = synth
    pts = path_points(stocks[0], source.load("book", stocks[0]), [209, 269, 310], timing)
    assert {"mid_209", "mid_269", "mid_299", "mid_310", "age_exit_quote", "mid_first"} <= set(pts.columns)


def test_book_cols_order_matches_kaggle_schema():
    assert BOOK_COLS == ["bid_price1", "ask_price1", "bid_price2", "ask_price2",
                         "bid_size1", "ask_size1", "bid_size2", "ask_size2"]
