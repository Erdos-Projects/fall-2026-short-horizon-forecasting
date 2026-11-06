"""Transaction-cost accounting and decision rules."""
import inspect

import numpy as np
import pandas as pd
import pytest

from obforecast.book import build_stock
from obforecast.checks import flat_round_trip, quote, synthetic_book
from obforecast.trading import (decide, fillable, net_return_bps, paired_bootstrap, per_opportunity,
                                policy_summary, rule_grid, select_rule, window_t)


def outcomes(asks, bids, ask_size=10, bid_size=10):
    return pd.DataFrame({"ask_300": asks, "bid_599": bids, "ask_size_300": ask_size, "bid_size_599": bid_size})


def test_flat_round_trip_loses_exactly_the_spread(timing):
    assert all(flat_round_trip(timing).values())


def test_net_return_formula_and_costs():
    o = outcomes([100.0], [101.0])
    assert net_return_bps(o, 0.0)[0] == pytest.approx(100.0)
    # 0.5 bp per leg: exit 101*(1-0.00005), entry 100*(1+0.00005)
    assert net_return_bps(o, 0.5)[0] == pytest.approx(1e4 * (101 * 0.99995 - 100 * 1.00005) / 100)
    nets = [net_return_bps(o, c)[0] for c in (0, 0.5, 1, 5)]
    assert all(np.diff(nets) < 0)


def test_spread_is_not_charged_twice(timing):
    """With a constant mid, the gross loss equals the quoted spread (no extra spread deduction)."""
    _, o, _, _ = build_stock(0, synthetic_book([quote(1, 0, 99.0, 101.0)]), timing)
    assert net_return_bps(o, 0.0, timing)[0] == pytest.approx(1e4 * (99 - 101) / 101)


def test_per_opportunity_accounting():
    o = outcomes([100.0, 100.0, 100.0, np.nan], [102.0, 98.0, 102.0, 102.0], ask_size=[10, 10, 0, 10])
    enter = np.array([True, True, True, True])
    po = per_opportunity(enter, o, 0.0)
    assert po.tolist() == [200.0, -200.0, 0.0, 0.0]            # unfillable/no-quote -> 0, not dropped
    s = policy_summary(enter, o, 0.0)
    assert s["entries"] == 4 and s["completed"] == 2 and s["unresolved"] == 2
    assert s["net bps per opportunity"] == pytest.approx(0.0) and s["win rate"] == 0.5
    assert per_opportunity(np.zeros(4, bool), o, 0.0).tolist() == [0.0] * 4


def test_price_scaling_invariance(synth, timing):
    source, stocks = synth
    bk = source.load("book", stocks[0])
    scaled = bk.copy()
    scaled[["bid_price1", "ask_price1", "bid_price2", "ask_price2"]] *= 37.5
    _, o_a, _, _ = build_stock(stocks[0], bk, timing)
    _, o_b, _, _ = build_stock(stocks[0], scaled, timing)
    np.testing.assert_allclose(net_return_bps(o_a, 0.5, timing), net_return_bps(o_b, 0.5, timing), equal_nan=True)


def test_decide_uses_only_decision_time_inputs():
    assert list(inspect.signature(decide).parameters) == ["pred_price", "pred_spread", "q_299", "margin_bps",
                                                          "tau", "cost_bps"]
    # hurdle = 1e4*q + 2c + margin = 5 + 1 + 2 = 8 bps
    e = decide(np.array([7.9, 8.1, 8.1]), np.array([0.0, 0.0, 0.3]), np.full(3, 5e-4), 2.0, 0.2, 0.5)
    assert e.tolist() == [False, True, False]


def test_rule_grid_prefers_simpler_rules_on_ties():
    assert rule_grid("price only", [0, 1], [0.0, np.inf]) == [(0, np.inf), (1, np.inf)]
    grid = rule_grid("price + spread", [0, 1], [np.log(0.9), np.inf])
    assert grid[0] == (0, np.inf) and grid[-1][1] == np.log(0.9)
    pp, ps, q = np.full(300, 10.0), np.zeros(300), np.zeros(300)
    best, tab = select_rule("price + spread", pp, ps, q, np.ones(300), 0.0, [0, 1], [np.log(0.9), np.inf], 10)
    assert best[:2] == (0, np.inf) and len(tab) == 4             # all tie -> no filter, smallest margin


def test_select_rule_respects_min_trades():
    best, _ = select_rule("price only", np.array([10.0, -1]), np.zeros(2), np.zeros(2), np.ones(2), 0.0, [0], [np.inf], 5)
    assert best is None


def test_paired_bootstrap_and_window_t():
    g = np.repeat(np.arange(50), 4)
    m, lo, hi = paired_bootstrap(np.ones(200), np.ones(200), g, 200, 0)
    assert m == lo == hi == 0.0
    rng = np.random.default_rng(0)
    a = rng.normal(1, 1, 200)
    m, lo, hi = paired_bootstrap(a, np.zeros(200), g, 500, 0)
    assert lo < m < hi
    t, n = window_t(a, np.ones(200, bool), g)
    assert n == 50 and t > 0


def test_fillable_requires_size_and_quotes():
    o = outcomes([1.0, 1.0, np.nan], [1.0, 1.0, 1.0], ask_size=[1, 0, 1])
    assert fillable(o).tolist() == [True, False, False]
