"""Trading policy and quote-based simulation (merged notebook sections 5 and 6).

Long only, one share, at most one trade per stock and window. The decision uses only the
forecasts and the quote at the last feature second; fills come from the forward-filled book:
buy at the ask at ``entry + delay``, sell at the bid at ``exit``. Crossing the quotes already
pays the spread, so the spread is not subtracted again; a proportional cost ``c`` is charged
on each leg.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from .config import Timing


def net_return_bps(o: pd.DataFrame, cost_bps: float, timing: Timing = Timing(), delay: int = 0,
                   quantity: int = 1) -> np.ndarray:
    """10,000 * (Q*B*(1-c) - Q*A*(1+c)) / (Q*A)."""
    A = o[f"ask_{timing.entry_second + delay}"].to_numpy(dtype=float)
    B = o[f"bid_{timing.exit_second}"].to_numpy(dtype=float)
    Q, c = quantity, cost_bps / 1e4
    entry_cash, exit_cash = Q * A * (1 + c), Q * B * (1 - c)
    return 1e4 * (exit_cash - entry_cash) / (Q * A)


def fillable(o: pd.DataFrame, timing: Timing = Timing(), delay: int = 0, quantity: int = 1) -> np.ndarray:
    """Both legs executable for Q shares at the standing quote (otherwise: unresolved)."""
    a, b = f"ask_{timing.entry_second + delay}", f"bid_{timing.exit_second}"
    return ((o[f"ask_size_{timing.entry_second + delay}"] >= quantity)
            & (o[f"bid_size_{timing.exit_second}"] >= quantity)
            & o[a].notna() & o[b].notna()).to_numpy()


def decide(pred_price, pred_spread, q_299, margin_bps, tau, cost_bps):
    """Enter only on information available at the decision second: forecasts and the current quote.
    Hurdle = current relative spread + round-trip cost + margin, all in bps."""
    hurdle = np.asarray(q_299) * 1e4 + 2 * cost_bps + margin_bps
    return (np.asarray(pred_price) > hurdle) & (np.asarray(pred_spread) <= tau)


DECIDE_INPUTS = ["pred_price", "pred_spread", "q_299", "margin_bps", "tau", "cost_bps"]
assert list(inspect.signature(decide).parameters) == DECIDE_INPUTS


def rule_grid(policy: str, margins, taus) -> list[tuple]:
    """Simplest first: no spread filter, then smaller margins; ties keep the earlier rule."""
    ts = [float("inf")] if policy == "price only" else list(taus)
    return [(m, t) for t in sorted(ts, reverse=True) for m in margins]


def select_rule(policy, pred_price, pred_spread, q_299, net_ref, cost_bps, margins, taus, min_trades):
    """Pick (margin, tau) maximising mean net bps per opportunity on INNER out-of-fold forecasts.
    Returns ``(best, table)`` with ``best = (margin, tau, score, n_trades)`` or None."""
    best, rows = None, []
    for m, t in rule_grid(policy, margins, taus):
        enter = decide(pred_price, pred_spread, q_299, m, t, cost_bps)
        n = int(enter.sum())
        score = float(np.where(enter, net_ref, 0.0).mean())
        rows.append({"margin_bps": m, "tau": t, "inner_trades": n, "per_opportunity_bps": score})
        if n >= min_trades and (best is None or score > best[2] + 1e-12):
            best = (m, t, score, n)
    return best, pd.DataFrame(rows)


def per_opportunity(enter, o, cost_bps, timing: Timing = Timing(), delay: int = 0, quantity: int = 1):
    """Net bps for every opportunity: the trade's net return if entered and fillable, else 0."""
    can = fillable(o, timing, delay, quantity)
    return np.where(np.asarray(enter, bool) & can, net_return_bps(o, cost_bps, timing, delay, quantity), 0.0)


def policy_summary(enter, o, cost_bps, timing: Timing = Timing(), delay: int = 0, extra_mask=None,
                   quantity: int = 1) -> dict:
    net = net_return_bps(o, cost_bps, timing, delay, quantity)
    can = fillable(o, timing, delay, quantity)
    enter = np.asarray(enter, bool) & (True if extra_mask is None else extra_mask)
    done = enter & can
    per_opp = np.where(done, net, 0.0)
    tr = net[done]
    return {"opportunities": len(enter), "entries": int(enter.sum()), "completed": int(done.sum()),
            "unresolved": int((enter & ~can).sum()), "trade frequency": float(enter.mean()),
            "net bps per trade": float(tr.mean()) if len(tr) else np.nan,
            "net bps per opportunity": float(per_opp.mean()),
            "win rate": float((tr > 0).mean()) if len(tr) else np.nan,
            "5th pct trade bps": float(np.percentile(tr, 5)) if len(tr) else np.nan,
            "worst trade bps": float(tr.min()) if len(tr) else np.nan}


def paired_bootstrap(a, b, groups, n: int, seed: int):
    """Mean of (a - b) per opportunity with a 95% interval, resampling whole time_id groups."""
    d = pd.DataFrame({"d": np.asarray(a) - np.asarray(b), "g": groups}).groupby("g").d.agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    boot = d["sum"].to_numpy()[idx].sum(1) / d["count"].to_numpy()[idx].sum(1)
    return float((np.asarray(a) - np.asarray(b)).mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def window_t(net, done, groups) -> tuple[float, int]:
    """t-statistic of the mean trade return per traded window, and the number of traded windows."""
    w = pd.Series(np.asarray(net)[done]).groupby(np.asarray(groups)[done]).mean()
    if len(w) < 2:
        return float("nan"), int(len(w))
    return float(w.mean() / w.std() * np.sqrt(len(w))), int(len(w))
