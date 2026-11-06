"""Static, publication-style figures. Every figure is drawn from data passed in (no typed-in
numbers). Palette: categorical slots 1-2 of the validated reference palette."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Timing  # noqa: E402

SPREAD, PRICE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200})


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return Path(path)


def design_figure(path, timing: Timing = Timing(), outer_folds: int = 5, inner_folds: int = 3):
    """Study design: window timing and the nested, window-grouped validation. Data-free."""
    fig, (a, b) = plt.subplots(2, 1, figsize=(9.5, 4.6), gridspec_kw={"height_ratios": [1.1, 1]})
    cut, ent, ex = timing.cut, timing.entry_second, timing.exit_second
    a.barh(0, cut, left=0, height=0.5, color=SPREAD, alpha=0.18, edgecolor=SPREAD)
    a.barh(0, 600 - cut, left=cut, height=0.5, color=PRICE, alpha=0.14, edgecolor=PRICE)
    a.text(cut / 2, 0, f"features: order book seconds 0–{cut - 1}\n(+ trades, upgraded models)",
           ha="center", va="center", color=INK)
    a.text(cut + (600 - cut) / 2, 0.02, "outcomes (never model inputs)\nprice: mid return 299→599 (bps)\n"
           "spread: log(mean rel. spread 300–599 / 0–299)", ha="center", va="center", color=INK, fontsize=9)
    for x, lab, ha in [(cut - 1, f"decide after {cut - 1} ", "right"), (ent, f" buy at ask ({ent})", "left"),
                       (ex, f"sell at bid ({ex})", "right")]:
        a.axvline(x, color=INK2, lw=1, ls=":")
        a.text(x, 0.42, lab, ha=ha, va="bottom", fontsize=8.5, color=INK2)
    a.set(xlim=(0, 600), ylim=(-0.4, 0.75), yticks=[], xlabel="second within the 10-minute window")
    a.set_title("Each sample: one stock × one 10-minute window", loc="left", color=INK, fontsize=11)
    a.spines["left"].set_visible(False)

    b.set_title(f"Validation: whole windows stay together; {outer_folds} outer × {inner_folds} inner grouped folds",
                loc="left", color=INK, fontsize=11)
    widths = [0.8 / outer_folds] * outer_folds
    for i in range(outer_folds):
        b.barh(1, widths[i] - 0.004, left=i * widths[i], height=0.5,
               color=PRICE if i == 0 else SPREAD, alpha=0.75 if i == 0 else 0.35)
    b.barh(1, 0.196, left=0.804, height=0.5, color=INK2, alpha=0.35)
    b.text(0.4, 1.42, "development windows (80%)", ha="center", fontsize=9, color=INK)
    b.text(0.902, 1.42, "legacy benchmark (20%)\npreviously examined", ha="center", fontsize=8.5, color=INK)
    b.text(widths[0] / 2, 1, "score", ha="center", va="center", fontsize=8.5, color="white")
    for j in range(inner_folds):
        w = (0.8 - widths[0]) / inner_folds
        b.barh(0.3, w - 0.004, left=widths[0] + j * w, height=0.35, color=SPREAD, alpha=0.2 + 0.2 * j)
    b.text(widths[0] + (0.8 - widths[0]) / 2, 0.3, "inner folds choose model head and trading rule",
           ha="center", va="center", fontsize=8.5, color=INK)
    b.set(xlim=(0, 1), ylim=(-0.05, 1.75), xticks=[], yticks=[])
    for s_ in ["left", "bottom"]:
        b.spines[s_].set_visible(False)
    fig.tight_layout()
    return _save(fig, path)


def figure1(skill: pd.DataFrame, value: pd.DataFrame, path, title_note: str = ""):
    """Forecast accuracy vs trading value.

    ``skill``: columns model, spread, price (out-of-fold R^2).
    ``value``: columns cost_bps, mean_bps, ci_low, ci_high (price-only rule, per opportunity).
    """
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(len(skill))
    for i, (t, color) in enumerate([("spread", SPREAD), ("price", PRICE)]):
        vals = skill[t].to_numpy()
        bars = a.bar(x + (i - 0.5) * 0.38, np.clip(vals, -0.05, None), 0.36, color=color, label=f"{t} change")
        a.bar_label(bars, labels=[f"{v:.2f}" if v > -0.05 else f"{v:.2f}↓" for v in vals], fontsize=8, padding=2,
                    color=INK2)
    a.axhline(0, color=INK2, lw=0.8)
    a.set(xticks=x, xticklabels=skill.model, ylabel="out-of-fold R²", ylim=(-0.11, max(0.45, skill[["spread", "price"]].max().max() + 0.05)))
    a.set_title("Forecast accuracy by model and target", loc="left", color=INK)
    a.legend(frameon=False)
    a.grid(axis="y", color=GRID, lw=0.8)
    a.set_axisbelow(True)
    xs = np.arange(len(value))
    m = value.mean_bps.to_numpy()
    b.errorbar(xs, m, yerr=[m - value.ci_low.to_numpy(), value.ci_high.to_numpy() - m], fmt="o", ms=8,
               capsize=4, color=PRICE, label="price-only rule (95% window-clustered CI)")
    b.axhline(0, color=INK2, lw=0.8, label="cash (no trading)")
    b.set(xticks=xs, xticklabels=[f"{c:g} bp/side" for c in value.cost_bps], ylabel="net bps per opportunity",
          xlim=(-0.5, len(value) - 0.5))
    b.set_title("Simulated trading value of the price-only rule", loc="left", color=INK)
    b.legend(frameon=False, fontsize=8.5)
    b.grid(axis="y", color=GRID, lw=0.8)
    b.set_axisbelow(True)
    if title_note:
        fig.suptitle(title_note, x=0.01, ha="left", fontsize=9, color=INK2, y=1.02)
    fig.tight_layout()
    return _save(fig, path)


def calibration_figure(cal: pd.DataFrame, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (t, color) in zip(axes, [("price", PRICE), ("spread", SPREAD)]):
        c = cal[cal.target == t]
        ax.errorbar(c.pred, c.actual, yerr=1.96 * c.se, fmt="o", ms=7, capsize=3, color=color, label="decile mean ± 95% CI")
        lim = [min(c.pred.min(), c.actual.min()), max(c.pred.max(), c.actual.max())]
        ax.plot(lim, lim, color=INK2, ls="--", lw=0.8, label="perfect calibration")
        unit = "bps" if t == "price" else "log ratio"
        ax.set(xlabel=f"mean forecast ({unit})", ylabel=f"mean outcome ({unit})")
        ax.set_title(f"{t}: out-of-fold calibration, selected head", loc="left", color=INK)
        ax.grid(color=GRID, lw=0.8)
        ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    return _save(fig, path)


def historical_skill_figure(hist: pd.DataFrame, path):
    """R^2 values parsed from SAVED outputs of the archived notebook (not re-executed)."""
    piv = hist.pivot(index="model", columns="target", values="R2").reindex(
        ["zero", "stock mean", "persistence", "ridge", "HGB"])
    fig, a = plt.subplots(figsize=(7.5, 4.1))
    x = np.arange(len(piv))
    for i, (t, color) in enumerate([("spread", SPREAD), ("price", PRICE)]):
        vals = piv[t].to_numpy()
        bars = a.bar(x + (i - 0.5) * 0.38, np.clip(vals, -0.05, None), 0.36, color=color, label=f"{t} change")
        a.bar_label(bars, labels=[f"{v:.3f}" if v > -0.05 else f"{v:.2f}↓" for v in vals], fontsize=8, padding=2, color=INK2)
    a.axhline(0, color=INK2, lw=0.8)
    a.set(xticks=x, xticklabels=piv.index, ylabel="out-of-fold R² (development, nested grouped CV)", ylim=(-0.11, 0.42))
    a.set_title("Historical saved output (merged notebook §5.1, executed 2026-11-05 03:48)\n"
                "Not re-executed in this repository; ↓ = value below −0.05, bar clipped", loc="left", color=INK, fontsize=9.5)
    a.legend(frameon=False)
    a.grid(axis="y", color=GRID, lw=0.8)
    a.set_axisbelow(True)
    fig.tight_layout()
    return _save(fig, path)
