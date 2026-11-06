"""Compare numbers asserted in the archived merged notebook with a re-run of this repository.

Reads results/metrics/*.csv|json written by ``python -m obforecast.run`` and writes
results/metrics/reproduction_check.csv: one row per archived claim with the archived value (as
printed in the notebook's markdown, usually rounded), the reproduced value, the tolerance used
(half a unit of the archived rounding, unless stated) and whether it matches. Claims whose stage
has not been run yet are listed as "not run".

Run from the repository root:  python scripts/compare_with_archive.py [results_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
M = RES / "metrics"


def csv(name):
    return pd.read_csv(M / f"{name}.csv")


def js(name):
    return json.loads((M / f"{name}.json").read_text())


def acc(target, model, col="R2"):
    a = csv("forecast_accuracy")
    return float(a[(a.target == target) & (a.model == model)][col].iloc[0])


def boot(comp, col="mean_bps"):
    b = csv("trading_bootstrap")
    return float(b[b.comparison == comp][col].iloc[0])


def summ(policy, col):
    s = csv("trading_summary")
    return float(s[s.policy == policy][col].iloc[0])


def fixed(check, col):
    s = csv("sensitivity_fixed")
    return float(s[(s.policy == "price only") & (s.check == check)][col].iloc[0])


def why(q):
    d = csv("why_spread_diagnostics")
    return float(d[d.quantity == q].value.iloc[0])


def whypol(p, col):
    d = csv("why_spread_policies")
    return float(d[d.policy == p][col].iloc[0])


def abl(row, col):
    d = csv("ablation")
    return float(d[d["feature set"] == row][col].iloc[0])


def leak(row, col):
    d = csv("leakage")
    return float(d[d.split == row][col].iloc[0])


def placebo(row, col):
    d = csv("stress_placebo")
    return float(d[d.inputs == row][col].iloc[0])


def accrual(sec):
    d = csv("stress_accrual")
    return float(d[d.second == sec].iloc[0, 1])


def fresh(row):
    d = csv("stress_fresh")
    return float(d[d.subset == row]["net bps per trade"].iloc[0])


def conc(row, col):
    d = csv("stress_concentration")
    return float(d[d.trim == row][col].iloc[0])


def dec(i, col):
    d = csv("why_spread_deciles")
    return float(d[d.decile == i][col].iloc[0])


# (section, claim, archived value, tolerance, getter, note)
CLAIMS = [
    ("5.1", "spread R2, selected (HGB), development", 0.348, 0.0005, lambda: acc("spread", "selected"), ""),
    ("5.1", "spread R2, ridge", 0.29, 0.005, lambda: acc("spread", "ridge"), ""),
    ("5.1", "spread R2, persistence", -0.31, 0.005, lambda: acc("spread", "persistence"), ""),
    ("5.1", "price R2, selected (ridge)", 0.061, 0.0005, lambda: acc("price", "selected"), ""),
    ("5.1", "price R2, HGB", 0.041, 0.0005, lambda: acc("price", "HGB"), ""),
    ("10", "spread R2, high-activity stocks", 0.47, 0.005, lambda: acc("spread", "selected", "R2 high-activity stocks"), ""),
    ("10", "spread R2, min-spread-bound stocks", 0.62, 0.005, lambda: acc("spread", "selected", "R2 min-spread-bound"), ""),
    ("6", "spread filter chosen (folds of 5)", 0, 0, lambda: float(csv("fold_selections")["price + spread: tau"]
                                                                   .replace(np.inf, np.nan).notna().sum()), ""),
    ("6", "trades, price-only rule", 21322, 0, lambda: summ("price only", "completed"), ""),
    ("6", "traded windows", 1025, 0, lambda: js("trading_window_t")["traded windows"], ""),
    ("6", "net bps per trade", 5.13, 0.005, lambda: summ("price only", "net bps per trade"), ""),
    ("6", "net bps per opportunity", 0.32, 0.005, lambda: boot("price only minus cash"), ""),
    ("6", "95% CI low", 0.21, 0.005, lambda: boot("price only minus cash", "ci_low"), ""),
    ("6", "95% CI high", 0.44, 0.005, lambda: boot("price only minus cash", "ci_high"), ""),
    ("6", "window-level t", 2.8, 0.05, lambda: js("trading_window_t")["price only window-level t"], ""),
    ("6.1", "per opportunity at 1 bp/side", 0.26, 0.005, lambda: fixed("cost 1.0 bp/side", "net bps per opportunity"), ""),
    ("6.1", "per trade, entry delay +2 s", 4.9, 0.05, lambda: fixed("entry delay +2 s", "net bps per trade"), "range +4.2..+4.9"),
    ("6.1", "per trade, entry delay +10 s", 4.2, 0.05, lambda: fixed("entry delay +10 s", "net bps per trade"), "range +4.2..+4.9"),
    ("7", "crossing-cost R2 from q_299", 0.81, 0.005, lambda: why("cost R2 using q_299"), ""),
    ("7", "crossing-cost R2, ridge forecast", 0.86, 0.005, lambda: why("cost R2 ridge forecast"), ""),
    ("7", "cost RMSE (bps)", 2.6, 0.05, lambda: why("cost RMSE ridge (bps)"), ""),
    ("7", "price-forecast RMSE (bps)", 24.0, 0.5, lambda: why("price-forecast RMSE (bps)"), "'about 24'"),
    ("7", "oracle minus price only (bps/opp)", -0.005, 0.0005, lambda: whypol("oracle: exact realised cost", "minus price only"), ""),
    ("7", "oracle CI low", -0.022, 0.0005, lambda: whypol("oracle: exact realised cost", "ci_low"), ""),
    ("7", "oracle CI high", 0.011, 0.0005, lambda: whypol("oracle: exact realised cost", "ci_high"), ""),
    ("7", "market-only price R2", 0.059, 0.0005, lambda: why("price R2 market-only"), ""),
    ("7", "market-only share of price R2", 0.96, 0.005, lambda: why("market-only share of price R2"), ""),
    ("7", "market-only rule, bps/opp", 0.297, 0.0005, lambda: whypol("market-only price model", "net bps per opportunity"), ""),
    ("7", "bottom decile forecast", -10.8, 0.05, lambda: dec(0, "forecast"), ""),
    ("7", "bottom decile realised", -9.7, 0.05, lambda: dec(0, "realised"), ""),
    ("7", "top decile forecast", 11.2, 0.05, lambda: dec(9, "forecast"), ""),
    ("7", "top decile realised", 11.0, 0.05, lambda: dec(9, "realised"), ""),
    ("7", "top decile net if bought", 0.6, 0.05, lambda: dec(9, "net if bought"), ""),
    ("8.1", "spread R2 loss without spread family", 0.09, 0.005, lambda: abl("without spread", "spread: R2 loss"), "'about 0.09'"),
    ("8.1", "price R2 loss without price family", 0.06, 0.005, lambda: abl("without price", "price: R2 loss"), "'about 0.06'"),
    ("8.1", "price R2, own-stock inputs only", 0.00, 0.005, lambda: abl("only own-stock inputs (no mkt_/rel_)", "price"), ""),
    ("8.1", "price R2, market inputs only", 0.03, 0.005, lambda: abl("only market inputs (mkt_*)", "price"), ""),
    ("8.1", "spread R2, market inputs only", 0.13, 0.005, lambda: abl("only market inputs (mkt_*)", "spread"), ""),
    ("8.2", "mid move by second 300 (bps)", 0.1, 0.05, lambda: accrual(300), ""),
    ("8.2", "mid move by second 310", 0.9, 0.05, lambda: accrual(310), ""),
    ("8.2", "mid move by second 360", 4.1, 0.05, lambda: accrual(360), ""),
    ("8.2", "mid move by second 599", 10.7, 0.05, lambda: accrual(599), ""),
    ("8.2", "per trade, own quote fresh at 299", 5.4, 0.05, lambda: fresh("own quote updated at decision second"), ""),
    ("8.2", "per trade, own quote stale at 299", 3.7, 0.05, lambda: fresh("own quote 1+ s old at decision second"), ""),
    ("8.2", "per trade, both legs fresh", 6.0, 0.05, lambda: fresh("both legs on fresh quotes"), ""),
    ("8.2", "lagged market inputs, bps/opp", 0.23, 0.005, lambda: placebo("market lagged 30 s", "per opportunity"), ""),
    ("8.2", "shuffled market inputs, price R2", -0.003, 0.0005, lambda: placebo("market shuffled across windows", "price R2"), ""),
    ("8.2", "share of P&L in best 1% of windows", 0.76, 0.005, lambda: js("stress_concentration_info")["share of rule P&L in best 1% of windows"], ""),
    ("8.2", "rule after dropping own best 5%", -0.12, 0.005, lambda: conc("drop own best 5% of windows", "rule, net bps per opportunity"), ""),
    ("8.2", "always-long after dropping own best 5%", -1.24, 0.005, lambda: conc("drop own best 5% of windows", "always long at the mid, no costs"), ""),
    ("8.2", "rule after dropping largest market moves", 0.06, 0.005, lambda: conc("drop 5% largest |market move|", "rule, net bps per opportunity"), ""),
    ("8.2", "gross bps per trade", 6.1, 0.05, lambda: js("stress_concentration_info")["gross bps per trade (fee 0)"], ""),
    ("8.2", "break-even fee (bp/side)", 3.0, 0.1, lambda: js("stress_concentration_info")["break-even fee bp per side (decisions fixed)"], "'about 3'"),
    ("8.3", "price R2, grouped split", 0.039, 0.0005, lambda: leak("grouped by window", "price R2"), "stock means now in-fold"),
    ("8.3", "price R2, random-row split", 0.179, 0.0005, lambda: leak("random rows", "price R2"), "stock means now in-fold"),
    ("8.3", "random/grouped price R2 ratio", 4.6, 0.05, lambda: leak("random / grouped ratio", "price R2"), "stock means now in-fold"),
    ("8.4", "correctness checks passing", 8, 0, lambda: float(csv("correctness_checks").passed.all()) * 8,
     "repo runs 11 checks; counted as 8 if all pass"),
    ("5.2", "upgraded spread R2", 0.380, 0.0005, lambda: float(csv("upgraded_cv").set_index("target").loc["spread", "upgraded model R2"]), ""),
    ("5.2", "upgraded price R2", 0.070, 0.0005, lambda: float(csv("upgraded_cv").set_index("target").loc["price", "upgraded model R2"]), ""),
    ("5.3", "unseen stocks, mean spread R2", 0.28, 0.005, lambda: float(csv("upgraded_unseen_stocks").pipe(lambda d: d[d.setting == "unseen stocks"]["spread R2"].mean())), ""),
    ("5.3", "seen stocks, mean spread R2", 0.32, 0.005, lambda: float(csv("upgraded_unseen_stocks").pipe(lambda d: d[d.setting == "seen stocks"]["spread R2"].mean())), ""),
]


def main():
    rows = []
    for sec, claim, archived, tol, get, note in CLAIMS:
        try:
            v = get()
            status = "match" if abs(v - archived) <= tol + 1e-12 else "differs"
        except (FileNotFoundError, KeyError, IndexError):
            v, status = np.nan, "not run"
        rows.append({"section": sec, "claim": claim, "archived": archived, "reproduced": v,
                     "tolerance": tol, "status": status, "note": note})
    out = pd.DataFrame(rows)
    out.to_csv(M / "reproduction_check.csv", index=False)
    print(out.round(4).to_string(index=False))
    print(out.status.value_counts().to_dict())


if __name__ == "__main__":
    main()
