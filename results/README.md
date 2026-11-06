# Results

Everything in this folder is generated; nothing is typed in by hand.

| Path | Produced by |
|---|---|
| `metrics/*.csv`, `metrics/*.json` (forecast accuracy, fold selections, trading, sensitivity, cost/oracle analysis, ablation, stress tests, leakage, correctness checks) | `python -m obforecast.run --config configs/default.yaml --skip-upgraded`, full data, 2026-11-06 |
| `figures/figure1_accuracy_vs_value.png`, `figures/calibration.png` | same run |
| `splits/benchmark_time_ids.csv`, `splits/development_folds.csv` | same run; the 766 benchmark windows are identical to the team's saved `test_time_ids.csv` |
| `run_manifest.json`, `run_full.log` | same run: configuration and hash, package versions, per-stage runtime and status |
| `metrics/upgraded_cv.csv`, `metrics/upgraded_unseen_stocks.csv`, `metrics/upgraded_drivers.csv`, `run_manifest_upgraded.json`, `run_upgraded.log` | `python -m obforecast.run --config configs/default.yaml --stages cv,upgraded`, full data, 2026-11-06 (3.4 h of LightGBM) |
| `metrics/reproduction_check.csv` | `scripts/compare_with_archive.py`: archived notebook values vs this run |
| `metrics/source_output_audit.csv`, `metrics/headline_claims.csv`, `metrics/historical_saved_forecast_r2.csv`, `metrics/feature_inventory.csv`, `figures/study_design.png`, `figures/historical_saved_forecast_r2.png` | `scripts/audit_source_notebooks.py` (what the archived notebooks support on their own) |

The manifest records commit `6512fe0`; the run also had the look-ahead-check fix from commit
`326314a` applied, which changes no analysis code. Benchmark results will be added once it is scored. Smoke runs write to `results/smoke/` and synthetic test runs
to `results/synthetic/`; both are labelled and not committed.
