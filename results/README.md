# Results

Everything in this folder is generated. Nothing is typed in by hand.

| Path | Produced by | Status in this commit |
|---|---|---|
| `metrics/source_output_audit.csv` | `scripts/audit_source_notebooks.py` | Present. Saved output / saved error of every code cell in `archive/` |
| `metrics/headline_claims.csv` | same | Present. Each published claim → source cell → verification status |
| `metrics/historical_saved_forecast_r2.csv` | same (parsed from the merged notebook's saved §5.1 table) | Present. **Historical**, not re-executed |
| `metrics/feature_inventory.csv` | same (from `src/obforecast/features.py`) | Present. 47 baseline / 131 upgraded inputs |
| `figures/study_design.png`, `figures/historical_saved_forecast_r2.png` | same | Present |
| `metrics/*.csv` and `*.json` for §5–§9, `figures/figure1_accuracy_vs_value.png`, `figures/calibration.png`, `splits/`, `run_manifest.json` | `python -m obforecast.run --config configs/default.yaml` | **Not yet generated**: needs the Kaggle data (see the top-level README) |

Smoke runs write to `results/smoke/` and synthetic test runs to `results/synthetic/`. Each is stamped
with its label, and neither is committed.
