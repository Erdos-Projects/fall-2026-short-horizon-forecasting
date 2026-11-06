# Order-Book Forecasting: Price, Liquidity, and Trading Decisions

*Erdős Institute · Fall 2026*

Forecasting short-horizon market behavior requires understanding both price movements and the liquidity available to trade them. This project studies the two together: **can five minutes of order-book and trade data predict the next five minutes of mid-price return and average bid-ask spread, and how should those predictions inform trading decisions?**

Using **428,932 stock-windows across 112 stocks**, we built a research pipeline spanning quote reconstruction, market microstructure features, predictive modeling, nested validation, and transaction-cost-aware simulation.

The team's saved development results show a clear difference between the two forecasting tasks: **spread-change R² of 0.348** and **price-return R² of 0.061**, evaluated with window-grouped, nested cross-validation. The analysis also separates forecast accuracy from decision value by testing whether an additional liquidity forecast improves a price-based trading rule.

## Project highlights

- **Market microstructure data pipeline:** reconstructs one-second order books using only previously observed quotes, with explicit handling of invalid snapshots and quote age.
- **Price and liquidity forecasting:** compares simple baselines, regularized linear models, and gradient boosting for two distinct market outcomes.
- **Cross-stock context:** combines stock-level signals with leave-one-out market features and training-fold stock normalization.
- **Nested model and policy selection:** keeps complete market windows together and selects models and trading thresholds within inner folds.
- **Trading and robustness framework:** evaluates bid/ask execution, additional transaction costs, delayed entry, feature ablations, and sensitivity to market context.

![Study design](results/figures/study_design.png)

*Each sample uses seconds 0–299 as inputs. The simulation enters at the ask at second 300 and exits at the bid at second 599. Model and policy selection take place within nested, window-grouped folds.*

## Forecasting results

The table below summarizes the saved development outputs from the team's merged notebook. These are historical notebook results; a full-data rerun of the packaged pipeline remains pending.

| Forecast target | Selected model | Out-of-fold R² | Reference models |
|---|---|---:|---|
| Change in average spread, `log(S2/S1)` | Histogram gradient boosting | **0.348** | Ridge: 0.291; stock mean: 0.003; persistence: −0.313 |
| Five-minute mid-price return | Ridge regression | **0.061** | Histogram gradient boosting: 0.041; zero-return baseline: approximately 0 |

Both tasks use **47 inputs** and **5 outer × 3 inner folds**, grouped by market window. Histogram gradient boosting was selected for spread forecasting in every outer fold, while ridge was selected for price forecasting.

![Forecast accuracy from saved notebook outputs](results/figures/historical_saved_forecast_r2.png)

*Historical out-of-fold performance on development windows, extracted from the archived notebook's saved §5.1 outputs.*

### From forecasts to decisions

The project evaluates a price-based trading rule alongside a version that also filters trades using predicted spread changes. In all five saved outer-fold selections, inner validation favored the simpler price-only policy.

This gives the project a useful decision-level finding: **a more predictable target does not automatically provide additional value to a particular trading rule.** The result concerns the incremental value of the tested spread filter; other uses of liquidity forecasts remain open for investigation.

The repository also implements expanded LightGBM models with **131 total inputs**, including **31 additional raw features**. Their performance and the trading-profit analyses are reserved for the full-data rerun rather than included as headline results here.

## Data and research design

The source is the Kaggle **Optiver Realized Volatility Prediction** dataset: **112 stocks, 3,830 ten-minute windows, and 428,932 stock-window observations**.

Although the competition focuses on realized volatility, this project uses the underlying order books and trades to construct price-return and liquidity targets.

### Quote reconstruction

[`book.py`](src/obforecast/book.py) drops invalid level-1 snapshots, including non-positive prices or sizes and crossed or locked quotes. Within each window, it carries forward the latest valid quote onto a one-second grid. It does not backfill from future observations. Quote age is tracked separately from snapshot validity.

### Forecast targets

| Target | Definition | Interpretation |
|---|---|---|
| Price return | `(m_599 / m_299 − 1) × 10⁴` | Mid-price return in basis points from the end of the input period to the end of the window |
| Spread change | `log(S2/S1)` | Change between first-half and second-half time-weighted average relative spreads |

Here, `m_t` is the mid-price at second `t`; `S1` and `S2` summarize seconds 0–299 and 300–599, respectively. The spread target measures average liquidity over the forecast period.

### Feature engineering

The baseline feature set combines first-half measurements of spread, minimum-spread/tick behavior, volatility, price movement and tilt, quote activity, depth, and imbalance. Stock-relative features use statistics learned on training folds. Leave-one-out market summaries capture the behavior of other stocks in the same window, alongside relative-to-market inputs.

The expanded feature set adds spread-path dynamics, depth trends, trade-print information, cross-stock dispersion, additional market context, stock target statistics, and a categorical stock identifier.

See [`features.py`](src/obforecast/features.py) and the [feature inventory](results/metrics/feature_inventory.csv) for the full specification.

### Model evaluation

[`validation.py`](src/obforecast/validation.py) uses **five outer folds and three inner folds**, with entire market windows assigned to the same fold. Inner folds select forecasting models and trading-rule parameters; outer folds evaluate those selections. Preprocessing is fitted within training folds, and split identifiers are saved for reproducibility.

Reference predictors include zero, training-fold stock means, and persistence. The baseline learned models are ridge regression and histogram gradient boosting; the expanded modeling stage uses LightGBM.

### Trading simulation and sensitivity analysis

[`trading.py`](src/obforecast/trading.py) implements a long-only rule that enters when the predicted return exceeds the current relative spread, additional costs, and a selected margin. An optional spread-forecast threshold provides the liquidity-filter comparison. Parameters are selected on inner folds, with ties favoring the simpler policy.

Trades buy at the displayed ask and sell at the displayed bid. Additional costs of **0, 0.5, or 1 basis point per side** are applied on top of spread crossing. Unfillable trades contribute zero instead of being removed from the opportunity set.

The robustness framework includes feature-family ablations, lagged and shuffled market-context controls, quote-freshness comparisons, return-concentration trims, delayed entry, cost sensitivity, random-row versus grouped splits, and unseen-stock evaluation for the spread model.

## Evaluation scope and reproducibility

The results support **cross-window forecasting within this dataset**. The window identifiers do not provide chronological ordering, so grouped validation is not an out-of-time test. A separate legacy benchmark was examined during earlier team experiments and is treated as a reference evaluation rather than an untouched final holdout. Earlier price experiments also used a different split that included approximately 80% of those benchmark windows in training; the [archive](archive/README.md#was-the-benchmark-holdout-untouched) documents that history.

Trading results describe a simplified one-share, displayed-quote simulation. Queue position, market impact, realistic latency, and venue-specific fees require further modeling. Normalized prices support basis-point comparisons rather than dollar P&L or capacity estimates. The archived analysis also motivates checking return concentration before drawing conclusions about trading performance.

The package, synthetic tests, and synthetic end-to-end runs have been exercised. Full-data execution remains the next reproducibility milestone, including the parquet loader, LightGBM stages, trading outcomes, and robustness results. Earlier upgraded-model and random-split source cells contain saved execution errors; their numerical claims are therefore excluded from the results above. The earlier test environment exercised upgraded-model plumbing with a stand-in estimator because LightGBM and pyarrow were unavailable.

The expanded models currently compute training-row stock target means using each row's own target. This affects training-fit interpretation and is a candidate for out-of-fold encoding in future work; validation targets are not used to construct those training statistics.

Detailed result provenance and source-cell references are recorded in [`headline_claims.csv`](results/metrics/headline_claims.csv), the [results documentation](results/README.md), and the [archive](archive/README.md).

## Repository guide

| Path | Purpose |
|---|---|
| [`notebooks/00_data_exploration.ipynb`](notebooks/00_data_exploration.ipynb) | Exploratory analysis using the final input timing |
| [`notebooks/01_forecasting_and_trading.ipynb`](notebooks/01_forecasting_and_trading.ipynb) | Main forecasting and trading walkthrough |
| [`src/obforecast/`](src/obforecast) | Data loading, reconstruction, features, models, validation, simulation, figures, and CLI |
| [`configs/`](configs) | Full-run and smoke-run configurations |
| [`results/`](results) | Metrics, figures, saved splits, and result documentation |
| [`tests/`](tests) | Data boundaries, splitting, leakage, cost accounting, and notebook execution checks |
| [`scripts/audit_source_notebooks.py`](scripts/audit_source_notebooks.py) | Source-notebook provenance audit |
| [`archive/`](archive) | Original notebooks and earlier experiments |

## Running the project

### 1. Download the data

Accept the [Kaggle competition rules](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/rules) and configure Kaggle API credentials, then run:

```bash
pip install kaggle
kaggle competitions download -c optiver-realized-volatility-prediction -p data/
unzip -q data/optiver-realized-volatility-prediction.zip -d data/optiver-realized-volatility-prediction
```

Preserve the competition directory layout:

```text
<data-dir>/train.csv
<data-dir>/book_train.parquet/stock_id=<id>/*.parquet
<data-dir>/trade_train.parquet/stock_id=<id>/*.parquet
```

Set `OBF_DATA_DIR` to the dataset directory or supply `--data-dir` when running the pipeline. Data files are excluded from version control.

### 2. Install the environment

Use Python 3.10 or later:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
pytest
```

The synthetic test suite does not require the competition dataset.

### 3. Run the analysis

```bash
# Small end-to-end run: 8 stocks and 3 × 2 folds
python -m obforecast.run --config configs/smoke.yaml --data-dir "$OBF_DATA_DIR"

# Full development analysis
python -m obforecast.run --config configs/default.yaml --data-dir "$OBF_DATA_DIR"

# Legacy benchmark evaluation after development choices are finalized
python -m obforecast.run --config configs/default.yaml --data-dir "$OBF_DATA_DIR" --score-benchmark
```

Use `--skip-upgraded` to omit LightGBM stages, or `--stages cv,trading` to select stages. Each run writes its configuration and hash, package versions, Git commit, stage runtimes and execution status, and split details to `run_manifest.json`.

To execute the notebooks:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/00_data_exploration.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/01_forecasting_and_trading.ipynb
```

Historical notebook timings were approximately **7.6 minutes for feature construction** and **12.9 minutes for baseline nested validation** across 112 stocks. Expanded models and the full robustness suite take longer; actual runtime depends on hardware and configuration. The implementation is CPU-bound.

### Google Colab

Once these notebooks are available on the repository's `main` branch, open [data exploration](https://colab.research.google.com/github/Erdos-Projects/fall-2026-short-horizon-forecasting/blob/main/notebooks/00_data_exploration.ipynb) or [forecasting and trading](https://colab.research.google.com/github/Erdos-Projects/fall-2026-short-horizon-forecasting/blob/main/notebooks/01_forecasting_and_trading.ipynb) in Colab. Provide `KAGGLE_API_TOKEN` through Colab Secrets, or set `DATA_DIR` in the first cell to an existing dataset folder. The setup cell handles repository installation and data download.

## Contributors

- **Francis White:** development of the spread-change forecasting approach, including the `log(S2/S1)` target, window-grouped validation, and market-context features that became the main notebook's spread model.
- **Nhi Dinh:** realized-volatility and liquidity exploration; spread-widening event classification; extensions to the team's spread-forecasting model; trading simulation and robustness analysis in the merged notebook; repository restructuring.
- **Swarnendu Banerjee:** data coverage analysis, order-book and trade feature engineering, and exploratory price-level and return modeling.

Data use is subject to the Kaggle competition rules. No separate repository license has been added.
