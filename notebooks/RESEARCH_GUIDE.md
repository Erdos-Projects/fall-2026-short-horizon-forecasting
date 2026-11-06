# Notebook guide

**Start with [Forecasting and trading](01_forecasting_and_trading.ipynb)** for the final results. The other notebooks show the work behind them.

| Notebook | What you will find |
|---|---|
| [Data exploration](00_data_exploration.ipynb) | Prices, volatility, spreads, liquidity and differences between stocks |
| [Forecasting and trading](01_forecasting_and_trading.ipynb) | Final models, results, trading simulation and checks |
| [Final mid-price forecasting](02_forecasting_final_midprice.ipynb) | Swarnendu’s book and trade features and a comparison of price models against a persistence baseline |
| [Spread models](03_spread_model_development.ipynb) | Francis’s step-by-step comparison of eight models |
| [Spread-widening events](04_spread_widening_events.ipynb) | Nhi’s event definition, threshold comparisons and classifier |

Notebooks 00, 02, 03 and 04 were run from top to bottom on the full data (all 112 stocks), so their outputs are real results from those runs.

## How the pieces connect

The price experiments explore useful signals. The spread models forecast changes in average liquidity. The event classifier asks a different question: which windows are at risk of a large spread increase? The final study brings price and spread forecasts together and tests a simple trading rule.

Each experiment keeps its own target and evaluation method. A price-level score, return score and classification score describe different tasks.

## Running the notebooks

The analysis code is in [`src/obforecast`](../src/obforecast) and the settings are in [`configs/`](../configs). Install with `pip install -r requirements.txt` and `pip install -e .`, set `OBF_DATA_DIR` to the Kaggle data folder, and run any notebook. `configs/smoke.yaml` runs a quick three-stock version. Swarnendu's notebook reads the data from the team's shared Google Drive folder.

## Original work and details

Both [original notebooks](../archive/README.md) keep all 148 cells and 20 saved figures. The original author name Eloise refers to **Nhi Dinh**.

See the [experiment history](../archive/README.md) for version-specific notes.
