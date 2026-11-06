# Notebook guide

**Start with [Forecasting and trading](01_forecasting_and_trading.ipynb)** for the final results. The other notebooks show the work behind them.

| Notebook | What you will find | Reading / running |
|---|---|---|
| [Data exploration](00_data_exploration.ipynb) | Prices, volatility, spreads, liquidity and differences between stocks | Includes the original volatility/liquidity figure; rerun with the dataset for the remaining plots |
| [Forecasting and trading](01_forecasting_and_trading.ipynb) | Final models, results, trading simulation and checks | Read saved result tables on GitHub, or run the analysis |
| [Price experiments](02_price_experiments.ipynb) | Swarnendu’s features, model comparisons and figures | Original research record; some cells need updating before a full rerun |
| [Spread models](03_spread_model_development.ipynb) | Francis’s step-by-step comparison of eight models | Original saved results; setup adapted for local use and Colab |
| [Spread-widening events](04_spread_widening_events.ipynb) | Nhi’s event definition, threshold comparisons and classifier | Code is included; this version has no saved performance results |

## How the pieces connect

The price experiments explore useful signals. The spread models forecast changes in average liquidity. The event classifier asks a different question: which windows are at risk of a large spread increase? The final study brings price and spread forecasts together and tests a simple trading rule.

Each experiment keeps its own target and evaluation method. A price-level score, return score and classification score describe different tasks.

## Running the notebooks

Use the setup cells and the Kaggle dataset. The final study defaults to a small **smoke run** for trying the code; choose `configs/default.yaml` for the full analysis. The extracted spread-model and event notebooks retain their original full-data experiments, which can take longer.

Saved tables at the start of the final study come from the published full-data runs. They stay separate from any new run you perform. The new spread/event extractions have been checked with synthetic data but have not been rerun on the full Kaggle dataset.

## Original work and details

Both [original notebooks](../archive/README.md) keep all 148 cells and 20 saved figures. The original author name Eloise refers to **Nhi Dinh**.

See [technical notes and setup](../docs/TECHNICAL_NOTES.md) for methods and reproduction instructions, [experiment history](../archive/README.md) for version-specific notes, and the [preservation record](../results/research_preservation.json) for the cell inventory.
