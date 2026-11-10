# Price and Liquidity Forecasting

Erdős Institute, Fall 2026

This project uses order-book and trade data for 112 stocks to forecast two things over a five-minute horizon: the price return and the change in the bid–ask spread (the gap between the best bid and the best ask). We use the first five minutes of each ten-minute window to predict the next five. We explore the data, compare models for both targets, and then check whether the forecasts help in a simple trading simulation.

The main results are in [04_forecasting_and_trading.ipynb](notebooks/04_forecasting_and_trading.ipynb). The notebooks are numbered in the order we did the work, so you can also read them straight through. All of them are saved with outputs, so they render on GitHub without running anything.

## Notebooks

| Notebook | Contents |
|---|---|
| [00 Data exploration](notebooks/00_data_exploration.ipynb) | Prices, trading activity, volatility and liquidity |
| [01 Mid-price forecasting](notebooks/01_midprice_forecasting.ipynb) | Price features and model comparisons against a persistence baseline |
| [02 Spread models](notebooks/02_spread_models.ipynb) | How the spread model was built up |
| [03 Spread-widening events](notebooks/03_spread_widening_events.ipynb) | Classifying unusually large spread increases |
| [04 Forecasting and trading](notebooks/04_forecasting_and_trading.ipynb) | Final models and the trading simulation |
| [05 Volatility forecasting](notebooks/05_volatility_forecasting.ipynb) | The original Kaggle task, the same idea over five minutes, and what volatility adds to the spread and return forecasts |

[RESEARCH_GUIDE.md](notebooks/RESEARCH_GUIDE.md) explains how the notebooks fit together and how to rerun them. The original working notebooks, with the full history and figures, are in [archive/](archive/README.md).

## Results

Spread changes were much easier to predict than returns. Adding trade data and more detailed features helped both targets a little.

| Target | Baseline R² | Expanded R² |
|---|---:|---:|
| Change in average spread | 0.348 | 0.380 |
| 5-minute return | 0.061 | 0.070 |

R² is the share of variance explained. The scores come from grouped cross-validation, so rows from the same time window always stay in the same fold.

![Forecast accuracy and trading results](results/figures/figure1_accuracy_vs_value.png)

In the trading simulation, a simple rule based on the return forecast made a small profit after costs. Adding the spread forecast to the rule didn't improve it, so better forecast accuracy didn't turn into better trades here.

Forecasting the final mid-price level directly tells the same story: across 10 grouped folds, simply carrying forward the price at five minutes (R² 0.693) was as good as or better than Ridge (0.686), Elastic Net (0.683) and XGBoost (0.664). Most of a price level is explained by the current price; the hard part is the change.

## Data

We used the [Optiver Realized Volatility Prediction](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/data) dataset from Kaggle: order books and trades for 112 stocks across 3,830 ten-minute windows. The data isn't included in this repo, and its use follows the competition rules. To rerun the notebooks, install the code with `pip install -r requirements.txt` and `pip install -e .`, then set `OBF_DATA_DIR` to the downloaded data folder.

## What's left in the repo

| Folder | What's in it |
|---|---|
| [`notebooks/`](notebooks/) | The six project notebooks (00–05) and the [notebook guide](notebooks/RESEARCH_GUIDE.md) |
| [`results/`](results/README.md) | Saved outputs of the full-data run: figures, metric tables, which windows were used for development and held out, and run logs |
| [`src/obforecast/`](src/obforecast/) | The Python code the notebooks use: rebuilding the order book, features, models, validation and the trading simulation |
| [`scripts/`](scripts/) | `build_vol_features.py` builds the cached volatility features for notebook 05; `make_kaggle_notebook.py` writes the Kaggle submission notebook |
| [`kaggle/`](kaggle/) | A self-contained notebook for a late submission to the competition, using notebook 05's model. Private leaderboard RMSPE: 0.2141 with the neighbour features (0.2186 with an earlier version), 0.2455 without |
| [`configs/`](configs/) | Run settings: `default.yaml` for the full analysis, `smoke.yaml` for a quick three-stock test |
| [`archive/`](archive/README.md) | The original Colab notebooks, kept unchanged with their saved outputs and figures |

`requirements.txt` and `pyproject.toml` list the Python packages needed to rerun the notebooks.

## Team

- **Swarnendu Banerjee**: data checks, book and trade features, price-model experiments
- **Nhi Dinh**: volatility and liquidity exploration, spread-widening classification, model extensions
- **Francis White**: spread-model development, market features, grouped validation
