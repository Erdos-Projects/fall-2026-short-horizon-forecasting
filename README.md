# Price and Liquidity Forecasting

*Erdős Institute · Fall 2026*

**Can five minutes of market activity help predict the next five?**

We study price changes and the bid–ask spread—the gap between the best buying and selling prices. Using order-book and trade data for **112 stocks**, we explore market patterns, compare forecasting models, and test how useful the predictions are in a simple trading simulation.

## Start here

Open [the main notebook](notebooks/01_forecasting_and_trading.ipynb) for the results and trading study. Start with [data exploration](notebooks/00_data_exploration.ipynb) if you want to see the data first. You can read the saved results directly on GitHub.

## What we worked on

| Part of the project | Notebook |
|---|---|
| Understanding prices, trading activity, volatility and liquidity | [Data exploration](notebooks/00_data_exploration.ipynb) |
| Building price features and comparing models | [Price experiments](notebooks/02_price_experiments.ipynb) |
| Improving spread forecasts step by step | [Spread models](notebooks/03_spread_model_development.ipynb) |
| Predicting unusually large spread increases | [Spread-widening events](notebooks/04_spread_widening_events.ipynb) |
| Bringing the forecasts together and testing trading decisions | [Forecasting and trading](notebooks/01_forecasting_and_trading.ipynb) |

The [notebook guide](notebooks/RESEARCH_GUIDE.md) explains where to start and which experiments have saved results. The [original notebooks](archive/README.md) also keep the full research history and figures.

## What we found

Spread changes were easier to predict than price returns. Adding trade information and more detailed features improved both forecasts.

| Forecast | Baseline model R² | Expanded model R² |
|---|---:|---:|
| Change in average spread | 0.348 | 0.380 |
| Five-minute price return | 0.061 | 0.070 |

R² measures how much variation a model explains; higher is better. These scores come from validation that keeps each market window together.

![Forecast accuracy and trading results](results/figures/figure1_accuracy_vs_value.png)

The price-based rule earned a small simulated return after costs. Adding the spread forecast did not improve that particular rule. **Better forecasting accuracy did not automatically mean better trading decisions.**

The trading result depends heavily on a few windows, and the data does not support a chronological test. This is a research simulation. See the [main notebook](notebooks/01_forecasting_and_trading.ipynb) for the numbers and their limits.

## Data

We used the [Optiver Realized Volatility Prediction dataset on Kaggle](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/data): order books and trades for 112 stocks over 3,830 ten-minute windows. The data is not included in this repository.

## Team

- **Swarnendu Banerjee:** data checks, book/trade features and price-model experiments.
- **Nhi Dinh:** volatility and liquidity exploration, spread-widening classification, model extensions, trading simulation and robustness checks.
- **Francis White:** spread-model development, market features and grouped validation.

Data use follows the Kaggle competition rules.
