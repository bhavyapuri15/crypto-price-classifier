# Crypto Price Movement Classifier

A beginner ML project predicting short-term BTC/USDT price direction, built alongside Andrew Ng's Machine Learning Specialization. The goal wasn't to beat the market — it was to build a complete, honest pipeline (data → features → multiple models → evaluation) and compare how different algorithms from the specialization behave on the same real-world task.

## Task

Given the last hour of BTC/USDT 5-minute candle data, predict whether the price will be **higher or lower 15 minutes (3 candles) from now**. A binary classification problem.

## Data

- Source: Binance public REST API (`/api/v3/klines`)
- 21 days of 5-minute OHLCV candles for BTC/USDT (~7,000 rows)
- Chronological train/test split (70/30) — **no random shuffling**, since adjacent rows share overlapping price history. A random split would leak information from the test set into training and produce misleadingly good results.

## Features

| Feature | Description |
|---|---|
| `return_1` | % price change from 1 candle ago |
| `return_3` | % price change from 3 candles ago (15 min momentum) |
| `rolling_vol` | Std. dev. of `return_1` over the last 6 candles |
| `volume_change` | % change in volume vs. previous candle |

## Models

Four models from the ML Specialization, trained on identical features and evaluated on the same held-out test set:

1. Logistic Regression
2. Small Neural Network (1 hidden layer, 8 units)
3. Decision Tree (max depth 5)
4. Random Forest (100 trees, max depth 5)

## Results

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| Logistic Regression | 0.503 | 0.453 | 0.277 |
| Neural Network | 0.518 | 0.449 | 0.116 |
| Decision Tree | 0.493 | 0.467 | 0.554 |
| Random Forest | 0.504 | 0.477 | 0.558 |

### Key findings

- **All four models perform close to chance level (~50% accuracy).** This is expected, not a failure: at a 15-minute horizon, BTC price movement is close to a random walk relative to these simple technical features, consistent with market efficiency at short timescales.
- **Accuracy alone is misleading.** The neural network's low recall (0.12) reveals it mostly predicted "price will go down" — a lazy strategy that scores ~52% accuracy on a near-balanced dataset without learning anything useful. The decision tree and random forest predicted both classes in reasonable proportion, giving much healthier recall (~0.55) despite near-identical accuracy to the other models.
- **Logistic regression and the tree-based models disagreed on feature importance.** Logistic regression found almost no linear signal in any feature except `volume_change`, while the tree models ranked `rolling_vol` and the return features as more important. This suggests any structure in the data is more likely non-linear or threshold-based — something only tree-based models could represent — even though no model converted that structure into real predictive accuracy.

## What I'd try next

- A longer label horizon (e.g. 1 hour instead of 15 minutes) to see if signal strengthens at lower-noise timescales
- Order-book-derived features instead of just OHLCV
- Testing on a second symbol to check if results are BTC-specific or general

## Project structure

```
crypto_classifier.ipynb   # Main notebook: data pipeline, features, all 4 models
btcusdt_5m.csv             # Raw cleaned 21-day OHLCV dataset
train_data.csv              # Chronological training split (features + label)
test_data.csv               # Chronological test split (features + label)
results.csv                 # Logistic regression + neural network results
final_results.csv           # All 4 models' results
requirements.txt            # Python dependencies
```

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Built as part of the Machine Learning Specialization (Andrew Ng, DeepLearning.AI / Stanford).
