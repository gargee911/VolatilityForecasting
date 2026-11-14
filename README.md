# Comparative Forecasting of Cryptocurrency Realized Volatility Using GARCH-Family and Multivariate Bidirectional LSTM Models

**Author:** Gargee Kate (AU25110010)
**Track:** Experimentation / Research-based Project (AI Labs)

This project forecasts the **next 7 days' average daily realized volatility** of five cryptocurrencies and compares classical **GARCH-family** econometric models with **LSTM / Bidirectional LSTM** neural networks. It extends a reference Bitcoin notebook to Ethereum, Solana, XRP and Dogecoin, and adds a **lookback-window study** (7, 14, 21 and 30 days) for every neural network.

---

## Research question

> **H1:** A multivariate two-layer Bidirectional LSTM achieves lower out-of-sample realized-volatility forecasting error (primarily RMSPE, secondarily RMSE) than the GARCH-family models across the selected cryptocurrencies.
>
> **H0:** The multivariate Bidirectional LSTM does not consistently outperform the GARCH-family models.

**Outcome:** H1 is **not supported**. The multivariate BiLSTM beat the best GARCH-family model only for SOL and DOGE, and it was not the best model for any asset.

---

## Data

| Crypto | Yahoo Finance ticker |
|---|---|
| Bitcoin | `BTC-USD` |
| Ethereum | `ETH-USD` |
| Solana | `SOL-USD` |
| XRP | `XRP-USD` |
| Dogecoin | `DOGE-USD` |

- **Source:** Yahoo Finance via `yfinance`, daily OHLCV, **1 Jan 2021 – 31 Dec 2025**.
- **Split (chronological):** last 30 usable days = test, the 365 days before = validation, the rest = training.
  For Bitcoin: 1,788 usable days → 1,393 train / 365 validation / 30 test.
- Tether (USDT) was also modelled but excluded from the comparison because its results were very poor.

---

## Method

1. **Returns:** daily percent returns (input to GARCH models) and log returns.
2. **Realized volatility:** 30-day rolling window, `RV_t = sqrt( Σ r² / (n − 1) )`.
3. **Target:** realized volatility shifted 7 days ahead (`vol_future`).
4. **Scaling:** `MinMaxScaler` fitted on training volatility.
5. **Baselines:** Mean (training average) and Random Walk (current volatility as forecast).
6. **GARCH family** (constant mean, refitted at every validation step, 7-day horizon):
   - GARCH(1,1), normal errors
   - Analytical GJR-GARCH(1,1,1), skewed Student-t
   - TARCH(1,1), skewed Student-t, bootstrap and simulation forecasts
   - TARCH with orders *(p, q, o)* chosen by grid search over p ∈ {1,2,3}, q, o ∈ {0,…,3}
7. **Neural networks** (each trained with lookback windows of **7, 14, 21 and 30 days**):

   | Notebook name | Architecture |
   |---|---|
   | Simple Linear Regression Fully Connected Network | Flatten → Dense(1) |
   | 1-Layered LSTM 20 units | LSTM(20) → Dense(1) |
   | 2-Layered Bidirectional LSTMs (32/16 units) | BiLSTM(32) → BiLSTM(16) → Dense(1) |
   | 1 Conv1D 2-Layered Bidirectional LSTMs (32/16 units) | Conv1D(32, k=4, s=2) → BiLSTM(32) → BiLSTM(16) → Dense(1) |
   | Stacked Bi-LSTM (Tuned LR) | Same as above, SGD + momentum 0.9, learning rate from a range test |
   | Multivariate Stacked Bi-LSTM (lstm_5) | LayerNorm → BiLSTM(32) → Dropout → BiLSTM(16) → Dropout → Dense(1) |
   | Multivariate 3 Bidirect LSTM layers (64/32/16 units) (Layer Normalization) | LayerNorm → BiLSTM 64/32/16 with Dropout(0.1) → Dense(1) |

   Multivariate inputs: `log((High − Low)/Close)`, `(Close − Open)/Open`, `log(Volume)` and current realized volatility.
   Training: MSE loss, RMSPE metric, batch size 64, early stopping (patience 30), seed 2026.
8. **Metrics:** RMSPE (primary) and RMSE on the 365-day validation set.

---

## Results (validation set)

| Crypto | Best model | RMSPE | RMSE | Best GARCH-family (RMSPE) | Random walk (RMSPE) |
|---|---|---|---|---|---|
| BTC | Bootstrap TARCH(3, 2, 0) | **0.2071** | **0.0512** | 0.2071 | 0.2682 |
| ETH | Random Walk | **0.2066** | **0.0573** | Bootstrap TARCH(2, 1, 0): 0.2097 | 0.2066 |
| SOL | Random Walk | **0.3994** | **0.0562** | Bootstrap TARCH(1, 3, 2): 0.4379 | 0.3994 |
| XRP | Stacked Bi-LSTM (Tuned LR=5e-4), n_past=21 | **0.2892** | **0.0773** | Simulation TARCH(1,1): 0.4289 | 0.3383 |
| DOGE | 1 Conv1D 2-Layered Bidirectional LSTMs (32/16 units), n_past=30 | **0.1743** | **0.0215** | Bootstrap TARCH(2, 3, 0): 0.4084 | 0.2153 |

**Key findings**

- **No single best model.** The winner was a GARCH-family model for BTC, the random walk for ETH and SOL, and a univariate BiLSTM-based network for XRP and DOGE.
- **The random walk is a strong benchmark.** It was in the top 5 for every asset.
- **The univariate 2-Layered BiLSTM (32/16)** was the most consistent network: it was in the top 5 for BTC, ETH, XRP and DOGE.
- **Lookback window matters.** Across 31 network–asset combinations, 30 days gave the lowest RMSPE in 13 cases, 21 days in 8, 7 days in 6 and 14 days in 4. A poor window raised a network's error by up to ~4×.
- **GARCH family:** the TARCH models with skewed Student-t errors were the strongest. GARCH(1,1) with normal errors was the weakest GARCH-family model for four of the five assets.

Full lookback matrices, GARCH comparisons and top-5 leaderboards for every crypto are in the report.

---

## Repository structure

> Adjust the file names below to match your repository.

```
├── bitcoin.ipynb        # BTC-USD pipeline
├── ethereum.ipynb       # ETH-USD pipeline
├── solana.ipynb         # SOL-USD pipeline
├── xrp.ipynb            # XRP-USD pipeline
├── dogecoin.ipynb       # DOGE-USD pipeline
├── report/
│   ├── main.tex         # LaTeX report
│   └── main.pdf         # Compiled report
└── README.md
```

Each notebook runs the same pipeline: data download → realized volatility → train/validation/test split → baselines → GARCH family → neural networks for each lookback window → lookback matrix → final leaderboard.

---

## How to run

```bash
# 1. Install dependencies (Python 3.10+ recommended)
pip install numpy pandas matplotlib seaborn scipy scikit-learn statsmodels arch yfinance tensorflow jupyter

# 2. Launch Jupyter and run a notebook top to bottom
jupyter notebook bitcoin.ipynb
```

Notes:
- The notebooks download data from Yahoo Finance, so an internet connection is needed.
- The GARCH forecasts refit the model at every validation step (365 refits per model), so those cells take a while to run.
- To reproduce the report numbers, run every cell in order once, without re-running individual cells out of order.

---

## Limitations

- All comparisons use the 365-day validation set, which was also used to choose the lookback windows and the tuned TARCH orders. The final 30-day test window was not evaluated.
- Each network was trained with a single random seed, and no statistical significance test was applied.
- The multivariate two-layer BiLSTM (lstm_5) appears in the result tables only for XRP.

## Future work

- Report final errors on a held-out test period.
- Use several random seeds and apply Diebold–Mariano tests.
- Add HAR benchmarks and hybrid GARCH–LSTM models.
- Analyse low-, medium- and high-volatility regimes separately.

---

## References

1. *Bitcoin's Realized Volatility Forecasting: A Comparison between GARCH and LSTM* (2023), the reference study for this project.
2. Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987–1007.
3. Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
4. Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance*, 48(5), 1779–1801.
5. Zakoïan, J.-M. (1994). Threshold heteroskedastic models. *Journal of Economic Dynamics and Control*, 18(5), 931–955.
6. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
7. Zahid, M., Iqbal, F., & Koutmos, D. (2022). Forecasting Bitcoin volatility using hybrid GARCH models with machine learning. *Risks*.
8. Amirshahi, B., & Lahmiri, S. (2023). Hybrid deep learning and GARCH-family models for forecasting volatility of cryptocurrencies. *Machine Learning with Applications*.
9. Pratas, T. E., Ramos, F. R., & Rubio, L. J. (2023). Forecasting bitcoin volatility: Exploring the potential of deep learning. *Eurasian Economic Review*, 13, 285–305.
10. Dudek, G., Fiszeder, P., Kobus, P., & Orzeszko, W. (2024). Forecasting cryptocurrencies volatility using statistical and machine learning methods: A comparative study. *Applied Soft Computing*, 151, 111132.
