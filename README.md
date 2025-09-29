Quant Research Project: Shaswat

## What’s inside
- `zscore_base_strategy.ipynb` — Baseline rolling **z-score** strategy with parameter sweeps.  
- `ml_enhanced_strategy.ipynb` — My attempts to improve entries/exits (grid, ML, gradient/SPSA, coordinate descent, adaptive ML).

---

## 1) Problem & Dataset
We test a volatility pairs strategy on **minute-level implied volatilities (IVs)** of **Nifty** and **Bank Nifty**.  
Trading horizon targeted: **30 minutes to ~5 days**.

**Spread & P/L**
- `Spread = IV_banknifty − IV_nifty`  
- `P/L = Spread × (TTE)^0.7`

---

## 2) Data Cleaning Done
1. **Filter trading calendar:** removed weekends; kept **09:15–15:30 IST**.  
2. **Handle missing data:** **forward-filled** (`ffill`) — consistent with financial time-series practice where the last known market state is assumed to persist until new data arrives. (Dropping or filling with 0 would distort spreads and IV dynamics.)
3. **De-dup & typing:** drop duplicate timestamps; ensure numeric types; time index sorted.  
4. **Pandas-ready::** Structure as: `time, banknifty, nifty, tte`.

---

## 3) Base Model — Rolling Z-Score Strategy
- `spread = banknifty − nifty`
- Rolling window (tested): **30, 120, 375** minutes  
- `z = (spread − μ_roll) / σ_roll`
- **Entry/Exit (mean reversion, crossing logic):**
  - Long when z **crosses down** through `−entry_z`; exit when it **reverts up** to `−exit_z`.
  - Short when z **crosses up** through `+entry_z`; exit when it **reverts down** to `+exit_z`.
  - Crossing conditions avoid flicker/double trades.
- Metrics per configuration: **PnL, Sharpe, Max Drawdown, Win rate**.

> Selection note: **Sharpe** and **Win rate** are more stable selectors than raw PnL / Drawdown alone (those vary more with regime and trade count).

**Baseline sweep (examples)**

| window | entry_z | exit_z | trades | pnl | sharpe | max_dd | winrate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 1.5 | 0.00 | 2343 | 78.32 | 3.396 | -2.323 | 0.777 |
| 30 | 2.0 | 0.00 | 1965 | 76.32 | 3.666 | -1.774 | 0.790 |
| 120 | 1.5 | 0.00 | 1503 | 88.23 | 4.424 | -1.362 | 0.882 |
| 375 | 1.5 | 0.00 | 784  | 79.60 | 5.583 | -2.210 | 0.856 |

*(Full table in notebook.)*

---

## 4) “Better than z-score” — What I tried

### A) Grid Search (25% train / 75% test)
Pick best `(entry_z, exit_z)` on **train**, evaluate on **test**.

| window | best_entry_z | best_exit_z | train_pnl | train_sharpe | test_trades | test_pnl | test_sharpe | test_max_dd | test_winrate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30  | 1.00 | 0.00 | 21.36 | 3.771 | 1959 | 71.47 | 3.391 | -2.203 | 0.759 |
| 120 | 1.25 | 0.00 | 21.99 | 4.717 | 1286 | 68.06 | 3.911 | -2.491 | 0.880 |
| 375 | 1.25 | 0.00 | 31.23 | 5.826 | 713  | 58.83 | 5.369 | -0.926 | 0.871 |

### B) ML Optimization of (entry_z, exit_z) (same split)
Weighted objective (PnL, Sharpe, Drawdown) using Optuna/TPE.

| window | best_entry_z | best_exit_z | train_pnl | train_sharpe | test_trades | test_pnl | test_sharpe | test_max_dd | test_winrate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30  | 1.3063 | 0.0278 | 24.49 | 4.109 | 1882 | 69.55 | 3.665 | -1.430 | 0.779 |
| 120 | 1.3349 | 0.0354 | 25.54 | 5.278 | 1246 | 70.76 | 4.250 | -1.254 | 0.880 |
| 375 | 1.1139 | 0.0493 | 33.80 | 5.824 | 779  | 59.11 | 5.545 | -0.871 | 0.881 |

### C) Gradient / SPSA (same split)
| window | best_entry_z | best_exit_z | test_trades | test_pnl | test_sharpe | test_max_dd | test_winrate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30  | 1.4363 | 0.0000 | 1827 | 68.53 | 3.554 | -1.791 | 0.776 |
| 120 | 1.3677 | 0.0705 | 1227 | 73.06 | 4.464 | -1.072 | 0.878 |
| 375 | 1.1125 | 0.0000 | 764  | 59.83 | 5.654 | -0.871 | 0.884 |

### D) Coordinate Descent (same split)
| window | best_entry_z | best_exit_z | test_trades | test_pnl | test_sharpe | test_max_dd | test_winrate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30  | 2.5882 | 0.7703 | 941  | 32.50 | 3.569 | -0.961 | 0.744 |
| 120 | 2.7163 | 1.4991 | 451  | 17.97 | 3.371 | -0.926 | 0.752 |
| 375 | 3.0935 | 0.8347 | 224  | 16.80 | 5.346 | -0.820 | 0.848 |

### E) Adaptive ML (parameter prediction)
- Rich feature set; predicted window, entry/exit thresholds, horizons, and position sizing per regime.
- Results (reporting rates only to avoid mixing train/test PnL scales):
  - **Sharpe ≈ 4.21**, **Win rate ≈ 86.5%**, **Drawdown ≈ −1.54**, **Trades = 497**  
  - *(Total PnL omitted here — training segments differ across methods.)*

---

## 5) Findings & Interpretation
- The **base rolling z-score** remains a **strong benchmark**: simple, interpretable, robust.  
- Most optimizers (grid, Optuna/TPE, SPSA, coordinate) **converged to low entry z** and **small exit z** (often exit≈0), which matches mean-reversion intuition on spreads.
- **Adaptive ML** showed promise on risk metrics (Sharpe/win rate) but didn’t decisively beat the base across all windows—likely due to **limited sample size** and **regime shifts**.
- **Lead–lag idea** (Bank Nifty as a leading indicator) did **not** show persistent edge in this dataset, despite the hypothesis (≈20% direct bank weight in Nifty; ≈36% incl. NBFCs).

---

## 6) Why my enhanced models didn’t beat the base (yet)

In theory, the enhanced approaches (ML optimization, SPSA gradient descent, coordinate descent, adaptive ML) **should have outperformed** the simple z-score strategy. They incorporate richer signals, dynamic parameter adaptation, and multi-objective scoring.  

However, in practice, on this dataset they did **not consistently beat the base strategy**. The most likely reasons are:  

- **Limited dataset**: more complex/ML-driven methods need a longer history with diverse market regimes to generalize well.  
- **Parameter instability**: with fewer samples, optimized thresholds can become noisy and less robust out-of-sample.  
- **Base model robustness**: rolling z-scores with well-chosen windows are surprisingly resilient to small sample sizes and regime drift.  

### Future Directions
With more data or further tuning, the advanced approaches should theoretically yield stronger results. Immediate next steps could be:  
- Adding **regime segmentation** (e.g., high vs. low volatility periods with separate thresholds).  
- Penalizing **excessive trade density** in the optimization objective to reduce churn.  
- Incorporating **transaction cost sensitivity** to test robustness under realistic frictions.  
