<img width="1154" height="547" alt="image" src="https://github.com/user-attachments/assets/51ec1923-bce2-49dc-abaf-25d5186b2884" /># 📈 Stock Forecasting: BBCA (IDX:BBCA.JK)

This repository implements a stock forecasting pipeline for **Bank Central Asia (BBCA.JK)** using **LSTM** and heuristic scoring as described in the research paper (reference in repo).

## 🚀 Project Overview
- **Objective**: Predict **daily return (DR)** of BBCA based on 3-day historical windows.
- **Method**:
  - LSTM regression model (predict DR one day ahead).
  - Heuristic scoring to evaluate prediction quality and trading usefulness.

## 🗂️ Dataset
- Historical OHLCV data from **Yahoo Finance** (`yfinance`).
- Range: `2023-01-01` → `2025-08-31`
- Splitting:
  - **Train**: from start until ~mid 2025
  - **Validation**: last **140 trading days** before test period
  - **Test**: last **70 trading days** (Jul–Aug 2025)

## 🧮 Feature Engineering
- **Target variable**:  
  \[
  DR_t = 100 \times \frac{Close_t - Open_t}{Open_t}
  \]

- **Technical indicators**:
  - Relative Strength Index (RSI)
  - Average True Range (ATR)
  - Awesome Oscillator (AO)

- **Seasonality features**:
  - Day of week (one-hot)
  - Month (one-hot)

- **Dummy sentiment/news**:
  - Sentiment ~ sign of yesterday's return + random noise
  - News count ~ Poisson distributed (λ=2)

## 🧠 Model
- Input: sliding window of **3 previous trading days**
- Output: predicted DR (t+1)
- Architecture:
  - LSTM(128, tanh)
  - Dropout(0.2)
  - Dense(64, ReLU)
  - Dense(1, linear)

- Optimizer: Adam (lr=0.001)  
- Loss: MSE  
- Metrics: MAE, RMSE, Directional Accuracy

## 📊 Evaluation
- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- **Direction Accuracy** (% correct sign predictions)
- Performance Metrics:
  - Train -> MAE: 0.0752, RMSE: 0.0989, DirAcc: 86.42%
  - Val   -> MAE: 1.3565, RMSE: 1.7367, DirAcc: 46.43%
  - Test  -> MAE: 1.6061, RMSE: 1.9165, DirAcc: 30.00%

Example results:
<img width="1154" height="547" alt="image" src="https://github.com/user-attachments/assets/83abab6d-8482-4dc2-97fd-155090c8f392" />

