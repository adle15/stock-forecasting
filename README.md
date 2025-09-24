# 📈 Stock Forecasting: BBCA (IDX:BBCA.JK)

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
- **MAPE** (less reliable for small returns)
- **Direction Accuracy** (% correct sign predictions)

Example results:
