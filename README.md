# This repository implements a stock forecasting pipeline for **Bank Central Asia (BBCA.JK)** using **LSTM**.

## 🚀 Project Overview
- **Objective**: Predict **daily return (DR)** of BBCA based on 3-day historical windows.
- **Method**:
  - LSTM regression model (predict DR one day ahead).
  - Heuristic scoring to evaluate prediction quality and trading usefulness.

## 🗂️ Dataset
- Historical OHLCV data from **Yahoo Finance** (`yfinance`).
- Range: `2023-01-01` → `2025-09-09`
- Splitting:
  - **Train**: from start until ~ last 2024
  - **Validation**: last **140 trading days** before test period
  - **Test**: last **70 trading days** (Aug–Sep 2025)

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
    - Train -> MAE: 0.0681, RMSE: 0.0932, DirAcc: 86.16%
    - Val   -> MAE: 1.2843, RMSE: 1.6685, DirAcc: 48.57%
    - Test  -> MAE: 1.2364, RMSE: 1.5170, DirAcc: 28.57%

Example results:
- **This image is the result for train, validation, and test set.**

<img width="1154" height="547" alt="image" src="https://github.com/user-attachments/assets/83abab6d-8482-4dc2-97fd-155090c8f392" />

- **This image is the result of predicted value on 10-19 September 2025 and comparison between actual value.**

<img width="590" height="390" alt="image" src="https://github.com/user-attachments/assets/712d9fb1-cb1a-4caa-8b66-e0a6a1e71ea8" />


## 📝 Notes
Paper reference: [Link](https://www.sciencedirect.com/science/article/pii/S0957417424021663)
