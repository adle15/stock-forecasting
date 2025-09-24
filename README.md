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
  - **Test**: last **70 trading days** (May–Sep 2025)

**Example of Processed Dataset**
| Date       |   Close   |    High   |    Low    |    Open   |  Volume   |    DR    |    RSI    |    ATR    |    AO     | dow | sentiment | news_count |
|------------|-----------|-----------|-----------|-----------|-----------|----------|-----------|-----------|-----------|-----|-----------|------------|
| 2023-01-03 | 7872.0419 | 7918.0773 | 7849.0243 | 7872.0419 |  27399100 |  0.00000 |   0.00000 |  80.56183 |   0.00000 |   1 |  -1.00000 |          0 |
| 2023-01-04 | 7687.9009 | 7895.0599 | 7687.9009 | 7849.0246 |  90918800 | -2.05279 |   0.00000 | 122.76089 |   0.00000 |   2 |   0.29629 |          3 |
| 2023-01-05 | 7595.8301 | 7710.9184 | 7503.7594 | 7687.9007 | 128838500 | -1.19761 |   0.00000 | 143.86042 |   0.00000 |   3 |  -0.98665 |          2 |
| 2023-01-06 | 7641.8657 | 7664.8834 | 7457.7244 | 7457.7244 |  69286600 |  2.46914 |  14.28580 | 156.52014 |   0.00000 |   4 |  -0.74662 |          0 |
| 2023-01-09 | 7779.9707 | 7779.9707 | 7664.8824 | 7664.8824 |  86916900 |  1.50150 |  39.99992 | 153.45094 | -26.08669 |   0 |   1.00000 |          3 |

## 🧮 Feature Engineering
- **Target variable**:  
  \[
  DR_t = 100 \times \frac{Close_t - Open_t}{Open_t}
  \]
  ```python
    # -------------------------------
    # 2. Daily Return (Target Variable)
    # -------------------------------
    bbca["DR"] = 100 * (bbca["Close"] - bbca["Open"]) / bbca["Open"]
  ```

- **Technical indicators**:
  - Relative Strength Index (RSI)
  - Average True Range (ATR)
  - Awesome Oscillator (AO)
  ```python
    # -------------------------------
    # 3. Technical Indicators
    # -------------------------------
    def compute_rsi(series, window=14):
      delta = series.diff()
      up = delta.clip(lower=0)
      down = -1 * delta.clip(upper=0)
      ma_up = up.rolling(window=window, min_periods=1).mean()
      ma_down = down.rolling(window=window, min_periods=1).mean()
      rs = ma_up / (ma_down + 1e-9)
      return 100 - (100 / (1 + rs))

    def compute_atr(high, low, close, window=14):
      prev_close = close.shift(1)
      tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
      ], axis=1).max(axis=1)
      return tr.rolling(window=window, min_periods=1).mean()

    def compute_ao(close, short=5, long=34):
      sma_short = close.rolling(window=short, min_periods=1).mean()
      sma_long = close.rolling(window=long, min_periods=1).mean()
      return sma_short - sma_long
  ```

- **Seasonality features**:
  - Day of week (one-hot)
  - Month (one-hot)
  ```python
    # -------------------------------
    # 4. Seasonality Features
    # -------------------------------
    bbca["dow"] = bbca.index.dayofweek
    bbca["month"] = bbca.index.month
    dow = pd.get_dummies(bbca["dow"], prefix="dow")
    month = pd.get_dummies(bbca["month"], prefix="m")
    bbca = bbca.join(dow).join(month)
  ```

- **Dummy sentiment/news**:
  - Sentiment ~ sign of yesterday's return + random noise
  - News count ~ Poisson distributed (λ=2)
  ```python
    # -------------------------------
    # 5. Dummy Sentiment / News
    # -------------------------------
    sentiment = np.sign(bbca["DR"].shift(1).fillna(0))
    sentiment = sentiment + np.random.normal(scale=0.2, size=len(sentiment))
    bbca["sentiment"] = np.clip(sentiment, -1, 1)
    bbca["news_count"] = np.random.poisson(lam=2.0, size=len(bbca))
  ```

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
<img width="1154" height="547" alt="image" src="https://github.com/user-attachments/assets/d7a636c7-4389-4f23-a637-ae009249f720" />

- **This image is the result of predicted value on 10-19 September 2025 and comparison between actual value.**

<img width="590" height="390" alt="image" src="https://github.com/user-attachments/assets/d104f49b-896b-426d-936a-99f5142342e2" />

## 📝 Notes
Paper reference: [Link](https://www.sciencedirect.com/science/article/pii/S0957417424021663)
