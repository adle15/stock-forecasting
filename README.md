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
| Date       | Close   | High    | Low     | Open    | Volume   | DR       | RSI     | ATR     | AO       | ADX     | AI       | dow_0 | dow_1 | dow_2 | dow_3 | dow_4 | m_1 | m_2 | ... | m_12 | sentiment | news_count |
|------------|---------|---------|---------|---------|----------|----------|---------|---------|----------|---------|----------|-------|-------|-------|-------|-------|-----|-----|-----|------|-----------|------------|
| 2023-01-03 | 7872.04 | 7918.08 | 7849.02 | 7872.04 | 27399100 | 0.000000 | 0.0000  | 80.56   | 0.000000 | 0.0000  | 0.000000 | True  | False | False | False | False | ... | ... | ... | ...  | -1.000000 | 2          |
| 2023-01-04 | 7687.90 | 7895.06 | 7687.90 | 7849.02 | 90918800 | -2.052786| 0.0000  | 122.76  | 0.000000 | 50.0000 | 0.000000 | False | True  | False | False | False | ... | ... | ... | ...  | -0.316286 | 2          |
| 2023-01-05 | 7595.83 | 7710.92 | 7503.76 | 7687.90 |128838500 | -1.197605| 0.0000  | 143.86  | 0.000000 | 66.6667 | 0.000000 | False | False | True  | False | False | ... | ... | ... | ...  | -1.000000 | 2          |
| 2023-01-06 | 7641.87 | 7664.88 | 7457.72 | 7457.72 | 69286600 | 2.469136 |14.2858  | 156.52  | 0.000000 | 75.0000 | 0.000000 | False | False | False | True  | False | ... | ... | ... | ...  | -0.842861 | 2          |
| 2023-01-09 | 7779.97 | 7779.97 | 7664.88 | 7664.88 | 86916900 | 1.501502 |39.9999  | 153.45  | -26.0867 | 70.9091 | -26.0867 | True  | False | False | False | False | ... | ... | ... | ...  | 0.964476  | 2          |

## 🧮 Feature Engineering
- **Target variable**:  
  \[
  DR_t = 100 * / (Close_t - Open_t)(Open_t)
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
  - Average Directional Index (ADX)
  - Aroon Indicator (AI)
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
  
    def compute_adx(high, low, close, window=14):
      # True Range components
      plus_dm = high.diff()
      minus_dm = low.diff() * -1

      plus_dm[plus_dm < 0] = 0
      minus_dm[minus_dm < 0] = 0

      tr1 = (high - low).abs()
      tr2 = (high - close.shift(1)).abs()
      tr3 = (low - close.shift(1)).abs()
      tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

      atr = tr.rolling(window=window, min_periods=1).mean()

      plus_di = 100 * (plus_dm.rolling(window=window, min_periods=1).mean() / atr)
      minus_di = 100 * (minus_dm.rolling(window=window, min_periods=1).mean() / atr)

      dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
      adx = dx.rolling(window=window, min_periods=1).mean()

      return adx

    def compute_aroon(high, low, window=25):
      # Aroon Up
      rolling_high_idx = high.rolling(window=window, min_periods=1).apply(lambda x: x.argmax(), raw=True)
      aroon_up = 100 * ((window - rolling_high_idx) / window)

      # Aroon Down
      rolling_low_idx = low.rolling(window=window, min_periods=1).apply(lambda x: x.argmin(), raw=True)
      aroon_down = 100 * ((window - rolling_low_idx) / window)

      # Aroon Oscillator (Up - Down)
      aroon_osc = aroon_up - aroon_down
      return aroon_osc
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
<img width="1154" height="547" alt="image" src="https://github.com/user-attachments/assets/47339bf6-75cc-4030-8598-f4ba1703b85d" />


- **This image is the result of predicted value on 10-19 September 2025 and comparison between actual value.**
<img width="590" height="390" alt="image" src="https://github.com/user-attachments/assets/dbff606f-14e8-466b-be51-edb3df899d2d" />



## 📝 Notes
Paper reference: [Link](https://www.sciencedirect.com/science/article/pii/S0957417424021663)
