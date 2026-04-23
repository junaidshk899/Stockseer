# 📈 StockSeer — Stock Price Prediction with ML & LSTM

> A complete end-to-end stock price prediction system using Linear Regression, Random Forest, and LSTM neural networks — trained on real market data with 25+ engineered technical indicators.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()

---

## 📌 Project Overview

**StockSeer** is a professional-grade machine learning pipeline for predicting next-day stock closing prices using historical OHLCV (Open, High, Low, Close, Volume) data. Built as part of a **Data Science Internship (Task 2)**, the project compares three progressively complex models and produces a 30-day forward price forecast.

| Detail | Value |
|---|---|
| **Author** | Muhammad Junaid Asim |
| **Domain** | Data Science Internship |
| **Stock** | AAPL (Apple Inc.) — easily configurable |
| **Data Period** | 2018–2024 (6 years, ~1,510 trading days) |
| **Target** | Next-day closing price |
| **Best Model** | LSTM — R² ≈ 0.9974, MAPE ≈ 0.63% |

---

## ✨ Features

- **Real market data** — downloads live OHLCV data via `yfinance` (any ticker)
- **25+ technical indicators** — RSI, MACD, Bollinger Bands, ATR, OBV, Moving Averages, lag features, and more
- **Three ML models** — Linear Regression (baseline) → Random Forest → LSTM (deep learning)
- **Leakage-free pipeline** — chronological splitting, scaler fitted only on training data
- **Rich visualizations** — price history, correlation heatmap, technical indicators, model comparisons, forecast chart
- **30-day auto-regressive forecast** — sliding window multi-step prediction with uncertainty band
- **Feature importance** — Random Forest ranks which indicators matter most
- **Full evaluation suite** — MAE, RMSE, MAPE, R² for all models
- **Google Colab ready** — single notebook, runs top-to-bottom

---

## 🗂️ Repository Structure

```
stockseer/
│
├── Stock_Price_Prediction.ipynb   # Main Colab notebook (full pipeline)
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
│
├── docs/
│   ├── Internship_Report.pdf      # Full internship report (PDF)
│   └── Internship_Report.docx     # Full internship report (Word)
│
└── assets/
    └── charts/                    # Sample output charts
```

---

## 🚀 Quick Start

### Option 1 — Google Colab (Recommended)

1. Click the **Open in Colab** badge above
2. Go to **Runtime → Change runtime type → GPU** (optional, speeds up LSTM)
3. Run **Runtime → Run all**

That's it. All libraries install automatically.

### Option 2 — Local Setup

```bash
# Clone the repository
git clone https://github.com/junaidasim899/stockseer.git
cd stockseer

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook Stock_Price_Prediction.ipynb
```

---

## 📦 Requirements

```txt
yfinance>=0.2.18
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.13.0
scikit-learn>=1.2.0
tensorflow>=2.11.0
```

Or install all at once:

```bash
pip install yfinance pandas numpy matplotlib seaborn plotly scikit-learn tensorflow
```

---

## 🛠️ Pipeline Walkthrough

### Step 1 — Data Collection

```python
import yfinance as yf

TICKER     = 'AAPL'          # Change to any valid ticker
START_DATE = '2018-01-01'
END_DATE   = '2024-12-31'

raw_df = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
```

### Step 2 — Feature Engineering

Over 25 technical indicators are computed from raw OHLCV data:

```python
def engineer_features(df):
    d = df.copy()

    # Trend indicators
    d['MA_7']  = d['Close'].rolling(window=7).mean()
    d['MA_21'] = d['Close'].rolling(window=21).mean()
    d['MA_50'] = d['Close'].rolling(window=50).mean()
    d['EMA_12'] = d['Close'].ewm(span=12, adjust=False).mean()
    d['EMA_26'] = d['Close'].ewm(span=26, adjust=False).mean()

    # MACD
    d['MACD']        = d['EMA_12'] - d['EMA_26']
    d['MACD_Signal'] = d['MACD'].ewm(span=9, adjust=False).mean()

    # RSI (14-day)
    delta = d['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # Bollinger Bands
    bb_mid = d['Close'].rolling(20).mean()
    bb_std = d['Close'].rolling(20).std()
    d['BB_Upper'] = bb_mid + 2 * bb_std
    d['BB_Lower'] = bb_mid - 2 * bb_std

    # ATR (Average True Range)
    hl  = d['High'] - d['Low']
    hc  = (d['High'] - d['Close'].shift()).abs()
    lc  = (d['Low']  - d['Close'].shift()).abs()
    d['ATR'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    # Lag features (price memory)
    for lag in [1, 2, 3, 5, 10]:
        d[f'Close_Lag_{lag}'] = d['Close'].shift(lag)

    return d
```

### Step 3 — Preprocessing

```python
from sklearn.preprocessing import MinMaxScaler

# CRITICAL: Never shuffle time-series data
split_idx      = int(len(X) * 0.80)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Fit scaler ONLY on training data — prevents data leakage
scaler_X = MinMaxScaler().fit(X_train)
X_train_scaled = scaler_X.transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)

# Build 3D sequences for LSTM: (samples, timesteps=60, features)
def create_sequences(X, y, look_back=60):
    return (np.array([X[i-look_back:i] for i in range(look_back, len(X))]),
            np.array([y[i]             for i in range(look_back, len(X))]))
```

### Step 4 — LSTM Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(60, n_features)),
    Dropout(0.2), BatchNormalization(),
    LSTM(64,  return_sequences=False),
    Dropout(0.2), BatchNormalization(),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(1,  activation='linear'),
])

model.compile(optimizer='adam', loss='huber', metrics=['mae'])
```

---

## 📊 Results

| Model | MAE (USD) | RMSE (USD) | MAPE (%) | R² Score |
|---|---|---|---|---|
| Linear Regression | ~2.10 | ~3.15 | ~1.20% | ~0.9870 |
| Random Forest | ~1.45 | ~2.20 | ~0.82% | ~0.9940 |
| **LSTM** ✅ | **~1.12** | **~1.68** | **~0.63%** | **~0.9974** |

> Exact values vary slightly per run due to TensorFlow's stochastic training. Model ranking is consistent.

### Key Findings

- **LSTM outperforms** both classical models on all metrics, confirming the advantage of sequential memory for time-series prediction
- **Random Forest feature importance** reveals that `Close_Lag_1`, `Close_Lag_2`, and `MA_7` are the most predictive signals
- **Linear Regression** still achieves R² > 0.98 — demonstrates that much of price variance is explained by recent price history alone
- **30-day forecast** generates day-by-day predicted prices with a ±2% uncertainty band

---

## 📐 Technical Indicators Reference

| Indicator | Formula / Method | Signal Type |
|---|---|---|
| `MA_7 / MA_21 / MA_50` | Rolling mean over 7, 21, 50 days | Trend |
| `EMA_12 / EMA_26` | Exponential weighted mean | Trend |
| `MACD` | EMA_12 − EMA_26 | Momentum |
| `RSI` | 100 − 100/(1 + avg_gain/avg_loss) | Momentum |
| `BB_Upper / BB_Lower` | MA_20 ± 2 × std_20 | Volatility |
| `ATR` | Mean of True Range over 14 days | Volatility |
| `OBV` | Cumulative ±Volume on up/down days | Volume |
| `Close_Lag_1..10` | Shifted close prices | Memory |
| `Return` | Close.pct_change() | Derived |
| `Volatility_7d` | Rolling 7-day std of returns | Derived |

---

## 🖼️ Sample Visualizations

The notebook produces the following charts:

1. **Historical OHLC price with daily range band**
2. **Trading volume + 30-day rolling average**
3. **Daily returns distribution (histogram)**
4. **Feature correlation heatmap**
5. **Technical indicator panel** (MA, Bollinger Bands, RSI, MACD)
6. **Linear Regression** — predictions vs actual + scatter plot
7. **Random Forest** — predictions + feature importance bar chart
8. **LSTM training history** — loss and MAE curves
9. **All models comparison** — single overlay chart
10. **30-day forecast** with uncertainty band

---

## ⚙️ Configuration

All key parameters are defined at the top of the notebook for easy modification:

```python
TICKER        = 'AAPL'       # Any valid Yahoo Finance ticker (TSLA, MSFT, GOOGL...)
START_DATE    = '2018-01-01'
END_DATE      = '2024-12-31'
TEST_RATIO    = 0.20         # Train/test split (80/20)
LOOK_BACK     = 60           # LSTM input window in trading days
FORECAST_DAYS = 30           # Days to forecast into the future
```

---

## 📁 Internship Report

A complete professional report documenting the full project is included in the `docs/` folder:

- **PDF** — `docs/Internship_Report.pdf` (submission-ready)
- **Word** — `docs/Internship_Report.docx` (editable)

The report covers: project overview, methodology, feature engineering, model architectures, results, code walkthrough, challenges, and learning outcomes.

---

## ⚠️ Disclaimer

This project is developed **exclusively for educational and internship purposes**. All predictions are for demonstration only and do **not** constitute financial advice. Past stock performance does not guarantee future results. Never make investment decisions based solely on machine learning model outputs. Always consult a qualified financial advisor.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Muhammad Junaid Asim**
Data Science Intern

- 📧 Email: [Junaidasim899@gmail.com](mailto:Junaidasim899@gmail.com)
- 📱 Phone: +92 3107974444
- 🐙 GitHub: [@junaidasim899](https://github.com/junaidasim899)

---

<div align="center">
  <sub>Built with Python, TensorFlow, and a lot of market data ☕</sub>
</div>
