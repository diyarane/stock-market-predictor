import os
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for macOS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from alpha_vantage.timeseries import TimeSeries
import yfinance as yf

# ---------------- API Key ----------------
API_KEY = 'A1AP3WSVCIITGGT8'

# ---------------- Fetch Stock Data Safely ----------------
def fetch_stock_data(symbol):
    """
    Fetch closing prices for a stock symbol.
    - Alpha Vantage for US stocks
    - Fallback to Yahoo Finance for US + Indian stocks
    Returns: pandas Series of closing prices (sorted ascending by date)
    """
    # Try Alpha Vantage first (US stocks)
    try:
        ts = TimeSeries(key=API_KEY, output_format='pandas')
        data, _ = ts.get_daily(symbol=symbol, outputsize='compact')
        if data.empty or '4. close' not in data.columns:
            raise ValueError("Alpha Vantage returned empty/invalid data")
        close = data['4. close'].sort_index()
        return close
    except Exception:
        # Fallback to Yahoo Finance
        data = yf.download(symbol, period="1mo", interval="1d")
        if data.empty or 'Close' not in data.columns:
            raise ValueError(f"No valid data returned for symbol {symbol}. Check symbol spelling or internet connection.")
        close = data['Close'].sort_index()
        return close

# ---------------- Predict Stock Price ----------------
def predict_stock_price(symbol):
    """
    Returns:
    - predicted_price: next day predicted price
    - last_close: last available closing price
    - data_recent: last 30 days of actual prices (pandas Series)
    """
    data = fetch_stock_data(symbol)
    last_close = data[-1]

    # Linear regression on last 30 days
    data_recent = data[-30:]
    X = np.arange(len(data_recent)).reshape(-1,1)
    y = data_recent.values.reshape(-1,1)
    model = LinearRegression()
    model.fit(X, y)
    predicted = model.predict(np.array([[len(data_recent)]]))[0][0]

    return round(predicted, 2), last_close, data_recent

# ---------------- Buy/Hold/Avoid Suggestion ----------------
def get_buy_suggestion(last_close, predicted):
    """
    Returns:
    - suggestion: Buy/Hold/Avoid string
    - reasoning: Explanation string
    """
    change_percent = ((predicted - last_close) / last_close) * 100
    
    if change_percent >= 2:
        suggestion = "✅ Suggested: Good time to BUY this stock."
        reasoning = f"Predicted price {predicted:.2f} is {change_percent:.2f}% higher than last close {last_close:.2f}, indicating upward momentum."
    elif change_percent <= -2:
        suggestion = "❌ Suggested: Avoid buying right now."
        reasoning = f"Predicted price {predicted:.2f} is {abs(change_percent):.2f}% lower than last close {last_close:.2f}, suggesting a potential downward trend."
    else:
        suggestion = "⏳ Suggested: Hold / Wait."
        reasoning = f"Predicted price {predicted:.2f} is close to last close {last_close:.2f}, indicating minimal expected movement."
    
    return suggestion, reasoning

# ---------------- Save Trend Plot ----------------
def save_plot(symbol, predicted, data_recent, path):
    """
    Saves a plot showing last 30 days of prices and predicted next day
    """
    plt.figure(figsize=(8,4))
    plt.plot(data_recent.index, data_recent.values, marker='o', label='Actual Price', color='blue')

    # Predicted next day
    next_day = data_recent.index[-1] + pd.Timedelta(days=1)
    plt.scatter(next_day, predicted, color='green', s=100, label='Predicted Price')
    plt.plot([data_recent.index[-1], next_day], [data_recent.values[-1], predicted], linestyle='--', color='green')

    plt.title(f"{symbol} Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
