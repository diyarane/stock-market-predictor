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
def fetch_stock_data(symbol, period="1y"):  # Changed from "6mo" to "1y"
    """
    Fetch closing prices for a stock symbol.
    - Alpha Vantage for US stocks
    - Fallback to Yahoo Finance for US + Indian stocks
    Returns: pandas Series of closing prices (sorted ascending by date)
    """
    # Try Alpha Vantage first (US stocks)
    try:
        ts = TimeSeries(key=API_KEY, output_format='pandas')
        data, _ = ts.get_daily(symbol=symbol, outputsize='full')
        if data.empty or '4. close' not in data.columns:
            raise ValueError("Alpha Vantage returned empty/invalid data")
        close = data['4. close'].sort_index()
        return close
    except Exception:
        # Fallback to Yahoo Finance
        data = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=True)
        if data.empty or 'Close' not in data.columns:
            raise ValueError(f"No valid data returned for symbol {symbol}. Check symbol spelling or internet connection.")
        close = data['Close'].squeeze().sort_index()
        return close

# ---------------- Predict Stock Price ----------------
def predict_stock_price(symbol):
    """
    Returns:
    - predicted_price: next day predicted price
    - last_close: last available closing price
    - data_recent: last 120 days of actual prices (pandas Series)
    """
    data = fetch_stock_data(symbol)
    last_close = data.iloc[-1]
    
    # Linear regression on last 120 days
    data_recent = data[-120:]
    X = np.arange(len(data_recent)).reshape(-1, 1)
    y = data_recent.values.reshape(-1, 1)
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
    # Convert last_close to scalar if it's a Series
    if isinstance(last_close, pd.Series):
        last_close = last_close.item()
    else:
        last_close = float(last_close)
    
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
    Saves a plot showing last 120 days of prices and predicted next day
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data_recent.index, data_recent.values, marker='o', markersize=3, label='Actual Price', color='blue', linewidth=2)
    
    # Predicted next day
    next_day = data_recent.index[-1] + pd.Timedelta(days=1)
    plt.scatter(next_day, predicted, color='green', s=150, label='Predicted Price', zorder=5)
    plt.plot([data_recent.index[-1], next_day], [data_recent.values[-1], predicted], linestyle='--', color='green', linewidth=2)
    
    plt.title(f"{symbol} - 120 Day Price Trend & Prediction", fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price ($)", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()

# ---------------- Calculate accuracy ----------------
def calculate_accuracy(symbol, days_back=30):
    """
    Backtest the model by predicting past days and comparing with actual prices.
    Returns accuracy percentage.
    """
    try:
        print(f"   🔍 Starting accuracy calculation for {symbol}...")
        
        # Fetch data for backtesting - 1 year to ensure we have enough
        data = fetch_stock_data(symbol, period="1y")
        
        print(f"   📊 Fetched {len(data)} days of data for backtesting")
        print(f"   📅 Date range: {data.index[0]} to {data.index[-1]}")
        
        if len(data) < 150:  # Need enough data for backtesting (120 + 30 buffer)
            print(f"   ⚠️  Not enough data ({len(data)} days). Need at least 150 days.")
            return None
        
        predictions = []
        actuals = []
        
        # Test on last 'days_back' days
        test_days = min(days_back, len(data) - 120)
        print(f"   🧪 Will test {test_days} predictions")
        
        for i in range(test_days, 0, -1):
            # Use data up to 'i' days ago
            historical_data = data[:-i]
            
            # Get actual price - convert to float properly
            actual_price = data.iloc[-i]
            if isinstance(actual_price, pd.Series):
                actual_price = float(actual_price.item())
            else:
                actual_price = float(actual_price)
            
            # Use last 120 days of historical data for prediction
            data_recent = historical_data[-120:]
            
            if len(data_recent) < 120:
                continue
                
            X = np.arange(len(data_recent)).reshape(-1, 1)
            y = data_recent.values.reshape(-1, 1)
            
            model = LinearRegression()
            model.fit(X, y)
            
            predicted = model.predict(np.array([[len(data_recent)]]))[0][0]
            
            predictions.append(float(predicted))
            actuals.append(actual_price)
        
        print(f"   📈 Total successful predictions: {len(predictions)}")
        
        # Calculate accuracy using MAPE (Mean Absolute Percentage Error)
        if len(predictions) == 0:
            print(f"   ⚠️  No predictions made")
            return None
            
        errors = [abs((actual - pred) / actual) * 100 for actual, pred in zip(actuals, predictions)]
        mape = np.mean(errors)
        accuracy = 100 - mape  # Convert error to accuracy
        
        calculated_accuracy = max(0, min(100, round(accuracy, 2)))  # Ensure between 0-100
        print(f"   ✅ MAPE: {mape:.2f}%, Final Accuracy: {calculated_accuracy}%")
        
        return calculated_accuracy
        
    except Exception as e:
        print(f"   ❌ Error calculating accuracy: {e}")
        import traceback
        traceback.print_exc()
        return None
