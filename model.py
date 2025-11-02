import os
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for macOS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from alpha_vantage.timeseries import TimeSeries
import yfinance as yf
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Set TensorFlow to only show errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------------- API Key ----------------
API_KEY = 'A1AP3WSVCIITGGT8'

# ---------------- Fetch Stock Data Safely ----------------
def fetch_stock_data(symbol, period="1y"):
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

# ---------------- Create LSTM Dataset ----------------
def create_lstm_dataset(data, lookback=60):
    """
    Create sequences for LSTM training
    lookback: number of previous days to use for prediction
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# ---------------- Build LSTM Model ----------------
def build_lstm_model(lookback=60):
    """
    Build and compile LSTM neural network
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ---------------- Predict Stock Price ----------------
def predict_stock_price(symbol):
    """
    Returns:
    - predicted_price: next day predicted price
    - last_close: last available closing price
    - data_recent: last 120 days of actual prices (pandas Series)
    """
    try:
        print(f"   🧠 Training LSTM model for {symbol}...")
        
        # Fetch data
        data = fetch_stock_data(symbol)
        last_close = data.iloc[-1]
        
        # Use last 120 days for display (but train on more data)
        data_recent = data[-120:]
        
        # Prepare data for LSTM
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
        
        # Create sequences (use 60-day lookback)
        lookback = 60
        X, y = create_lstm_dataset(scaled_data, lookback)
        
        if len(X) == 0:
            print("   ⚠️  Not enough data for LSTM, falling back to simple prediction")
            # Fallback to last price if not enough data
            return round(float(last_close), 2), last_close, data_recent, None
        
        # Split data (use 80% for training)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Reshape for LSTM [samples, time steps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Build and train model
        model = build_lstm_model(lookback)
        
        print(f"   📚 Training on {len(X_train)} samples...")
        history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0, validation_split=0.1)
        
        # Evaluate model on test set
        train_loss = model.evaluate(X_train, y_train, verbose=0)
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        
        # Calculate accuracy from loss (lower loss = higher accuracy)
        # MSE to percentage accuracy estimation
        train_mse_percent = (train_loss ** 0.5) / np.mean(y_train) * 100
        test_mse_percent = (test_loss ** 0.5) / np.mean(y_test) * 100
        accuracy = max(0, min(100, 100 - test_mse_percent))
        
        print(f"   ✅ Model trained! Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        print(f"   📊 Estimated Accuracy: {accuracy:.2f}%")
        
        # Predict next day
        last_sequence = scaled_data[-lookback:].reshape((1, lookback, 1))
        predicted_scaled = model.predict(last_sequence, verbose=0)
        predicted = scaler.inverse_transform(predicted_scaled)[0][0]
        
        return round(float(predicted), 2), last_close, data_recent, round(accuracy, 2)
        
    except Exception as e:
        print(f"   ❌ Error in predict_stock_price: {e}")
        import traceback
        traceback.print_exc()
        # Return with None accuracy on error
        data = fetch_stock_data(symbol)
        last_close = data.iloc[-1]
        data_recent = data[-120:]
        return round(float(last_close), 2), last_close, data_recent, None

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
        reasoning = f"LSTM model predicts price {predicted:.2f} is {change_percent:.2f}% higher than last close {last_close:.2f}, indicating upward momentum."
    elif change_percent <= -2:
        suggestion = "❌ Suggested: Avoid buying right now."
        reasoning = f"LSTM model predicts price {predicted:.2f} is {abs(change_percent):.2f}% lower than last close {last_close:.2f}, suggesting a potential downward trend."
    else:
        suggestion = "⏳ Suggested: Hold / Wait."
        reasoning = f"LSTM model predicts price {predicted:.2f} is close to last close {last_close:.2f}, indicating minimal expected movement."
    
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
    plt.scatter(next_day, predicted, color='green', s=150, label='LSTM Predicted Price', zorder=5)
    plt.plot([data_recent.index[-1], next_day], [data_recent.values[-1], predicted], linestyle='--', color='green', linewidth=2)
    
    plt.title(f"{symbol} - LSTM Neural Network Prediction", fontsize=16, fontweight='bold')
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
    Backtest the LSTM model by predicting past days and comparing with actual prices.
    Returns accuracy percentage.
    """
    try:
        print(f"   🔍 Starting LSTM accuracy calculation for {symbol}...")
        
        # Fetch data for backtesting
        data = fetch_stock_data(symbol, period="1y")
        
        print(f"   📊 Fetched {len(data)} days of data for backtesting")
        print(f"   📅 Date range: {data.index[0]} to {data.index[-1]}")
        
        if len(data) < 150:
            print(f"   ⚠️  Not enough data ({len(data)} days). Need at least 150 days.")
            return None
        
        predictions = []
        actuals = []
        
        # Prepare scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
        
        lookback = 60
        test_days = min(days_back, len(data) - 150)
        print(f"   🧪 Will test {test_days} LSTM predictions")
        
        for i in range(test_days, 0, -1):
            # Use data up to 'i' days ago
            historical_data = scaled_data[:-i]
            
            # Get actual price
            actual_price = data.iloc[-i]
            if isinstance(actual_price, pd.Series):
                actual_price = float(actual_price.item())
            else:
                actual_price = float(actual_price)
            
            # Need enough data for training
            if len(historical_data) < lookback + 50:
                continue
            
            # Create sequences
            X, y = create_lstm_dataset(historical_data, lookback)
            
            if len(X) < 50:
                continue
            
            # Use last 80% for training
            train_size = int(len(X) * 0.8)
            X_train = X[:train_size]
            y_train = y[:train_size]
            
            # Reshape
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            
            # Build and train model
            model = build_lstm_model(lookback)
            model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
            
            # Predict
            last_sequence = historical_data[-lookback:].reshape((1, lookback, 1))
            predicted_scaled = model.predict(last_sequence, verbose=0)
            predicted = scaler.inverse_transform(predicted_scaled)[0][0]
            
            predictions.append(float(predicted))
            actuals.append(actual_price)
        
        print(f"   📈 Total successful predictions: {len(predictions)}")
        
        # Calculate accuracy using MAPE
        if len(predictions) == 0:
            print(f"   ⚠️  No predictions made")
            return None
            
        errors = [abs((actual - pred) / actual) * 100 for actual, pred in zip(actuals, predictions)]
        mape = np.mean(errors)
        accuracy = 100 - mape
        
        calculated_accuracy = max(0, min(100, round(accuracy, 2)))
        print(f"   ✅ MAPE: {mape:.2f}%, Final Accuracy: {calculated_accuracy}%")
        
        return calculated_accuracy
        
    except Exception as e:
        print(f"   ❌ Error calculating accuracy: {e}")
        import traceback
        traceback.print_exc()
        return None