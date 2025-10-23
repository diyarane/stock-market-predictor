import matplotlib
matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from alpha_vantage.timeseries import TimeSeries
from utils import smooth_data

def predict_stock(symbol):
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, _ = ts.get_daily(symbol=symbol, outputsize='compact')

    data = data.rename(columns={
        '1. open': 'open',
        '2. high': 'high',
        '3. low': 'low',
        '4. close': 'close',
        '5. volume': 'volume'
    })

    data['close_smooth'] = smooth_data(data['close'])

    data['day'] = np.arange(len(data))
    X = data[['day']]
    y = data['close_smooth']

    model = LinearRegression()
    model.fit(X, y)

    next_day = np.array([[len(data) + 1]])
    prediction = model.predict(next_day)[0]

    # Plot actual + predicted
    plt.figure(figsize=(8, 4))
    plt.plot(data['day'], data['close_smooth'], label='Smoothed Close Prices')
    plt.scatter(len(data) + 1, prediction, color='red', label='Predicted Next Day')
    plt.title(f'{symbol} Stock Prediction')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()

    plot_path = os.path.join('static', f'{symbol}_plot.png')
    plt.savefig(plot_path)
    plt.close()

    return round(prediction, 2), plot_path
