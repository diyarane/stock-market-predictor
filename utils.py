import pandas as pd

def smooth_data(series, window=5):
    """Smooth noisy stock data using moving average."""
    return series.rolling(window=window, min_periods=1).mean()
