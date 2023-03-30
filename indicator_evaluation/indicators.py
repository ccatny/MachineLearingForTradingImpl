import pandas as pd
import numpy as np

def preprocess(df_prices):
    df = df_prices.copy()
    return df

def exponential_moving_average(df, window_size=10):
    df_price = preprocess(df)
    ema = df_price.ewm(span=window_size, adjust=False).mean()
    df_price["ema"] = ema
    return df_price

def bollinger_bound(df, window_size=20):
    df_prices = preprocess(df)
    sma = df_prices.rolling(window=window_size).mean()
    std = df_prices.rolling(window=window_size).std()
    df_prices["upper"] = sma + 2 * std
    df_prices["lower"] = sma - 2 * std
    result = (df_prices - sma) / 2 * std
    return df_prices, result

def relative_strength_indicator(df, window_size=14):
    df_price = preprocess(df)
    rets = df_price.pct_change(periods=1)
    gain = rets.copy()
    gain[gain < 0] = 0
    loss = rets.copy()
    loss[loss > 0] = 0
    loss = np.abs(loss)
    avg_gain = gain.rolling(window=window_size).mean()
    avg_loss = np.abs(loss.rolling(window=window_size).mean())
    rsi = 1 - (1 / (1 + (avg_gain / avg_loss)))
    df_price["rsi"] = rsi
    df_price["upper"] = 0.7
    df_price["lower"] = 0.3
    return df_price

def rate_of_change(df, window_size=12):
    df_price = preprocess(df)
    df_price["roc"] = (df_price - df_price.shift(window_size)) / df_price.shift(window_size)
    return df_price

def MACD(df, short_window=12, long_window=26):
    df_price = preprocess(df)
    short = df_price.ewm(span=short_window).mean()
    long = df_price.ewm(span=long_window).mean()
    dif = short - long
    dem = dif.ewm(span=9).mean()
    df_price["dif"] = dif
    df_price["dem"] = dem
    return df_price


def author():
    return 'czhang669'