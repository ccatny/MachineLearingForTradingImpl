import pandas as pd
import numpy as np

def preprocess(df_prices):
    df = df_prices.copy()
    return df

def exponential_moving_average(df, window_size=10, generate_signal=False, symbol='JPM'):
    df_price = preprocess(df)
    ema = df_price.ewm(span=window_size, adjust=False).mean()
    if not generate_signal:
        return ema
    signal = ema.copy()
    signal[df_price < ema] = -1
    signal[df_price > ema] = 1
    signal[df_price == ema] = 0
    return signal


def bollinger_bound(df, window_size=20, generate_signal=False):
    df_prices = preprocess(df)
    sma = df_prices.rolling(window=window_size).mean()
    std = df_prices.rolling(window=window_size).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    result = (df_prices - lower) / (upper - lower)
    if not generate_signal:
        return result
    signal = result.copy() * 0
    signal[result > 0.8] = -1
    signal[result < 0.2] = 1
    return signal


def relative_strength_indicator(df, window_size=14, generate_signal=False):
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
    #rsi_vector = np.zeros(df.shape[0])
    signal = rets.copy()
    if not generate_signal:
        return rsi
    #for i in range(df.shape[0] - len(avg_gain), df.shape[0]):
    for i in range(df.shape[0]):
        if rsi[i-1] < 0.3 and rsi[i] >= 0.3:
            signal[i] = 1
        elif rsi[i-1] > 0.7 and rsi[i] <= 0.7:
            signal[i] = -1
        else:
            signal[i] = 0
    return signal

def rate_of_change(df, window_size=12, generate_signal=False):
    df_price = preprocess(df)
    roc = (df_price - df_price.shift(window_size)) / df_price.shift(window_size)
    if not generate_signal:
        return roc
    signal = roc.copy() * 0
    signal[roc > 0.05] = -1
    signal[roc < -0.05] = 1
    return signal


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