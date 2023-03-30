import datetime as dt
from util import get_data
import pandas as pd
import indicators
import matplotlib.pyplot as plt
import TheoreticallyOptimalStrategy
import marketsimcode
import numpy as np

def calDailyReturn(price):
    dailyReturn = price.copy()
    dailyReturn[1:] = round((price[1:] / price[:-1].values), 6)- 1 # .values is needed!
    dailyReturn = dailyReturn[1:]
    return dailyReturn

def calCumulativeReturn(portVal):
    return (portVal.iloc[-1, 0] / portVal.iloc[0, 0]) - 1.0

def calAverageDailyReturn(dailyReturn):
    return dailyReturn.mean()

def calStdDailyReturn(dailyReturn):
    return dailyReturn.std()

def plot(df, title):
    fig, ax = plt.subplots()
    ax.set(title=title
           , xlabel='Date'
           , ylabel='Index value and price')
    ax.plot(df)
    plt.legend(df.columns)
    plt.savefig(title + ".png")

def normolize_plot(df, title, normalized_data):
    fig, ax = plt.subplots()
    df["normalized_price"] = normalized_data
    ax.set(title=title
           , xlabel='Date'
           , ylabel='Index value and normalized price')
    ax.plot(df)
    plt.legend(df.columns)
    plt.savefig(title + ".png")

def test():
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    symbol = "JPM"
    data = get_data([symbol], pd.date_range(sd, ed))
    data = data.drop(columns="SPY")
    normalized_data = data / data.iloc[0]

    #EMA
    ema = indicators.exponential_moving_average(data, 10)
    plot(ema, "Exponential Moving Average")

    #bolinger band
    bb, answer = indicators.bollinger_bound(data, 20)
    plot(bb, "bolinger band")

    #RSI
    rsi = indicators.relative_strength_indicator(data)
    rsi = rsi.drop(columns=symbol)
    plot(rsi, "relative_strength_indicator")

    #rate_of_change
    roc = indicators.rate_of_change(data, 12)
    roc = roc.drop(columns=symbol)
    normolize_plot(roc, "Momentum or Rate of Change", normalized_data)

    #MACD
    macd = indicators.MACD(data, 12, 26)
    macd = macd.drop(columns=symbol)
    normolize_plot(macd, "MACD", normalized_data)


    result = TheoreticallyOptimalStrategy.testPolicy(symbol=symbol,sd=sd, ed=ed, sv=100000)
    orders = pd.DataFrame(index=result.index.values, columns=["Symbol", "Order", "Shares"])
    orders["Symbol"] = symbol
    orders["Order"] = np.select([result >= 0, result < 0], ["BUY", "SELL"])
    orders["Shares"] = abs(result)
    value = marketsimcode.compute_portvals(orders_df=orders, start_val=100000, impact=0.0, commission=0.0)
    normalized_value = value / value.iloc[0]

    bench_orders = pd.DataFrame(index=result.index.values, columns=["Symbol", "Order", "Shares"])
    bench_orders["Symbol"] = symbol
    bench_orders["Order"] = "BUY"
    bench_orders["Shares"] = 0.0
    bench_orders.iloc[0, 2] = 1000.0
    bench_value = marketsimcode.compute_portvals(bench_orders, start_val=100000, impact=0.0, commission=0.0)
    normalized_bench_value = bench_value / bench_value.iloc[0]

    fig, ax = plt.subplots()
    ax.set(title="Tos and benchmark"
           , xlabel='Date'
           , ylabel='protfolio')
    ax.plot(normalized_value, 'r')
    ax.plot(normalized_bench_value, 'purple')
    plt.legend(["Tos", "benchmark"])
    plt.savefig("TosVSBenchmark" + ".png")

    dr_b = round(normalized_bench_value[1:] / normalized_bench_value[:-1].values - 1, 6)
    dr_v = round(normalized_value[1:] / normalized_value[:-1].values - 1, 6)

    # calculate cumulative returns
    cr_b = (normalized_bench_value.iloc[-1, 0] / normalized_bench_value.iloc[0, 0]) - 1
    cr_v = (normalized_value.iloc[-1, 0] / normalized_value.iloc[0, 0]) - 1

    # mean daily return
    mdr_b = dr_b.mean()
    mdr_v = dr_v.mean()

    # std daily return
    std_b = dr_b.std()
    std_v = dr_v.std()

    with open("p6_results.txt", "w") as f:
        f.write("cumulative return of tos:" + str(format(cr_v, '.6f')) + '\r')
        f.write("average daily return of tos:" + str(format(mdr_v[0], '.6f')) + '\n')
        f.write("standard deviation of daily return of tos:" + str(format(std_v[0], '.6f')) + '\n')
        f.write("cumulative return of bench:" + str(format(cr_b, '.6f')) + '\n')
        f.write("average daily return of bench:" + str(format(mdr_b[0], '.6f')) + '\n')
        f.write("standard deviation of daily return of bench:" + str(format(std_b[0], '.6f')) + '\n')
        f.close()

def author():
    return 'czhang669'


if __name__ == "__main__":
    test()