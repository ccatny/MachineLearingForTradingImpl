from experiment1 import run_exp1
from util import get_data
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import ManualStrategy as ms
import StrategyLearner as sl
from experiment2 import run_exp2
import marketsimcode as sim
def author():
    return 'czhang669'

def calDailyReturn(price):
    dailyReturn = price.copy()
    dailyReturn[1:] = (price[1:] / price[:-1].values) - 1 # .values is needed!
    dailyReturn = dailyReturn[1:]
    return dailyReturn

def calCumulativeReturn(dailyReturn):
    return (dailyReturn.iloc[-1] / dailyReturn.iloc[0]) - 1.0

def calAverageDailyReturn(dailyReturn):
    return dailyReturn.mean()

def calStdDailyReturn(dailyReturn):
    return dailyReturn.std()

def cal_stats(portval):
    dailyReturn = calDailyReturn(portval)
    cr = calCumulativeReturn(portval)
    std = calStdDailyReturn(dailyReturn)
    ar = calAverageDailyReturn(dailyReturn)
    return cr, std, ar

def get_benchmark(symbol='AAPL', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000,
                  position=1000, commission=9.95, impact=0.005):
    # commission, impact = 0, 0
    dates = pd.date_range(sd, ed)
    dfPrice = get_data([symbol], dates, True, colname='Adj Close').drop(columns=['SPY'])
    dfPrice.sort_index()
    benchmark_orders = dfPrice * 0
    benchmark_orders.iloc[0][0] = position
    benchmark_portfolio = ms.compute_portvals(benchmark_orders, sd, ed, symbol, sv, commission, impact)
    # generate_plot(benchmark_portfolio, benchmark_portfolio)
    return benchmark_portfolio
#benchmark
def compare(tag):
    data = get_data([symbol], pd.date_range(start_date, end_date))
    data = data.drop(columns="SPY")
    b_order = pd.DataFrame(columns=["Shares", "Symbol", "Order"], index=data.index.values)
    b_order['Shares'] = 0
    b_order.loc[b_order.index[0], 'Shares'] = 1000
    b_order['Order'] = 'BUY'
    b_order['Symbol'] = symbol
    b_port = sim.compute_portvals(b_order, sv, commission, impact)
    normalized_b_port = b_port / b_port.iloc[0]
    data["benchmark"] = normalized_b_port
    # ManualStrategy
    ma = ms.ManualStrategy()
    ms_trades = ma.testPolicy(symbol, start_date, end_date, sv)
    ms_order = ma.generate_orders(ms_trades)
    ms_port = sim.compute_portvals(ms_order, sv, commission, impact)
    normalized_ms_port = ms_port / ms_port.iloc[0]
    data["manual_strategy"] = normalized_ms_port

    '''
    long = ms_order[
        (ms_order['Shares'] > 0) & (ms_order['Order'] == 'BUY') & (ms_order['Shares'] != ms_order['Shares'].shift(1))]
    # long.dropna(inplace=True)
    short = ms_order[
        (ms_order['Shares'] > 0) & (ms_order['Order'] == 'SELL') & (ms_order['Shares'] != ms_order['Shares'].shift(1))]
    '''
    long = ms_order[
        (ms_order['Shares'] > 0) & (ms_order['Order'] == 'BUY') ]
    # long.dropna(inplace=True)
    short = ms_order[
        (ms_order['Shares'] < 0) & (ms_order['Order'] == 'SELL') ]

    fig, ax = plt.subplots()
    ax.set(title="Manual strategy and Benchmark" + tag
           , xlabel='Date'
           , ylabel='protfolio and sginals')
    ax.plot(data['manual_strategy'], 'r')
    ax.plot(data['benchmark'], 'purple')
    for i in (long.index):
        plt.axvline(x=i, color='blue')
    for i in (short.index):
        plt.axvline(x=i, color='black')
    plt.legend(["manual_strategy", "benchmark", "Long", "Short"])
    plt.savefig("Manual strategy and Benchmark " + tag +".png")

    stats = np.zeros((2, 3))
    cr, std, ar = cal_stats(b_port)
    stats[0, 0] = cr
    stats[0, 1] = std
    stats[0, 2] = ar

    cr, std, ar = cal_stats(ms_port)
    stats[1, 0] = cr
    stats[1, 1] = std
    stats[1, 2] = ar
    np.savetxt("manual vs bench " + tag + ".txt", stats, delimiter=',')


random.seed(367)

start_date = dt.datetime(2008, 1, 1)
end_date = dt.datetime(2009, 12, 31)
sv = 100000
commission = 9.95
impact = 0.005
symbol = "JPM"

tag = "in_sample"
compare(tag)
run_exp1(start_date, end_date, tag)


start_date = dt.datetime(2010, 1, 1)
end_date = dt.datetime(2011, 12, 31)
sv = 100000
commission = 9.95
impact = 0.005
symbol = "JPM"

tag = "out_sample"
compare(tag)
run_exp1(start_date, end_date, tag)


run_exp2()