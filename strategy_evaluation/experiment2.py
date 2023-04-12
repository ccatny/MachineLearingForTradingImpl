import datetime as dt
import StrategyLearner as sl
import marketsimcode as sim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import get_data

sv = 100000
commission = 0.0
impacts = [0, 0.005, 0.01, 0.02]
symbol = "JPM"

num_of_trades_list = []
def author():
    return 'czhang669'

def calDailyReturn(price):
    dailyReturn = price.copy()
    dailyReturn[1:] = (price[1:] / price[:-1].values) - 1 # .values is needed!
    dailyReturn = dailyReturn[1:]
    return dailyReturn

def calCumulativeReturn(dailyReturn):
    return (dailyReturn.iloc[-1] / dailyReturn.iloc[0]) - 1.0

def run_exp2(start_date = dt.datetime(2008, 1, 1), end_date = dt.datetime(2009, 12, 31)):
    data = get_data([symbol], pd.date_range(start_date, end_date))
    data = data.drop(columns="SPY")
    for impact in impacts:
        learner = sl.StrategyLearner(impact=impact)
        learner.add_evidence(symbol=symbol, sd=start_date, ed=end_date)
        sl_trades = learner.testPolicy(symbol=symbol, sd=start_date, ed=end_date)
        sl_order = learner.generate_orders(sl_trades)
        sl_port = sim.compute_portvals(sl_order, sv, commission, impact)
        sl_port = sl_port / sl_port.iloc[0]
        data[impact] = sl_port
        valid_order = sl_order[sl_order['Shares'] != 0]
        num_of_trades_list.append(len(valid_order))

    num_stat = pd.DataFrame(data=np.array(num_of_trades_list), index=impacts)

    fig, ax = plt.subplots()
    ax.set(title="experiment 2, number of trades vary with impact"
           , xlabel='impact'
           , xlim=((0, 0.02))
           , ylabel='number of trades')
    ax.plot(num_stat, 'r')
    plt.legend(["number of trades"])
    plt.savefig("experiment 2 number of trades" + ".png")


    fig, ax = plt.subplots()
    ax.set(title="experiment 2, protfolio under different impact"
           , xlabel='Date'
           , ylabel='protfolio')
    for impact in impacts:
        ax.plot(data[impact])
    plt.legend([impacts[0],impacts[1],impacts[2], impacts[2]])
    plt.savefig("experiment 2 protfolio under different impact " + ".png")
