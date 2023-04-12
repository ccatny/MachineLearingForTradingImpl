import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import ManualStrategy as ms
from util import get_data
import marketsimcode as sim
import StrategyLearner as sl
sv = 100000
commission = 9.95
impact = 0.005
symbol = "JPM"
def author():
    return 'czhang669'

def run_exp1(start_date = dt.datetime(2008, 1, 1), end_date = dt.datetime(2009, 12, 31), tag="in sample"):
    data = 1000 * get_data([symbol], pd.date_range(start_date, end_date))
    data = data.drop(columns="SPY")
    b_order = pd.DataFrame(columns=["Shares", "Symbol", "Order"], index=data.index.values)
    b_order['Shares'] = 0
    b_order.loc[b_order.index[0], 'Shares'] = 1000
    b_order['Order'] = 'BUY'
    b_order['Symbol'] = symbol
    b_port = sim.compute_portvals(b_order, sv, commission, impact)
    b_port = b_port / b_port.iloc[0]
    data["benchmark"] = b_port

    ma = ms.ManualStrategy()
    ms_trades = ma.testPolicy(symbol, start_date, end_date, sv)
    ms_order = ma.generate_orders(ms_trades)
    ms_port = sim.compute_portvals(ms_order, sv, commission, impact)
    ms_port = ms_port / ms_port.iloc[0]
    data["manual_strategy"] = ms_port

    learner = sl.StrategyLearner(commission=commission, impact=impact)
    learner.add_evidence(symbol=symbol, sd=start_date, ed=end_date, sv =sv)
    sl_trades = learner.testPolicy(symbol, start_date, end_date)
    sl_order = learner.generate_orders(sl_trades)
    sl_port = sim.compute_portvals(sl_order, sv, commission, impact)
    sl_port = sl_port / sl_port.iloc[0]
    data["strategy_learner"] = sl_port

    fig, ax = plt.subplots()
    ax.set(title="experiment 1, compare of strategies" + tag
           , xlabel='Date'
           , ylabel='protfolio')
    ax.plot(data['manual_strategy'], 'r')
    ax.plot(data['strategy_learner'], 'b')
    ax.plot(data['benchmark'], 'purple')
    plt.legend(["manual_strategy", "strategy_learner", "benchmark"])
    plt.savefig("experiment 1 " + tag + ".png")








