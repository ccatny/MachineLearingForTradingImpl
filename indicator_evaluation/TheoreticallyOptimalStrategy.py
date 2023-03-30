import datetime as dt
from util import get_data, plot_data
import pandas as pd


def set_action(x):
    if x > 0:
        return "buy"
    elif x < 0:
        return "sell"
    else:
        return "wait"


def testPolicy(symbol="APPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2021, 12, 31), sv=100000):
    symbol = [symbol]
    data = get_data(symbol, pd.date_range(sd, ed))
    data.drop('SPY', axis=1, inplace=True)
    diff = data.diff(periods=1, axis=0)
    diff = diff.shift(-1)
    action = pd.DataFrame(index=data.index, columns=data.columns)
    action = diff.applymap(set_action)
    trades = pd.DataFrame(data=0.0, columns=["shares"], index=action.index.values)
    holds = pd.DataFrame(data=0.0, columns=["shares"], index=action.index.values)
    # I got hint from https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/
    # what we do here is: If tomorrow has higher price, then I want to short today. So I will sell 1000 and buy tomorrow
    # vice versa, If tomorrow has lower price, then I want to but 100 today and sell them tomorrow
    # I have to buy today and sell next day because 1.The cumulative value will be the same if I hold it 2.To meet the requirement
    # of I can not trade more than 2000 per day. Besides, In this way, we will never hold more than 1000 or less than 1000 in one day
    for i in range(action.shape[0]):
        for j in range(action.shape[1]):
            if action.iloc[i, j] == 'sell':
                trades.iloc[i, j] = trades.iloc[i, j] - 1000.0
                if i < action.shape[0] - 1:
                    trades.iloc[i + 1, j] = trades.iloc[i+1, j] + 1000.0
            elif action.iloc[i, j] == 'buy':
                trades.iloc[i, j] = trades.iloc[i, j] + 1000.0
                if i < action.shape[0] - 1:
                    trades.iloc[i+1, j] = trades.iloc[i+1, j] - 1000.0
            holds.iloc[i, j] = holds.iloc[i-1, j] + trades.iloc[i, j] if i > 0 else 0
    return trades

def author():
    return 'czhang669'


if __name__ == "__main__":
    testPolicy(symbol="JPM")
