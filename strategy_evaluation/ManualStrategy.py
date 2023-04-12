import datetime as dt
import indicators as indicator
import pandas as pd

from util import get_data

window_size = 20

def author():
    return 'czhang669'

class ManualStrategy:

    def author(self):
        return 'czhang669'

    def __init__(self):
        self.orders = None

    def testPolicy(self, symbol='JPM', start_date=dt.datetime(2010, 1, 1), end_date=dt.datetime(2011, 12, 31), sv=100000):
        self.data = get_data([symbol], pd.date_range(start_date, end_date))
        self.data.drop('SPY', axis=1, inplace=True)
        self.symbol = symbol
        #RSI
        self.data["RSI"] = indicator.relative_strength_indicator(self.data[symbol], generate_signal=True)
        #EMA
        #self.data["EMA"] = indicator.exponential_moving_average(self.data[symbol], symbol=symbol, generate_signal=True)
        #
        self.data["BB"] = indicator.bollinger_bound(self.data[symbol], generate_signal=True)
        #Momentum
        self.data["Momentum"] = indicator.rate_of_change(self.data[symbol], generate_signal=True)

        self.action = self.data.RSI + self.data.BB + self.data.Momentum
        self.action.loc[self.action > 1] = 1
        self.action.loc[self.action < -1] = -1


        trades = self.generate_trades()
        return trades




    def generate_trades(self):
        '''
        trades = pd.DataFrame(columns=["shares"], index=self.action.index.values)
        for i in range(trades.shape[0]):
            if self.action[i] != 0 and self.action[i-1] == 0:
                #in this case, we want to
                trades.iloc[i] = self.action.iloc[i] * 1000
            elif self.action.iloc[i] == 0 and self.action.iloc[i-1] != 0:
                #in this case,
                trades.iloc[i] = self.action.iloc[i-1] * -1000
            elif self.action.iloc[i] != self.action.iloc[i-1] and self.action.iloc[i-1] != 0:
                trades.iloc[i] = self.action.iloc[i-1] * -2000
            elif self.action.iloc[i] == self.action.iloc[i-1] and self.action.iloc[i-1] != 0:
                trades.iloc[i] = 0
        '''
        holding = 0
        orders = []
        shares = []
        for i in range(self.action.shape[0]):
            if self.action[i] == 1:
                orders.append('BUY')
                if holding == 0:
                    shares.append(1000)
                elif holding == -1000:
                    shares.append(2000)
                elif holding == 1000:
                    shares.append(0)
                holding = 1000
            elif self.action[i] == -1:
                orders.append('SELL')
                if holding == 0:
                    shares.append(-1000)
                elif holding == 1000:
                    shares.append(-2000)
                elif holding == -1000:
                    shares.append(0)
                holding = -1000
            else:
                orders.append('BUY')
                shares.append(0)

        self.orders = orders
        trades = pd.DataFrame(columns=["Shares"], index=self.action.index.values)
        trades["Shares"] = shares
        return trades

    def generate_orders(self, trades):
        orders = pd.DataFrame(columns=["Shares", "Symbol", "Order"], index=trades.index.values)
        orders["Symbol"] = self.symbol
        orders["Order"] = self.orders
        orders["Shares"] = trades
        return orders


