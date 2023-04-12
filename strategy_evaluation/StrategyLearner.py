""""""  		  	   		  		 			  		 			     			  	 
"""  		  	   		  		 			  		 			     			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		  		 			  		 			     			  	 
All Rights Reserved  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  		 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		  		 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  		 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 			  		 			     			  	 
or edited.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  		 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		  		 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 			  		 			     			  	 
GT honor code violation.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Student Name: Tucker Balch (replace with your name)  		  	   		  		 			  		 			     			  	 
GT User ID: tb34 (replace with your User ID)  		  	   		  		 			  		 			     			  	 
GT ID: 900897987 (replace with your GT ID)  		  	   		  		 			  		 			     			  	 
"""  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
import datetime as dt  		  	   		  		 			  		 			     			  	 
import random  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
import pandas as pd  		  	   		  		 			  		 			     			  	 
import BagLearner as bl
import RTLearner as rl
import numpy as np
import indicators as indicator
from util import get_data

def author():
    return 'czhang669'

class StrategyLearner(object):  		  	   		  		 			  		 			     			  	 
    """  		  	   		  		 			  		 			     			  	 
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 			  		 			     			  	 
        If verbose = False your code should not generate ANY output.  		  	   		  		 			  		 			     			  	 
    :type verbose: bool  		  	   		  		 			  		 			     			  	 
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		  		 			  		 			     			  	 
    :type impact: float  		  	   		  		 			  		 			     			  	 
    :param commission: The commission amount charged, defaults to 0.0  		  	   		  		 			  		 			     			  	 
    :type commission: float  		  	   		  		 			  		 			     			  	 
    """  		  	   		  		 			  		 			     			  	 
    # constructor  		  	   		  		 			  		 			     			  	 
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        Constructor method  		  	   		  		 			  		 			     			  	 
        """
        self.symbol = None
        self.orders = None
        self.verbose = verbose
        self.impact = impact  		  	   		  		 			  		 			     			  	 
        self.commission = commission
        self.learner = bl.BagLearner(learner=rl.RTLearner, kwargs={"leaf_size":5}, bags=20)
        self.window_size = 12
        self.impact = impact
        self.min_return = 0.02# used to generate y training data
    def author(self):
        return 'czhang669'

    def generate_value_x(self, data, symbol, sd, ed, set_window=False):
        #rsi
        rsi = indicator.relative_strength_indicator(data[symbol], window_size=self.window_size)
        #rsi.columns = ['RSI']
        # EMA
        #ema = indicator.exponential_moving_average(data[symbol], window_size=self.window_size)
        #ema.columns = ['ema']
        BB = indicator.bollinger_bound(data[symbol], window_size=self.window_size)
        #BB.columns = ['BB']
        # Momentum
        momentum = indicator.rate_of_change(data[symbol], window_size=self.window_size)
        #momentum.columns = ["Momentum"]
        data_x = pd.concat((rsi, BB, momentum), axis=1)
        data_x.columns = ['rsi', 'BB', 'Momentum']
        data_x.fillna(0, inplace=True)
        if set_window:
            data_x = data_x[self.window_size:-self.window_size]
        return data_x

    def generate_value_y(self, data, symbol, sd, ed):
        data_y = np.zeros(data.shape[0] - self.window_size)
        val = data[symbol]
        return_val = val.values[self.window_size:]/val.values[:-self.window_size] - 1
        for i in range(len(data) - self.window_size):
            if return_val[i] > self.min_return + self.impact:
                data_y[i] = 1
            elif return_val[i] < -self.min_return - self.impact:
                data_y[i] = -1
            else:
                data_y[i] = 0
        data_y = data_y[self.window_size:]
        return data_y


  		  	   		  		 			  		 			     			  	 
    # this method should create a QLearner, and train it for trading  		  	   		  		 			  		 			     			  	 
    def add_evidence(  		  	   		  		 			  		 			     			  	 
        self,  		  	   		  		 			  		 			     			  	 
        symbol="IBM",  		  	   		  		 			  		 			     			  	 
        sd=dt.datetime(2008, 1, 1),  		  	   		  		 			  		 			     			  	 
        ed=dt.datetime(2009, 1, 1),  		  	   		  		 			  		 			     			  	 
        sv=10000,  		  	   		  		 			  		 			     			  	 
    ):  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        Trains your strategy learner over a given time frame.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
        :param symbol: The stock symbol to train on  		  	   		  		 			  		 			     			  	 
        :type symbol: str  		  	   		  		 			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 			  		 			     			  	 
        :type sd: datetime  		  	   		  		 			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 			  		 			     			  	 
        :type ed: datetime  		  	   		  		 			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		  		 			  		 			     			  	 
        :type sv: int  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
        # add your code to do learning here
        data = get_data([symbol], pd.date_range(sd, ed))
        data.drop('SPY', axis=1, inplace=True)
        x_train = self.generate_value_x(data, symbol, sd, ed, set_window=True)
        y_train = self.generate_value_y(data, symbol, sd, ed)
        self.learner.add_evidence(x_train.values, y_train)
  		  	   		  		 			  		 			     			  	 
    # this method should use the existing policy and test it against new data  		  	   		  		 			  		 			     			  	 
    def testPolicy(  		  	   		  		 			  		 			     			  	 
        self,  		  	   		  		 			  		 			     			  	 
        symbol="IBM",  		  	   		  		 			  		 			     			  	 
        sd=dt.datetime(2009, 1, 1),  		  	   		  		 			  		 			     			  	 
        ed=dt.datetime(2010, 1, 1),  		  	   		  		 			  		 			     			  	 
        sv=10000,  		  	   		  		 			  		 			     			  	 
    ):  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        Tests your learner using data outside of the training data  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
        :param symbol: The stock symbol that you trained on on  		  	   		  		 			  		 			     			  	 
        :type symbol: str  		  	   		  		 			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 			  		 			     			  	 
        :type sd: datetime  		  	   		  		 			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 			  		 			     			  	 
        :type ed: datetime  		  	   		  		 			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		  		 			  		 			     			  	 
        :type sv: int  		  	   		  		 			  		 			     			  	 
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		  		 			  		 			     			  	 
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		  		 			  		 			     			  	 
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		  		 			  		 			     			  	 
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		  		 			  		 			     			  	 
        :rtype: pandas.DataFrame  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
        # here we build a fake set of trades  		  	   		  		 			  		 			     			  	 
        # your code should return the same sort of data  		  	   		  		 			  		 			     			  	 
        data = get_data([symbol], pd.date_range(sd, ed))
        data.drop('SPY', axis=1, inplace=True)
        test_x = self.generate_value_x(data, symbol,sd, ed)
        test_y = self.learner.query(test_x.values)

        orders = []
        shares = []
        holding = 0
        for i in range (test_y.shape[0]):
            if test_y[i] > 0:
                orders.append('BUY')
                if holding == 0:
                    shares.append(1000)
                elif holding == -1000:
                    shares.append(2000)
                elif holding == 1000:
                    shares.append(0)
                holding = 1000
            elif test_y[i] < 0:
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
        self.symbol = symbol
        trades = pd.DataFrame(columns=["Shares"], index=test_x.index.values)
        trades["Shares"] = shares
        return trades

    def generate_orders(self, trades):
        orders = pd.DataFrame(columns=["Shares", "Symbol", "Order"], index=trades.index.values)
        orders["Symbol"] = self.symbol
        orders["Order"] = self.orders
        orders["Shares"] = trades

        return orders

  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
if __name__ == "__main__":  		  	   		  		 			  		 			     			  	 
    print("One does not simply think up a strategy")  		  	   		  		 			  		 			     			  	 
