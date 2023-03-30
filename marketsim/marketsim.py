""""""
"""MC2-P1: Market simulator.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
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
import os

import numpy as np

import pandas as pd
from util import get_data, plot_data


def compute_portvals(
        orders_file="./orders/orders-01.csv",
        start_val=1000000,
        commission=9.95,
        impact=0.005,
):
    """
    Computes the portfolio values.

    :param orders_file: Path of the order file or the file object
    :type orders_file: str or file object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here
    orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    #orders_df.index = pd.to_datetime(list(orders_df.index.values))
    orders_df.sort_index(inplace=True)
    start_date = min(orders_df.index)
    end_date = max(orders_df.index)
    stock_list = list(set(orders_df["Symbol"]))
    prices_df = get_data(stock_list, pd.date_range(start_date, end_date))
    # Note, it is a little tricky here. There is a var in get_data named addSPY. However, if we select False, it will
    # return all the data even this day there is no SPY value, which means there is no trade this day.
    # Also, here the date are in different format between order and price. Higher level python will convert it but not for 3.6

    valid_dates = prices_df.index
    prices_df.drop('SPY', axis=1, inplace=True)
    prices_df["Cash"] = 1.0
    trades_df = pd.DataFrame(data=0.0, columns=prices_df.columns.values, index=prices_df.index.values)
    holding_df = pd.DataFrame(data=0.0, columns=prices_df.columns.values, index=prices_df.index.values)
    for i in range(0, orders_df.shape[0]):
        buy = True
        date = orders_df.index[i]
        if date not in valid_dates:
            continue
        symbol = orders_df.iloc[[i]]['Symbol'][0]
        operation = orders_df.iloc[[i]]['Order'][0]
        shareNum = orders_df.iloc[[i]]['Shares'][0]
        if operation != "BUY":
            buy = False
        change_shareNum = shareNum if buy else -1 * shareNum
        trades_df.at[date, symbol] = trades_df.at[date, symbol] + change_shareNum
        change_cash = prices_df.at[date, symbol] * change_shareNum * -1
        trades_df.at[date, 'Cash'] = trades_df.at[date, 'Cash'] + change_cash - commission - abs(change_cash * impact)

    holding_df.iloc[[0]] = trades_df.iloc[[0]]
    holding_df.at[start_date, 'Cash'] = holding_df.at[start_date, 'Cash'] + start_val
    for i in range(1, holding_df.shape[0]):
        holding_df.iloc[[i]] = holding_df.iloc[[i - 1]].values + trades_df.iloc[[i]]
    portvals = pd.DataFrame(index=holding_df.index)
    portvals['total_asset'] = 0.0
    for i, row in holding_df.iterrows():
        for symbol in holding_df.columns:
            if symbol == 'Cash':
                portvals.at[i, 'total_asset'] = portvals.at[i, 'total_asset'] + row[symbol]
            else:
                portvals.at[i, 'total_asset'] = portvals.at[i, 'total_asset'] + row[symbol] * prices_df.at[i, symbol]
    return portvals


def author():
    return 'czhang669'


def test_code():
    """
    Helper function to test code
    """
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-01.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv, commission=0.0, impact=0.0)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"


if __name__ == "__main__":
    test_code()