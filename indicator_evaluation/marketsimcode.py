
import datetime as dt
import numpy as np

import pandas as pd
from util import get_data, plot_data


def compute_portvals(
        orders_df,
        start_val=1000000,
        commission=0.00,
        impact=0.00,
):

    #orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
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

