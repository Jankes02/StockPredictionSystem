from agents.MACDAgent import MACDAgent
import pandas as pd
import os
import matplotlib.pyplot as plt
import utils.backtesting as bt

daily_data_folder = 'data\\daily'
intraday_data_folder = 'data\\5min'
symbols = [
    'ALE', 'ALR', 'BDX', 'CCC', 'CDR',
    'DNP', 'KGH', 'KRU', 'KTY', 'LPP',
    'MBK', 'OPL', 'PCO', 'PEO', 'PGE',
    'PKN', 'PKO', 'PZU', 'SPL', 'ZAB'
]

daily_data_frames = []
for filename in os.listdir(daily_data_folder):
    daily_data_frames.append(pd.read_csv(f'{daily_data_folder}\\{filename}', parse_dates=['Date'], index_col='Date'))

intraday_data_frames = []
for filename in os.listdir(intraday_data_folder):
    intraday_data_frames.append(pd.read_csv(f'{intraday_data_folder}\\{filename}', parse_dates=['Date'], index_col='Date'))

if __name__ == '__main__':
    macd_agent = MACDAgent()
    results = {}
    for i in range(len(daily_data_frames)):
        results[symbols[i]] = bt.backtest_daily(macd_agent, daily_data_frames[i])

    print(results)
    print(f'Average profit margin: {sum(results.values())/len(daily_data_frames)}')