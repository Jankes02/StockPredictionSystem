from agents.MACD import MACD
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

wig20 = pd.read_csv(
    'stocks\\wig20.csv',
    parse_dates=['Date'],
    index_col='Date'
)

daily_data_frames = []
for filename in os.listdir(daily_data_folder):
    daily_data_frames.append(pd.read_csv(f'{daily_data_folder}\\{filename}', parse_dates=['Date'], index_col='Date'))

intraday_data_frames = []
for filename in os.listdir(intraday_data_folder):
    intraday_data_frames.append(pd.read_csv(f'{intraday_data_folder}\\{filename}', parse_dates=['Date'], index_col='Date'))

if __name__ == '__main__':
    macd_agent = MACD()
    results = {}
    alpha = {}
    
    wig20_return = bt.buy_and_hold(wig20)

    for i in range(len(daily_data_frames)):
        strat_return = bt.backtest_daily(macd_agent, daily_data_frames[i])
        results[symbols[i]] = strat_return
        alpha[symbols[i]] = strat_return - wig20_return

    print('=== STRATEGY RESULTS ===')
    for k, v in results.items():
        print(f'{k}: {v:.2f}%')

    print('\n=== BENCHMARK ===')
    print(f'WIG20 Buy & Hold: {wig20_return:.2f}%')

    print('\n=== ALPHA vs WIG20 ===')
    for k, v in alpha.items():
        print(f'{k}: {v:.2f}%')

    print('\n=== SUMMARY ===')
    print(f'Average strategy return: {sum(results.values()) / len(results):.2f}%')
    print(f'Alpha vs WIG20: {(sum(results.values()) / len(results) - wig20_return):.2f}%')