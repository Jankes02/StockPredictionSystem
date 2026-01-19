

def backtest_daily(agent, data):
    signals = []
    for i in range(len(data)):
        window = data.iloc[:i+1]
        sig = agent.signal(window)
        signals.append(sig)

    data['Signal'] = signals

    positions = data['Signal'].replace(0, method='ffill')
    positions.iloc[0] = 0

    returns = data['Close'].pct_change().fillna(0)
    strategy_returns = positions.shift(1) * returns
    equity = (1 + strategy_returns).cumprod()
    return (equity.iloc[-1] - 1) * 100