import pandas as pd
import numpy as np

def strategy(df):
    """
    Simple moving average crossover strategy with RSI, ATR based stop loss and take profit.
    
    Args: df - pandas DataFrame with columns: open, high, low, close, volume
          Index is DatetimeIndex, daily frequency, at least 2 years of data.
    
    Returns: (trades_list, final_equity)
        trades_list: list of float (percent returns per trade)
        final_equity: float (starting from 100000)
    """
    trades = []
    equity = 100000
    position = None
    
    if df is None or len(df) == 0:
        return trades, equity

    if len(df) < 500:
        return trades, equity

    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    
    def rsi(series, period=14):
        delta = series.diff().dropna()
        u = delta * 0
        d = u.copy()
        u[delta > 0] = delta[delta > 0]
        d[delta < 0] = -delta[delta < 0]
        
        u_avg = u.rolling(window=period, min_periods=period).mean()
        d_avg = d.rolling(window=period, min_periods=period).mean()

        rs = u_avg/d_avg
        return 100 - 100/(1+rs)

    df['RSI'] = rsi(df['close'])
    
    def atr(df, period=14):
        df['H-L'] = df['high'] - df['low']
        df['H-PC'] = abs(df['high'] - df['close'].shift(1))
        df['L-PC'] = abs(df['low'] - df['close'].shift(1))
        tr = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        return atr

    df['ATR'] = atr(df)

    warmup_periods = 200
    
    for i in range(warmup_periods, len(df)):
        if position is None:
            if df['SMA_20'][i] > df['SMA_50'][i] and df['SMA_20'][i-1] <= df['SMA_50'][i-1] and df['RSI'][i] < 70:
                # Enter Long Position
                position_size = equity * 0.10
                entry_price = df['close'][i]
                position = {
                    'type': 'long',
                    'size': position_size / entry_price,
                    'entry_price': entry_price,
                    'stop_loss': entry_price - 2 * df['ATR'][i],
                    'take_profit': entry_price + 3 * df['ATR'][i]
                }

        elif position['type'] == 'long':

            if df['close'][i] <= position['stop_loss']:
                # Exit Long Position (Stop Loss Hit)
                exit_price = position['stop_loss']
                trade_return = (exit_price - position['entry_price']) / position['entry_price']
                trades.append(trade_return)
                equity += trade_return * equity * 0.10
                position = None

            elif df['close'][i] >= position['take_profit']:
                # Exit Long Position (Take Profit Hit)
                exit_price = position['take_profit']
                trade_return = (exit_price - position['entry_price']) / position['entry_price']
                trades.append(trade_return)
                equity += trade_return * equity * 0.10
                position = None
    if position is not None:
        if position['type'] == 'long':
            exit_price = df['close'][len(df)-1]

            trade_return = (exit_price - position['entry_price']) / position['entry_price']
            trades.append(trade_return)
            equity += trade_return * equity * 0.10
            position = None


    return trades, equity