import pandas as pd
import numpy as np

def strategy(df):
    """
    Implements a simple trend-following strategy with stop loss and take profit based on ATR and RSI.
    
    Args: df - pandas DataFrame with columns: open, high, low, close, volume
          Index is DatetimeIndex, daily frequency, at least 2 years of data.
    
    Returns: (trades_list, final_equity)
        trades_list: list of float (percent returns per trade)
        final_equity: float (starting from 100000)
    """
    
    def calculate_atr(df, period=14):
        atr = np.zeros(len(df))
        for i in range(1, len(df)):
            high_low = df['high'].iloc[i] - df['low'].iloc[i]
            high_close_prev = abs(df['high'].iloc[i] - df['close'].iloc[i-1])
            low_close_prev = abs(df['low'].iloc[i] - df['close'].iloc[i-1])
            tr = max(high_low, high_close_prev, low_close_prev)
            atr[i] = tr
        atr = pd.Series(atr).rolling(window=period).mean().to_numpy()
        return atr
        
    def calculate_rsi(df, period=14):
        delta = df['close'].diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        avg_gain = up.rolling(window=period).mean()
        avg_loss = abs(down.rolling(window=period).mean())
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.to_numpy()
    
    trades = []
    equity = 100000
    position = None
    position_size = 0.1
    
    if df is None or len(df) == 0:
        return trades, equity
    
    if len(df) < 500:
        return trades, equity #added check to ensure enough rows of data exists, can't run calculation on insufficient data
    
    atr = calculate_atr(df)
    rsi = calculate_rsi(df)
    
    for i in range(200, len(df)):
        if position is None:
            if rsi[i] > 70:  # Oversold, consider shorting
                stop_loss = df['close'].iloc[i] + 2 * atr[i]
                take_profit = df['close'].iloc[i] - 4 * atr[i]
                
                position_value = equity * position_size
                shares = position_value / df['close'].iloc[i]
                position = {'type': 'short', 'price': df['close'].iloc[i], 'shares': shares, 'stop_loss': stop_loss, 'take_profit': take_profit}
                
            elif rsi[i] < 30: # Oversold, consider longing
                stop_loss = df['close'].iloc[i] - 2 * atr[i]
                take_profit = df['close'].iloc[i] + 4 * atr[i]
                
                position_value = equity * position_size
                shares = position_value / df['close'].iloc[i]
                position = {'type': 'long', 'price': df['close'].iloc[i], 'shares': shares, 'stop_loss': stop_loss, 'take_profit': take_profit}
        
        elif position['type'] == 'long':
            if df['close'].iloc[i] <= position['stop_loss']:
                trade_return = (position['stop_loss'] - position['price']) / position['price'] * -1
                trades.append(trade_return * 100)
                equity *= (1 + trade_return)
                position = None
            elif df['close'].iloc[i] >= position['take_profit']:
                trade_return = (position['take_profit'] - position['price']) / position['price']
                trades.append(trade_return * 100)
                equity *= (1 + trade_return)
                position = None

        elif position['type'] == 'short':
            if df['close'].iloc[i] >= position['stop_loss']:
                trade_return = (position['price'] - position['stop_loss']) / position['price']
                trades.append(trade_return * 100) #Correct
                equity *= (1 + trade_return) #Correct
                position = None
            elif df['close'].iloc[i] <= position['take_profit']:
                trade_return = (position['price'] - position['take_profit']) / position['price']
                trades.append(trade_return * 100) #Correct
                equity *= (1 + trade_return) #Correct
                position = None
    
    return trades, equity