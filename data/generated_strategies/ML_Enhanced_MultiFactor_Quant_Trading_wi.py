import pandas as pd
import numpy as np

def strategy(df):
    """
    Multi-factor trading strategy using SMA, RSI, ATR, and Bollinger Bands with stop loss and take profit.
    
    Args: df - pandas DataFrame with columns: open, high, low, close, volume
          Index is DatetimeIndex, daily frequency, at least 2 years of data.
    
    Returns: (trades_list, final_equity)
        trades_list: list of float (percent returns per trade)
        final_equity: float (starting from 100000)
    """
    trades = []
    equity = 100000
    position = None
    
    if df.empty or len(df) < 250:
        return trades, equity
    
    # Calculate indicators
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    
    delta = df['close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    
    roll_up1 = up.ewm(span=14).mean()
    roll_down1 = np.abs(down.ewm(span=14).mean())
    
    RS = roll_up1 / roll_down1
    df['RSI'] = 100.0 - (100.0 / (1.0 + RS))
    
    df['ATR'] = (df['high'] - df['low']).rolling(window=14).mean()
    
    df['TP'] = (df['high'] + df['low'] + df['close']) / 3
    df['StdDev'] = df['TP'].rolling(window=20).std()
    df['Upper'] = df['SMA_20'] + 2 * df['StdDev']
    df['Lower'] = df['SMA_20'] - 2 * df['StdDev']
    
    # Trading logic
    for i in range(200, len(df)):
        # Close existing position
        if position == 'long':
            if df['close'][i] <= stop_loss or df['close'][i] >= take_profit:
                trade_return = (df['close'][i] - entry_price) / entry_price
                trades.append(trade_return)
                equity *= (1 + trade_return)
                position = None
        elif position == 'short':
            if df['close'][i] >= stop_loss or df['close'][i] <= take_profit:
                trade_return = (entry_price - df['close'][i]) / entry_price
                trades.append(trade_return)
                equity *= (1 + trade_return)
                position = None
        
        # Open new position if no open position
        if position is None:
            if df['SMA_20'][i] > df['SMA_50'][i] and df['RSI'][i] < 30 and df['close'][i] < df['Lower'][i]:
                position_size = equity * 0.1
                entry_price = df['close'][i]
                stop_loss = entry_price - 2 * df['ATR'][i]
                take_profit = entry_price + 3 * df['ATR'][i]
                
                if stop_loss > 0 and take_profit > 0:
                    position = 'long'
                    
            elif df['SMA_20'][i] < df['SMA_50'][i] and df['RSI'][i] > 70 and df['close'][i] > df['Upper'][i]:
                position_size = equity * 0.1
                entry_price = df['close'][i]
                stop_loss = entry_price + 2 * df['ATR'][i]
                take_profit = entry_price - 3 * df['ATR'][i]
                
                if stop_loss > 0 and take_profit > 0:
                    position = 'short'

    # Close any open position at the end
    if position == 'long':
        trade_return = (df['close'].iloc[-1] - entry_price) / entry_price
        trades.append(trade_return)
        equity *= (1 + trade_return)
            
    elif position == 'short':
        trade_return = (entry_price - df['close'].iloc[-1]) / entry_price
        trades.append(trade_return)
        equity *= (1 + trade_return)
        
    return trades, equity