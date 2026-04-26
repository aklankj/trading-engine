def strategy(df):
    """
    Implements a simple trend-following strategy with SMA, RSI, ATR, and Bollinger Bands.
    
    Args: df - pandas DataFrame with columns: open, high, low, close, volume
          Index is DatetimeIndex, daily frequency, at least 2 years of data.
    
    Returns: (trades_list, final_equity)
        trades_list: list of float (percent returns per trade)
        final_equity: float (starting from 100000)
    """
    import pandas as pd
    import numpy as np
    
    trades = []
    equity = 100000
    position = None
    position_size = 0.1
    
    if df.empty or len(df) < 500: # Ensure enough data
        return trades, equity
    
    # Calculate indicators
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['RSI_14'] = calculate_rsi(df['close'], window=14)
    df['ATR_14'] = calculate_atr(df['high'], df['low'], df['close'], window=14)
    df['BB_UPPER'], df['BB_LOWER'] = calculate_bollinger_bands(df['close'], window=20, num_std=2)
    
    # Backtesting loop
    for i in range(200, len(df)): # Start after warmup period
        # Generate signals
        if position is None:
            # Long condition
            if df['close'][i] > df['SMA_20'][i] and df['RSI_14'][i] < 30:
                # Calculate stop loss and take profit levels based on ATR
                stop_loss = df['close'][i] - (2 * df['ATR_14'][i])
                take_profit = df['close'][i] + (2 * df['ATR_14'][i])
                
                position = 'long'
                entry_price = df['close'][i]
                units = (equity * position_size) / entry_price
                
            # Short condition
            elif df['close'][i] < df['SMA_20'][i] and df['RSI_14'][i] > 70:
                # Calculate stop loss and take profit levels based on ATR
                stop_loss = df['close'][i] + (2 * df['ATR_14'][i])
                take_profit = df['close'][i] - (2 * df['ATR_14'][i])
                
                position = 'short'
                entry_price = df['close'][i]
                units = (equity * position_size) / entry_price
                
        # Check for exit conditions
        elif position == 'long':
            if df['close'][i] >= take_profit or df['close'][i] <= stop_loss:
                trade_return = (df['close'][i] - entry_price) / entry_price
                trades.append(trade_return)
                equity *= (1 + trade_return * units / (equity * position_size / entry_price))
                position = None
                
        elif position == 'short':
            if df['close'][i] <= take_profit or df['close'][i] >= stop_loss:
                trade_return = (entry_price - df['close'][i]) / entry_price
                trades.append(trade_return)
                equity *= (1 + trade_return * units / (equity * position_size / entry_price))
                position = None
    
    return trades, equity

def calculate_rsi(data, window=14):
    delta = data.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    
    roll_up1 = up.rolling(window).mean()
    roll_down1 = down.abs().rolling(window).mean()
    
    RS = roll_up1 / roll_down1
    RSI = 100.0 - (100.0 / (1.0 + RS))
    return RSI

def calculate_atr(high, low, close, window=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr

def calculate_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band