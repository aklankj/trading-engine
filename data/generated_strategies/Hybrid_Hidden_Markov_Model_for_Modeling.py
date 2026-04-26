import pandas as pd
import numpy as np

def strategy(df):
    """
    Implements a simple moving average crossover strategy with RSI confirmation, ATR-based stop loss and take profit.
    
    Args: df - pandas DataFrame with columns: open, high, low, close, volume
          Index is DatetimeIndex, daily frequency, at least 2 years of data.
    
    Returns: (trades_list, final_equity)
        trades_list: list of float (percent returns per trade)
        final_equity: float (starting from 100000)
    """
    trades = []
    equity = 100000
    position = None
    
    def calculate_sma(data, period):
        sma = np.zeros(len(data))
        for i in range(period, len(data)):
            sma[i] = np.mean(data[i-period:i])
        return sma
    
    def calculate_rsi(data, period):
        rsi = np.zeros(len(data))
        delta = np.diff(data)
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        roll_up1 = pd.Series(up).rolling(period).mean()
        roll_down1 = pd.Series(down).abs().rolling(period).mean()
        rs = roll_up1 / roll_down1
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = np.concatenate(([np.nan]*period, rsi[period:])) # Add initial NaN values
        return rsi

    def calculate_atr(df, period):
        atr = np.zeros(len(df))
        for i in range(1,len(df)):
            tr = max(df['high'].iloc[i] - df['low'].iloc[i], abs(df['high'].iloc[i] - df['close'].iloc[i-1]), abs(df['low'].iloc[i] - df['close'].iloc[i-1]))
            atr[i]=tr
        atr[0]=np.mean(atr[1:period+1])
        for i in range(period + 1,len(df)):
            atr[i] = (atr[i-1]*(period-1)+tr)/period
        return atr

    if df is None or len(df) < 500:  # Need sufficient data
        return trades, equity

    short_window = 20
    long_window = 50
    rsi_window = 14
    atr_window = 14
    warmup = max(long_window, rsi_window, atr_window) + 100

    sma_short = calculate_sma(df['close'].values, short_window)
    sma_long = calculate_sma(df['close'].values, long_window)
    rsi = calculate_rsi(df['close'].values, rsi_window)
    atr = calculate_atr(df, atr_window)

    if np.isnan(sma_short).all() or np.isnan(sma_long).all() or np.isnan(rsi).all() or np.isnan(atr).all():
        return trades, equity

    trade_size = 0.1
    
    for i in range(warmup, len(df)):
        current_price = df['close'].iloc[i]
        
        if sma_short[i] > sma_long[i] and rsi[i] < 70 and position is None:
             # Enter Long position
            position_size = equity * trade_size / current_price
            entry_price = current_price
            stop_loss = entry_price - atr[i] * 2
            take_profit = entry_price + atr[i] * 3
            position = 'long'

        elif sma_short[i] < sma_long[i] and rsi[i] > 30 and position is None:
           # Enter Short position
            position_size = equity * trade_size / current_price
            entry_price = current_price
            stop_loss = entry_price + atr[i] * 2
            take_profit = entry_price - atr[i] * 3
            position = 'short'

        elif position == 'long':
            if current_price <= stop_loss or current_price >= take_profit:
                # Exit Long position
                profit = (current_price - entry_price) / entry_price
                trades.append(profit * 100)
                equity *= (1 + profit)
                position = None

        elif position == 'short':
            if current_price >= stop_loss or current_price <= take_profit:
                # Exit Short position
                profit = (entry_price - current_price) / entry_price
                trades.append(profit * 100)
                equity *= (1 + profit)
                position = None

    return trades, equity