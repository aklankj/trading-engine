import pandas as pd
import numpy as np

def strategy(df):
    """
    Implements a simple moving average crossover strategy with RSI, ATR, and Bollinger Bands for confirmation, plus stop loss and take profit.
    
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
        sma = pd.Series(index=data.index, dtype=float)
        for i in range(period, len(data)):
            sma.iloc[i] = data.iloc[i-period:i].mean()
        return sma

    def calculate_rsi(close, period=14):
        delta = close.diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        roll_up1 = up.rolling(period).mean()
        roll_down1 = down.abs().rolling(period).mean()
        rs = roll_up1 / roll_down1
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def calculate_atr(df, period=14):
        atr = pd.Series(index=df.index, dtype=float)
        for i in range(1, len(df)):
            high_low = df['high'].iloc[i] - df['low'].iloc[i]
            high_close = np.abs(df['high'].iloc[i] - df['close'].iloc[i-1])
            low_close = np.abs(df['low'].iloc[i] - df['close'].iloc[i-1])
            atr.iloc[i] = np.max([high_low, high_close, low_close])

        atr_series = pd.Series(index=df.index, dtype=float)
        for i in range(period, len(df)):
            atr_series.iloc[i] = atr.iloc[i-period:i].mean()
        return atr_series


    def calculate_bollinger_bands(close, period=20, num_std=2):
        sma = calculate_sma(close, period)
        std = pd.Series(index=close.index, dtype=float)
        for i in range(period, len(close)):
            std.iloc[i] = close.iloc[i-period:i].std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band

    if df is None or len(df) == 0:
        return trades, equity

    if len(df) < 200:
        return trades, equity
    
    fast_period = 20
    slow_period = 50
    rsi_period = 14
    atr_period = 14
    bollinger_period = 20
    bollinger_std = 2
    
    sma_fast = calculate_sma(df['close'], fast_period)
    sma_slow = calculate_sma(df['close'], slow_period)
    rsi = calculate_rsi(df['close'], rsi_period)
    atr = calculate_atr(df, atr_period)
    upper_band, lower_band = calculate_bollinger_bands(df['close'], bollinger_period, bollinger_std)
       
    stop_loss_factor = 2
    take_profit_factor = 3

    position_size = 0.1  # 10% of equity

    for i in range(slow_period, len(df)):
        if position is None:
            # Entry condition
            if (sma_fast.iloc[i] > sma_slow.iloc[i] and 
                sma_fast.iloc[i-1] <= sma_slow.iloc[i-1] and 
                rsi.iloc[i] < 70 and
                df['close'].iloc[i] > lower_band.iloc[i]):  # Buy
                
                position = 'long'
                entry_price = df['close'].iloc[i]
                atr_val = atr.iloc[i]
                stop_loss = entry_price - (stop_loss_factor * atr_val)
                take_profit = entry_price + (take_profit_factor * atr_val)
                
                
            elif (sma_fast.iloc[i] < sma_slow.iloc[i] and 
                  sma_fast.iloc[i-1] >= sma_slow.iloc[i-1] and 
                  rsi.iloc[i] > 30 and
                  df['close'].iloc[i] < upper_band.iloc[i]):  # Sell
            
                position = 'short'
                entry_price = df['close'].iloc[i]
                atr_val = atr.iloc[i]
                stop_loss = entry_price + (stop_loss_factor * atr_val)
                take_profit = entry_price - (take_profit_factor * atr_val)


        elif position == 'long':
            # Exit condition for long position
            if df['close'].iloc[i] <= stop_loss or df['close'].iloc[i] >= take_profit:
                profit = (df['close'].iloc[i] - entry_price) / entry_price * 100
                trades.append(profit)
                equity += equity * position_size * profit / 100 
                position = None
                
        elif position == 'short':
            # Exit condition for short position
            if df['close'].iloc[i] >= stop_loss or df['close'].iloc[i] <= take_profit:
                profit = (entry_price - df['close'].iloc[i]) / entry_price * 100
                trades.append(profit)
                equity += equity * position_size * profit / 100
                position = None

    return trades, equity