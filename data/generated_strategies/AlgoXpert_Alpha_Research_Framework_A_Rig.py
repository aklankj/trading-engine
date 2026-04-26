import pandas as pd
import numpy as np

def strategy(df):
    """
    A simple moving average crossover strategy with RSI, ATR stop loss and take profit.
    
    Args: df - pandas DataFrame with columns: open, high, low, close, volume
          Index is DatetimeIndex, daily frequency, at least 2 years of data.
    
    Returns: (trades_list, final_equity)
        trades_list: list of float (percent returns per trade)
        final_equity: float (starting from 100000)
    """
    trades = []
    equity = 100000
    position = None
    
    if df is None or df.empty:
        return trades, equity
    
    if len(df) < 250:
        return trades, equity

    def calculate_sma(data, period):
        sma = np.zeros(len(data))
        for i in range(period, len(data)):
            sma[i] = np.mean(data[i-period:i])
        return sma

    def calculate_rsi(data, period):
        delta = np.diff(data)
        up, down = delta.copy(), delta.copy()
        up[up<0] = 0
        down[down>0] = 0
        
        roll_up1 = pd.Series(up).rolling(period).mean()
        roll_down1 = pd.Series(down).abs().rolling(period).mean()
        
        rs = roll_up1 / roll_down1
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = np.concatenate(([np.nan]*period, rsi))  # Pad with NaN values at the beginning
        return rsi
    
    def calculate_atr(df, period):
        atr = np.zeros(len(df))
        for i in range(1, len(df)):
            high_low = df['high'].iloc[i] - df['low'].iloc[i]
            high_close = np.abs(df['high'].iloc[i] - df['close'].iloc[i-1])
            low_close = np.abs(df['low'].iloc[i] - df['close'].iloc[i-1])
            atr[i] = np.max([high_low, high_close, low_close])

        atr_series = pd.Series(atr)
        atr_smooth = atr_series.rolling(period).mean().to_numpy()
        atr_smooth[:period]=np.nan
        return atr_smooth

    short_window = 20
    long_window = 50
    rsi_window = 14
    atr_window = 14

    sma_short = calculate_sma(df['close'].values, short_window)
    sma_long = calculate_sma(df['close'].values, long_window)
    rsi = calculate_rsi(df['close'].values, rsi_window)
    atr = calculate_atr(df, atr_window)
    
    position_size = 0.1

    for i in range(max(long_window, rsi_window, atr_window), len(df)):
        current_price = df['close'].iloc[i]

        if sma_short[i] > sma_long[i] and sma_short[i-1] <= sma_long[i-1] and rsi[i] < 70 and position is None:
            # Buy signal
            stop_loss = current_price - atr[i] * 2
            take_profit = current_price + atr[i] * 3
            
            shares_to_buy = (equity * position_size) // current_price
            if shares_to_buy > 0:
                position = {
                    'entry_price': current_price,
                    'shares': shares_to_buy,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'type': 'long'
                }

        elif sma_short[i] < sma_long[i] and sma_short[i-1] >= sma_long[i-1] and rsi[i] > 30 and position is None:
            # Short signal
            stop_loss = current_price + atr[i] * 2
            take_profit = current_price - atr[i] * 3
            
            shares_to_sell = (equity * position_size) // current_price
            if shares_to_sell > 0:
                position = {
                    'entry_price': current_price,
                    'shares': shares_to_sell,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'type': 'short'
                }

        if position is not None:
            if position['type'] == 'long':
                if current_price <= position['stop_loss'] or current_price >= position['take_profit']:
                    # Exit long position
                    trade_return = (current_price - position['entry_price']) / position['entry_price'] * 100
                    trades.append(trade_return)
                    equity += position['shares'] * (current_price - position['entry_price'])
                    position = None

            elif position['type'] == 'short':
                if current_price >= position['stop_loss'] or current_price <= position['take_profit']:
                    # Exit short position
                    trade_return = (position['entry_price'] - current_price) / position['entry_price'] * 100
                    trades.append(trade_return)
                    equity += position['shares'] * (position['entry_price'] - current_price)
                    position = None

    return trades, equity