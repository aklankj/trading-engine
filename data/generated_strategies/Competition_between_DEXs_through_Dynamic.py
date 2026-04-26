import pandas as pd
import numpy as np

def strategy(df):
    """
    Implements a simple mean-reversion strategy using RSI and Bollinger Bands, with stop loss and take profit.
    
    Args: df - pandas DataFrame with columns: open, high, low, close, volume
          Index is DatetimeIndex, daily frequency, at least 2 years of data.
    
    Returns: (trades_list, final_equity)
        trades_list: list of float (percent returns per trade)
        final_equity: float (starting from 100000)
    """
    
    def calculate_rsi(data, period=14):
        delta = data.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        
        roll_up1 = up.rolling(period).mean()
        roll_down1 = down.rolling(period).mean()
        
        rs = roll_up1 / roll_down1
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi
    
    def calculate_bollinger_bands(data, period=20, num_std=2):
        sma = data.rolling(period).mean()
        std = data.rolling(period).std()
        upper_band = sma + num_std * std
        lower_band = sma - num_std * std
        return sma, upper_band, lower_band

    if df.empty or len(df) < 500:
        return [], 100000.0
    
    trades = []
    equity = 100000.0
    position = None
    position_size = 0.1  # 10% of equity
    stop_loss_pct = 0.02  # 2% stop loss
    take_profit_pct = 0.03  # 3% take profit
    warmup_period = 200

    rsi = calculate_rsi(df['close'])
    sma, upper_band, lower_band = calculate_bollinger_bands(df['close'])
    
    for i in range(warmup_period, len(df)):
        current_price = df['close'].iloc[i]
        current_rsi = rsi.iloc[i]
        current_upper_band = upper_band.iloc[i]
        current_lower_band = lower_band.iloc[i]
        
        if position is None:
            if current_rsi < 30 and current_price < current_lower_band:
                # Buy
                position = 'long'
                entry_price = current_price
                units = (equity * position_size) / current_price
            elif current_rsi > 70 and current_price > current_upper_band:
                # Sell
                position = 'short'
                entry_price = current_price
                units = (equity * position_size) / current_price

        elif position == 'long':
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)
            
            if current_price <= stop_loss or current_price >= take_profit:
                # Exit long position
                trade_return = (current_price - entry_price) / entry_price
                trades.append(trade_return * 100)
                equity *= (1 + trade_return)
                position = None
        
        elif position == 'short':
            stop_loss = entry_price * (1 + stop_loss_pct)
            take_profit = entry_price * (1 - take_profit_pct)
            
            if current_price >= stop_loss or current_price <= take_profit:
                # Exit short position
                trade_return = (entry_price - current_price) / entry_price
                trades.append(trade_return * 100)
                equity *= (1 + trade_return)
                position = None
    
    return trades, equity