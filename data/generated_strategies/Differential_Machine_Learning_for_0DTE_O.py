import pandas as pd
import numpy as np

def strategy(df):
    """
    Implements a simple moving average crossover strategy with RSI, ATR, and Bollinger Bands for risk management.
    
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
        delta = np.diff(data)
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        
        roll_up1 = pd.Series(up).rolling(period).mean()
        roll_down1 = pd.Series(down).abs().rolling(period).mean()
        
        rs = roll_up1 / roll_down1
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = np.insert(rsi.to_numpy(), 0, np.zeros(period)) #pad with zeros
        return rsi
    
    def calculate_atr(df, period):
        atr = np.zeros(len(df))
        for i in range(1, len(df)):
            tr = max(df['high'].iloc[i] - df['low'].iloc[i], 
                         abs(df['high'].iloc[i] - df['close'].iloc[i-1]),
                         abs(df['low'].iloc[i] - df['close'].iloc[i-1]))
            atr[i] = (atr[i-1] * (period - 1) + tr) / period if i > period else tr
        
        atr[:period] = np.mean(atr[1:period+1]) if period + 1 <= len(df) else 0 # Mean fill before period
        return atr
    
    def calculate_bollinger_bands(data, period, num_std):
        sma = calculate_sma(data, period)
        std = np.zeros(len(data))
        for i in range(period, len(data)):
            std[i] = np.std(data[i-period:i])
        upper_band = sma + num_std * std
        lower_band = sma - num_std * std
        return upper_band, lower_band
    
    if df is None or len(df) < 250: # Min 200 warmup + some extra for indicators
        return trades, equity
    
    sma_short_period = 20
    sma_long_period = 50
    rsi_period = 14
    atr_period = 14
    bb_period = 20
    bb_std = 2
    
    sma_short = calculate_sma(df['close'].to_numpy(), sma_short_period)
    sma_long = calculate_sma(df['close'].to_numpy(), sma_long_period)
    rsi = calculate_rsi(df['close'].to_numpy(), rsi_period)
    atr = calculate_atr(df, atr_period)
    upper_band, lower_band = calculate_bollinger_bands(df['close'].to_numpy(), bb_period, bb_std)
    
    stop_loss_pct = 0.02 # 2% stop loss
    take_profit_pct = 0.05 # 5% take profit
    position_size_pct = 0.10 # 10% position size

    for i in range(max(sma_long_period, rsi_period, atr_period, bb_period), len(df)):
        
        # Check and close existing position
        if position is not None:
            entry_price = position['entry_price']
            trade_type = position['type']
            
            # Stop Loss Check
            if trade_type == 'long' and df['low'].iloc[i] <= position['stop_loss']:
                trade_return = (position['stop_loss'] - entry_price) / entry_price
                trades.append(trade_return * 100)
                equity *= (1 + trade_return)
                position = None
                continue
            elif trade_type == 'short' and df['high'].iloc[i] >= position['stop_loss']:
                trade_return = (entry_price - position['stop_loss']) / entry_price * -1
                trades.append(trade_return * 100)
                equity *= (1 + trade_return)
                position = None
                continue
            
            # Take Profit Check
            if trade_type == 'long' and df['high'].iloc[i] >= position['take_profit']:
                trade_return = (position['take_profit'] - entry_price) / entry_price
                trades.append(trade_return * 100)
                equity *= (1 + trade_return)
                position = None
                continue
            elif trade_type == 'short' and df['low'].iloc[i] <= position['take_profit']:
                trade_return = (entry_price - position['take_profit']) / entry_price * -1
                trades.append(trade_return * 100)
                equity *= (1 + trade_return)
                position = None
                continue
                
        # Open new position
        if position is None:
            if sma_short[i] > sma_long[i] and sma_short[i-1] <= sma_long[i-1] and \
               rsi[i] < 70 and df['close'].iloc[i] > lower_band[i]: # Add RSI and BB confirmation
                
                entry_price = df['close'].iloc[i]
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + take_profit_pct)
                
                position = {
                    'type': 'long',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
            elif sma_short[i] < sma_long[i] and sma_short[i-1] >= sma_long[i-1] and \
                 rsi[i] > 30 and df['close'].iloc[i] < upper_band[i]: # Add RSI and BB confirmation
                
                entry_price = df['close'].iloc[i]
                stop_loss = entry_price * (1 + stop_loss_pct)
                take_profit = entry_price * (1 - take_profit_pct)
                
                position = {
                    'type': 'short',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }

    return trades, equity