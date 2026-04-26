import pandas as pd
import numpy as np

def strategy(df):
    """
    Implements a simple technical analysis trading strategy using SMA, RSI, ATR, and Bollinger Bands.
    
    Args: df - pandas DataFrame with columns: open, high, low, close, volume
          Index is DatetimeIndex, daily frequency, at least 2 years of data.
    
    Returns: (trades_list, final_equity)
        trades_list: list of float (percent returns per trade)
        final_equity: float (starting from 100000)
    """
    def calculate_sma(data, period):
        sma = pd.Series(np.nan, index=data.index)
        if len(data) >= period:
            sma = data.rolling(window=period).mean()
        return sma

    def calculate_rsi(data, period):
        delta = data.diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        roll_up1 = up.ewm(span=period).mean()
        roll_down1 = np.abs(down.ewm(span=period).mean())
        rs = roll_up1 / roll_down1
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def calculate_atr(df, period):
        atr = pd.Series(np.nan, index=df.index)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean() if len(df) >= period else pd.Series([np.nan]*len(df), index=df.index)
        return atr

    def calculate_bollinger_bands(data, period, num_std):
        sma = calculate_sma(data, period)
        std = data.rolling(window=period).std()
        upper_band = sma + std * num_std
        lower_band = sma - std * num_std
        return upper_band, lower_band
    
    trades = []
    equity = 100000
    position = None
    
    if df is None or len(df) < 250:
        return trades, equity
    
    SMA_PERIOD = 20
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    BB_PERIOD = 20
    BB_STD = 2
    
    df['SMA'] = calculate_sma(df['close'], SMA_PERIOD)
    df['RSI'] = calculate_rsi(df['close'], RSI_PERIOD)
    df['ATR'] = calculate_atr(df, ATR_PERIOD)
    df['UpperBB'], df['LowerBB'] = calculate_bollinger_bands(df['close'], BB_PERIOD, BB_STD)
    
    RISK_PCT = 0.10
    
    for i in range(200, len(df)):
        current_price = df['close'].iloc[i]
        current_sma = df['SMA'].iloc[i]
        current_rsi = df['RSI'].iloc[i]
        current_atr = df['ATR'].iloc[i]
        current_upper_bb = df['UpperBB'].iloc[i]
        current_lower_bb = df['LowerBB'].iloc[i]
        
        if (current_sma is np.nan) or (current_rsi is np.nan) or (current_atr is np.nan) or (current_upper_bb is np.nan) or (current_lower_bb is np.nan):
            continue

        if position is None:
            # Buy Logic
            if current_rsi < 30 and current_price < current_lower_bb:
                position_size = (equity * RISK_PCT) / current_price
                position = {
                    'type': 'long',
                    'entry_price': current_price,
                    'position_size': position_size,
                    'stop_loss': current_price - 2 * current_atr,
                    'take_profit': current_price + 4 * current_atr
                }
                
        elif position['type'] == 'long':
            # Exit Logic for Long Position
            if current_price <= position['stop_loss']:
                trade_return = (position['stop_loss'] - position['entry_price']) / position['entry_price'] * 100
                trades.append(trade_return)
                equity += equity * RISK_PCT * trade_return / 100
                position = None
            elif current_price >= position['take_profit']:
                trade_return = (position['take_profit'] - position['entry_price']) / position['entry_price'] * 100
                trades.append(trade_return)
                equity += equity * RISK_PCT * trade_return / 100
                position = None
            elif current_rsi > 70 and current_price > current_upper_bb:
                 trade_return = (current_price - position['entry_price']) / position['entry_price'] * 100
                 trades.append(trade_return)
                 equity += equity * RISK_PCT * trade_return / 100
                 position = None
            
    if position is not None:
            if position['type'] == 'long':
                trade_return = (current_price - position['entry_price']) / position['entry_price'] * 100
                trades.append(trade_return)
                equity += equity * RISK_PCT * trade_return / 100
                position = None
    
    
    return trades, equity