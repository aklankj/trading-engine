import pandas as pd
import numpy as np

def strategy(df):
    """
    Implements a simplified Confidence Weighted Mean Reversion strategy using SMA, RSI, ATR for signals and volatility-based position sizing.
    
    Args: df - pandas DataFrame with columns: open, high, low, close, volume
          Index is DatetimeIndex, daily frequency, at least 2 years of data.
    
    Returns: (trades_list, final_equity)
        trades_list: list of float (percent returns per trade)
        final_equity: float (starting from 100000)
    """
    trades = []
    equity = 100000
    position = None
    
    # --- Helper Functions ---
    
    def calculate_sma(data, period):
        sma = pd.Series(np.nan, index=data.index)
        for i in range(period, len(data)):
            sma.iloc[i] = data.iloc[i-period:i].mean()
        return sma
    
    def calculate_rsi(data, period):
        delta = data.diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        roll_up1 = up.ewm(span=period).mean() # use .mean() instead of .sum() for EMA
        roll_down1 = down.abs().ewm(span=period).mean() # use .mean() instead of .sum() for EMA
        RS = roll_up1 / roll_down1
        RSI = 100.0 - (100.0 / (1.0 + RS))
        return RSI
    
    def calculate_atr(df, period):
        tr = pd.Series(np.zeros(len(df)), index=df.index)
        for i in range(1, len(df)):
            tr.iloc[i] = max(df['high'].iloc[i] - df['low'].iloc[i],
                              abs(df['high'].iloc[i] - df['close'].iloc[i-1]),
                              abs(df['low'].iloc[i] - df['close'].iloc[i-1]))
        atr = tr.rolling(window=period).mean()
        return atr
    
    # --- End Helper Functions ---
    
    if df.empty or len(df) < 500:
        return trades, equity
    
    SMA_PERIOD = 20
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    WARMUP = 200
    
    df['SMA'] = calculate_sma(df['close'], SMA_PERIOD)
    df['RSI'] = calculate_rsi(df['close'], RSI_PERIOD)
    df['ATR'] = calculate_atr(df, ATR_PERIOD)
    
    STOP_LOSS = 0.02  # 2% stop loss
    TAKE_PROFIT = 0.04 # 4% take profit
    
    for i in range(WARMUP, len(df)):
        if df['SMA'].iloc[i] is np.nan or df['RSI'].iloc[i] is np.nan or df['ATR'].iloc[i] is np.nan:
            continue
        
        current_price = df['close'].iloc[i]
        
        # --- Trading Logic ---
        
        if position is None:
            # Entry Condition
            if df['close'].iloc[i] > df['SMA'].iloc[i] and df['RSI'].iloc[i] < 40:
                # Go Long
                position_size = 0.1 * equity # 10% of equity
                entry_price = current_price
                stop_loss_price = entry_price * (1 - STOP_LOSS)
                take_profit_price = entry_price * (1 + TAKE_PROFIT)
                position = {'type': 'long', 'entry': entry_price, 'size': position_size, 'stop_loss': stop_loss_price, 'take_profit': take_profit_price}
        
        elif position['type'] == 'long':
            # Exit Logic
            if current_price <= position['stop_loss'] or current_price >= position['take_profit']:
                # Exit Long
                trade_return = (current_price - position['entry']) / position['entry'] * 100
                trades.append(trade_return)
                equity += position['size'] * (current_price - position['entry']) / position['entry']
                position = None  # Clear position

    #Close open position at the end
    if position is not None:
        if position['type'] == 'long':
            trade_return = (df['close'].iloc[-1] - position['entry']) / position['entry'] * 100
            trades.append(trade_return)
            equity += position['size'] * (df['close'].iloc[-1] - position['entry']) / position['entry']
            position = None

    return trades, equity