import pandas as pd
import numpy as np

def strategy(df):
    """
    A simple moving average crossover strategy with RSI, ATR, and stop-loss/take-profit.
    
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
    
    def calculate_sma(data, period):
        sma = np.zeros(len(data))
        sma[:] = np.nan
        for i in range(period, len(data)):
            sma[i] = np.mean(data.iloc[i-period:i])
        return sma

    def calculate_rsi(data, period):
        delta = data.diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        
        roll_up1 = up.rolling(period).mean()
        roll_down1 = np.abs(down.rolling(period).mean())
        RS1 = roll_up1 / roll_down1
        RSI1 = 100.0 - (100.0 / (1.0 + RS1))
        
        rsi = np.zeros(len(data))
        rsi[:] = np.nan
        rsi[period:] = RSI1[period:]
        return rsi
        
    def calculate_atr(df, period):
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr = np.zeros(len(df))
        tr[0] = np.nan
        for i in range(1, len(df)):
            tr[i] = max(high.iloc[i] - low.iloc[i], 
                        abs(high.iloc[i] - close.iloc[i-1]),
                        abs(low.iloc[i] - close.iloc[i-1]))
        
        atr = np.zeros(len(df))
        atr[:] = np.nan
        atr[period:] = pd.Series(tr).rolling(period).mean()[period:]
        return atr
        
    short_window = 20
    long_window = 50
    rsi_window = 14
    atr_window = 14
    
    df['SMA_short'] = calculate_sma(df['close'], short_window)
    df['SMA_long'] = calculate_sma(df['close'], long_window)
    df['RSI'] = calculate_rsi(df['close'], rsi_window)
    df['ATR'] = calculate_atr(df, atr_window)

    # Trading Logic
    position_size = 0.1
    stop_loss_multiple = 2
    take_profit_multiple = 3
    
    for i in range(max(long_window, rsi_window, atr_window) + 1, len(df)):
        if position is None:
            # Check for entry conditions
            if (df['SMA_short'].iloc[i] > df['SMA_long'].iloc[i] and
                df['SMA_short'].iloc[i-1] <= df['SMA_long'].iloc[i-1] and
                df['RSI'].iloc[i] < 70):

                # Enter Long Position
                entry_price = df['close'].iloc[i]
                stop_loss = entry_price - stop_loss_multiple * df['ATR'].iloc[i]
                take_profit = entry_price + take_profit_multiple * df['ATR'].iloc[i]
                
                position = {
                    'type': 'long',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'size': equity * position_size 
                }

        elif position['type'] == 'long':
            # Check for exit conditions (Stop Loss or Take Profit)
            current_price = df['close'].iloc[i]
            
            if current_price <= position['stop_loss'] or current_price >= position['take_profit']:
                # Exit Long Position
                exit_price = current_price
                trade_return = (exit_price - position['entry_price']) / position['entry_price']
                trade_return_percent = trade_return * 100
                trades.append(trade_return_percent)
                
                equity += position['size'] * trade_return
                position = None
                
        if equity <= 0:
            break

    return trades, equity