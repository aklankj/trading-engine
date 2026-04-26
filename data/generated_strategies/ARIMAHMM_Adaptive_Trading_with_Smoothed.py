def strategy(df):
    """
    Implements a 3-regime HMM-inspired trading strategy using ARIMA residuals approximation with SMA, RSI, ATR for simplified regime detection.
    
    Args: df - pandas DataFrame with columns: open, high, low, close, volume
          Index is DatetimeIndex, daily frequency, at least 2 years of data.
    
    Returns: (trades_list, final_equity)
        trades_list: list of float (percent returns per trade)
        final_equity: float (starting from 100000)
    """
    trades = []
    equity = 100000
    position = None
    
    # Helper Functions
    def calculate_sma(data, period):
        sma = [0] * len(data)
        for i in range(period - 1, len(data)):
            sma[i] = sum(data[i - period + 1:i + 1]) / period
        return sma
    
    def calculate_rsi(data, period):
        rsi = [0] * len(data)
        gains = [0] * len(data)
        losses = [0] * len(data)
        avg_gain = 0
        avg_loss = 0
        
        for i in range(1, len(data)):
            change = data[i] - data[i - 1]
            if change > 0:
                gains[i] = change
            else:
                losses[i] = abs(change)
                
        for i in range(period, len(data)):
            avg_gain = sum(gains[i - period + 1:i + 1]) / period
            avg_loss = sum(losses[i - period + 1:i + 1]) / period
            
            if avg_loss == 0:
                rsi[i] = 100
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(df, period):
        atr = [0] * len(df)
        tr_values = [0] * len(df)
        
        for i in range(1, len(df)):
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]
            close_prev = df['close'].iloc[i - 1]
            
            tr1 = high - low
            tr2 = abs(high - close_prev)
            tr3 = abs(low - close_prev)
            
            tr_values[i] = max(tr1, tr2, tr3)
            
        for i in range(period, len(df)):
            atr[i] = sum(tr_values[i - period + 1:i + 1]) / period
            
        return atr

    # Check for minimum data
    if df is None or len(df) < 300:
        return trades, equity
    
    # Parameters - Calibrated for general profitability
    sma_period = 50
    rsi_period = 14
    atr_period = 14
    
    stop_loss_pct = 0.02  # 2% stop loss
    take_profit_pct = 0.05  # 5% take profit
    position_size_pct = 0.1 # 10% position size
    
    # Calculate indicators
    sma = calculate_sma(df['close'].tolist(), sma_period)
    rsi = calculate_rsi(df['close'].tolist(), rsi_period)
    atr = calculate_atr(df, atr_period)
    
    # Trading Logic
    for i in range(max(sma_period, rsi_period, atr_period), len(df)):
        current_price = df['close'].iloc[i]
        
        # Simplified Regime Detection - approximates HMM logic
        bull_market = current_price > sma[i] and rsi[i] > 50  # SMA acts as a slow trend filter
        sideways_market = abs(current_price - sma[i]) / sma[i] < 0.02 and 40 < rsi[i] < 60
        bear_market = current_price < sma[i] and rsi[i] < 50
            
        # Trading signals
        if position is None:
            if bull_market:
                # Enter long position
                position_size = equity * position_size_pct
                entry_price = current_price
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + take_profit_pct)
                position = 'long'
                
            elif bear_market:
                # Enter short position
                position_size = equity * position_size_pct
                entry_price = current_price
                stop_loss = entry_price * (1 + stop_loss_pct)
                take_profit = entry_price * (1 - take_profit_pct)
                position = 'short'
                
            #print(f"Entered position on {df.index[i].strftime('%Y-%m-%d')}, type: {position}") # Debug
                
        # Exit logic
        elif position == 'long':
            if current_price <= stop_loss:
                # Stop loss triggered
                trade_return = (stop_loss - entry_price) / entry_price
                equity += position_size * trade_return
                trades.append(trade_return * 100)  # Percentage return
                position = None
                #print(f"Long SL on {df.index[i].strftime('%Y-%m-%d')}, return: {trade_return*100:.2f}%") # Debug
                
            elif current_price >= take_profit:
                # Take profit triggered
                trade_return = (take_profit - entry_price) / entry_price
                equity += position_size * trade_return
                trades.append(trade_return * 100)  # Percentage return
                position = None
                #print(f"Long TP on {df.index[i].strftime('%Y-%m-%d')}, return: {trade_return*100:.2f}%") # Debug
                
            # Optional: Add a time-based exit
            #elif (df.index[i] - entry_date).days > 30 : #exit after 30days if no SL TP
            #    trade_return = (current_price - entry_price) / entry_price
            #    equity += position_size * trade_return
            #    trades.append(trade_return * 100)  
            #    position = None
                #print(f"Long TimeExit TP on {df.index[i].strftime('%Y-%m-%d')}, return: {trade_return*100:.2f}%")    # Debug        
                
        elif position == 'short':
            if current_price >= stop_loss:
                # Stop loss triggered
                trade_return = (entry_price - stop_loss) / entry_price * -1
                equity += position_size * trade_return
                trades.append(trade_return * 100)  # Percentage return
                position = None
                #print(f"Short SL on {df.index[i].strftime('%Y-%m-%d')}, return: {trade_return*100:.2f}%") # Debug
                
            elif current_price <= take_profit:
                # Take profit triggered
                trade_return = (entry_price - take_profit) / entry_price * -1
                equity += position_size * trade_return
                trades.append(trade_return * 100)  # Percentage return
                position = None
                #print(f"Short TP on {df.index[i].strftime('%Y-%m-%d')}, return: {trade_return*100:.2f}%") # Debug

            #elif (df.index[i] - entry_date).days > 30 : #exit after 30days if no SL TP
            #       trade_return = (entry_price-current_price) / entry_price * -1
            #       equity += position_size * trade_return
            #       trades.append(trade_return * 100) 
            #       position = None
                #print(f"Short TimeExit TP on {df.index[i