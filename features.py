import pandas as pd
import ta
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators to the dataframe.
    These act as the "Frequency Analysis" features for our AI to learn from.
    """
    df = df.copy()
    
    # Check if we have enough data
    if len(df) < 50:
        return df

    # Ensure required columns exist
    if 'Close' not in df.columns or 'High' not in df.columns or 'Low' not in df.columns or 'Volume' not in df.columns:
        return df

    # Momentum Indicator: RSI
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    
    # Trend Indicator: MACD
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    # Volatility Indicator: Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    df['BB_Mid'] = bollinger.bollinger_mavg()
    
    # Average True Range (ATR) for volatility
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    
    # Moving Averages
    df['SMA_20'] = ta.trend.SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['SMA_50'] = ta.trend.SMAIndicator(close=df['Close'], window=50).sma_indicator()
    
    # Target: 1 if next candle's close is higher than current candle's close, else 0
    # (We shift(-1) so that today's row contains the target for tomorrow)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # The last row will have NaN for target, we keep it for prediction but drop it for training
    df['Return'] = df['Close'].pct_change()

    return df
