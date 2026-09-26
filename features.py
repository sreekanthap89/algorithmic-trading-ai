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
    
    # Log Returns
    df['Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Add VWAP
    df = add_vwap(df)
    
    # Add RSI Divergence
    df = add_rsi_divergence(df)

    # Target: 1 if next candle's close is higher than current candle's close, else 0
    # (We shift(-1) so that today's row contains the target for tomorrow)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    return df

def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Volume Weighted Average Price (VWAP).
    Resets daily if datetime index available, otherwise cumulative.
    """
    df = df.copy()
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
    tp_vol = typical_price * df['Volume']
    
    try:
        # Group by date for daily reset if Index is DatetimeIndex
        dates = pd.to_datetime(df.index).date
        cum_tp_vol = tp_vol.groupby(dates).cumsum()
        cum_vol = df['Volume'].groupby(dates).cumsum()
        df['VWAP'] = cum_tp_vol / cum_vol.replace(0, 1e-6)
    except Exception:
        df['VWAP'] = tp_vol.cumsum() / df['Volume'].cumsum().replace(0, 1e-6)

    df['VWAP_Dev'] = (df['Close'] - df['VWAP']) / (df['ATR'].replace(0, 1e-6) if 'ATR' in df.columns else df['Close'])
    return df

def add_volume_profile(df: pd.DataFrame, n_bins: int = 20) -> dict:
    """
    Bins price into nodes and identifies High Volume Nodes (HVN) and Low Volume Nodes (LVN).
    HVN = top 20% by volume, LVN = bottom 20% by volume.
    """
    if len(df) < 5 or 'Volume' not in df.columns:
        return {"hvn_levels": [], "lvn_levels": [], "poc": float(df['Close'].iloc[-1]) if len(df) > 0 else 0.0}

    low_min = float(df['Low'].min())
    high_max = float(df['High'].max())
    if low_min == high_max:
        return {"hvn_levels": [low_min], "lvn_levels": [low_min], "poc": low_min}

    bins = np.linspace(low_min, high_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    
    vol_hist, _ = np.histogram(df['Close'], bins=bins, weights=df['Volume'])
    total_vol = vol_hist.sum() if vol_hist.sum() > 0 else 1.0
    vol_pct = vol_hist / total_vol

    # High volume nodes (top 20th percentile of bin volume)
    hvn_threshold = np.percentile(vol_pct, 80)
    lvn_threshold = np.percentile(vol_pct, 20)

    hvn_levels = bin_centers[vol_pct >= hvn_threshold].tolist()
    lvn_levels = bin_centers[vol_pct <= lvn_threshold].tolist()
    poc = float(bin_centers[np.argmax(vol_hist)])  # Point of Control

    return {
        "hvn_levels": [round(x, 2) for x in hvn_levels],
        "lvn_levels": [round(x, 2) for x in lvn_levels],
        "poc": round(poc, 2),
        "vol_distribution": vol_pct.tolist()
    }

def add_rsi_divergence(df: pd.DataFrame, lookback: int = 14) -> pd.DataFrame:
    """
    Detects RSI divergences:
    - Bearish Divergence: Price makes new high, but RSI makes lower high.
    - Bullish Divergence: Price makes new low, but RSI makes higher low.
    """
    df = df.copy()
    if 'RSI' not in df.columns:
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()

    df['RSI_Divergence'] = 0

    if len(df) >= lookback * 2:
        price_high = df['High'].rolling(lookback).max()
        rsi_high = df['RSI'].rolling(lookback).max()
        
        price_low = df['Low'].rolling(lookback).min()
        rsi_low = df['RSI'].rolling(lookback).min()

        # Bearish divergence
        bearish_div = (df['High'] == price_high) & (df['RSI'] < rsi_high.shift(lookback))
        # Bullish divergence
        bullish_div = (df['Low'] == price_low) & (df['RSI'] > rsi_low.shift(lookback))

        df.loc[bearish_div, 'RSI_Divergence'] = -1
        df.loc[bullish_div, 'RSI_Divergence'] = 1

    return df

def compute_var(df: pd.DataFrame, confidence: float = 0.95) -> dict:
    """
    Calculates Historical Value-at-Risk (VaR) and Parametric VaR from returns.
    """
    if len(df) < 5 or 'Return' not in df.columns:
        return {"historical_var": 0.0, "parametric_var": 0.0, "confidence": confidence}

    returns = df['Return'].dropna().values
    if len(returns) == 0:
        return {"historical_var": 0.0, "parametric_var": 0.0, "confidence": confidence}

    # Historical VaR
    hist_var = -float(np.percentile(returns, (1.0 - confidence) * 100))
    
    # Parametric Normal VaR
    from scipy.stats import norm
    mu = np.mean(returns)
    sigma = np.std(returns)
    param_var = -float(norm.ppf(1.0 - confidence, mu, sigma))

    return {
        "historical_var": round(max(0.0, hist_var), 4),
        "parametric_var": round(max(0.0, param_var), 4),
        "confidence": confidence
    }

def compute_correlation_matrix(symbols: list, interval: str = "1d") -> pd.DataFrame:
    """
    Computes cross-asset return correlation matrix for a list of symbols.
    """
    from data_engine import fetch_data
    returns_dict = {}

    for sym in symbols:
        try:
            df_sym = fetch_data(sym, period="6mo", interval=interval)
            if df_sym is not None and 'Close' in df_sym.columns:
                returns_dict[sym] = df_sym['Close'].pct_change()
        except Exception:
            continue

    if not returns_dict:
        return pd.DataFrame()

    combined_returns = pd.DataFrame(returns_dict).dropna()
    return combined_returns.corr(method='pearson')

def fit_return_distribution(df: pd.DataFrame) -> dict:
    """
    Fits Normal and Student's t distributions to returns to assess skewness, kurtosis, and tail risk.
    """
    if len(df) < 10 or 'Return' not in df.columns:
        return {"skewness": 0.0, "kurtosis": 0.0, "tail_risk": "LOW"}

    from scipy.stats import skew, kurtosis, t

    returns = df['Return'].dropna().values
    if len(returns) < 10:
        return {"skewness": 0.0, "kurtosis": 0.0, "tail_risk": "LOW"}

    skew_val = float(skew(returns))
    kurt_val = float(kurtosis(returns))  # Excess kurtosis

    # Fit Student-t
    try:
        df_t, loc_t, scale_t = t.fit(returns)
    except Exception:
        df_t = 4.0

    tail_risk = "HIGH" if kurt_val > 3.0 or df_t < 4.0 else "MODERATE" if kurt_val > 1.0 else "NORMAL"

    return {
        "skewness": round(skew_val, 4),
        "kurtosis": round(kurt_val, 4),
        "student_t_df": round(float(df_t), 2),
        "tail_risk": tail_risk
    }

