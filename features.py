import pandas as pd
import ta
import numpy as np

# Canonical list of features actually computed below and consumed by the model.
# Kept in sync so training and inference use identical columns.
LIVE_FEATURE_COLS = [
    'RSI', 'MACD', 'MACD_Diff', 'MACD_Signal',
    'BB_High', 'BB_Low', 'BB_Mid', 'BB_Width', 'BB_pctB',
    'ATR', 'ATR_Pct',
    'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26',
    'Dist_SMA20', 'Dist_SMA50', 'Dist_EMA12',
    'Return', 'Log_Return',
    'Return_Vol_10', 'Return_Skew_20', 'Return_Kurt_20', 'Z_Score_Return',
    'Streak_Length', 'VWAP_Dev', 'RSI_Divergence',
    'Hawkes_Intensity', 'Pair_RSI_MACD_Lift',
]


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

    close = df['Close']

    # --- Momentum ---
    df['RSI'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    # --- Trend: MACD (with histogram) ---
    macd = ta.trend.MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()   # MACD histogram

    # --- Trend: EMAs (MACD components) ---
    df['EMA_12'] = ta.trend.EMAIndicator(close=close, window=12).ema_indicator()
    df['EMA_26'] = ta.trend.EMAIndicator(close=close, window=26).ema_indicator()

    # --- Volatility: Bollinger Bands ---
    bollinger = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    df['BB_Mid'] = bollinger.bollinger_mavg()
    _bb_width = (df['BB_High'] - df['BB_Low']) / df['BB_Mid'].replace(0, 1e-9)
    df['BB_Width'] = _bb_width
    # %B: position within the bands (1 = upper, 0 = lower)
    _bb_range = (df['BB_High'] - df['BB_Low']).replace(0, 1e-9)
    df['BB_pctB'] = (close - df['BB_Low']) / _bb_range

    # --- Volatility: ATR (absolute and normalized) ---
    df['ATR'] = ta.volatility.AverageTrueRange(
        high=df['High'], low=df['Low'], close=close, window=14).average_true_range()
    df['ATR_Pct'] = df['ATR'] / close.replace(0, 1e-9)

    # --- Trend: moving averages ---
    df['SMA_20'] = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
    df['SMA_50'] = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
    if len(df) >= 200:
        df['SMA_200'] = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()
    else:
        df['SMA_200'] = np.nan

    # --- Distance-to-trend (trend-strength / mean-reversion features) ---
    df['Dist_SMA20'] = (close - df['SMA_20']) / df['SMA_20'].replace(0, 1e-9)
    df['Dist_SMA50'] = (close - df['SMA_50']) / df['SMA_50'].replace(0, 1e-9)
    df['Dist_EMA12'] = (close - df['EMA_12']) / df['EMA_12'].replace(0, 1e-9)

    # --- Returns & log returns ---
    df['Return'] = close.pct_change()
    df['Log_Return'] = np.log(close / close.shift(1))

    # --- Return distribution statistics (rolling) ---
    _ret = df['Return']
    df['Return_Vol_10'] = _ret.rolling(10).std()
    df['Return_Skew_20'] = _ret.rolling(20).skew()
    df['Return_Kurt_20'] = _ret.rolling(20).kurt()
    _rollmean = _ret.rolling(20).mean()
    _rollstd = _ret.rolling(20).std().replace(0, 1e-9)
    df['Z_Score_Return'] = (_ret - _rollmean) / _rollstd

    # --- Consecutive-direction streak length (momentum persistence) ---
    _direction = np.sign(_ret.fillna(0.0))
    _streak = pd.Series(0, index=df.index, dtype='int64')
    for i in range(1, len(_direction)):
        if _direction.iloc[i] != 0 and _direction.iloc[i] == _direction.iloc[i - 1]:
            _streak.iloc[i] = abs(_streak.iloc[i - 1]) + 1
        else:
            _streak.iloc[i] = 1 if _direction.iloc[i] != 0 else 0
    df['Streak_Length'] = _streak

    # --- Cross-feature: RSI/MACD z-score lift (confluence) ---
    _rsi_z = (df['RSI'] - df['RSI'].rolling(20).mean()) / df['RSI'].rolling(20).std().replace(0, 1e-9)
    _macd_z = (df['MACD'] - df['MACD'].rolling(20).mean()) / df['MACD'].rolling(20).std().replace(0, 1e-9)
    df['Pair_RSI_MACD_Lift'] = _rsi_z * _macd_z

    # --- Hawkes self-exciting intensity (approx. via EW of positive returns) ---
    df['Hawkes_Intensity'] = _hawkes_intensity(_ret, alpha=0.4, beta=0.2)

    # --- VWAP + RSI divergence (already-implemented helpers) ---
    df = add_vwap(df)
    df = add_rsi_divergence(df)

    # Fill any leftover NaN/inf so downstream training can drop rows cleanly
    for col in LIVE_FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # Target: direction of the next close, gated by an ATR threshold so the model
    # only learns economically meaningful moves. Neutral bars (|move| < 0.5*ATR)
    # are left NaN and excluded by dropna, giving a clean up-vs-down signal that
    # matches the BUY/SELL prob_up thresholds used downstream.
    _atr_ref = df['ATR'].replace(0, 1e-9)
    _next_move = close.shift(-1) - close
    _thr = 0.5 * _atr_ref
    df['Target'] = np.float64(np.nan)
    df.loc[_next_move > _thr, 'Target'] = 1.0
    df.loc[_next_move < -_thr, 'Target'] = 0.0

    return df


def _hawkes_intensity(returns: pd.Series, alpha: float = 0.4, beta: float = 0.2,
                      decay_per_bar: float = 0.5) -> pd.Series:
    """Simple self-exciting Hawkes-style intensity: an exponential-weighted rate
    of *up-ticks* that excites intensity now. Returns a normalized 0..1 series."""
    excitation = np.maximum(returns, 0.0)  # only up-moves excite
    intensity = pd.Series(np.zeros(len(excitation)), index=excitation.index)
    lam = 0.0
    for i in range(len(excitation)):
        lam = lam * decay_per_bar + excitation.iloc[i]
        intensity.iloc[i] = alpha * excitation.iloc[i] + beta * lam
    out = intensity
    span = out.max() - out.min()
    if span and span > 0:
        out = (out - out.min()) / span
    else:
        out = pd.Series(0.5, index=excitation.index)
    return out


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

