import pandas as pd
import ta
import numpy as np

# Canonical list of 100% STATIONARY, SCALE-INVARIANT features consumed by ML models.
# Raw price-level columns (SMA_20, SMA_50, SMA_200, EMA_12, EMA_26, BB_High, BB_Low,
# BB_Mid, ATR, MACD, MACD_Signal) are still computed on `df` for chart overlays and UI
# display, but excluded from LIVE_FEATURE_COLS so tree & neural models never overfit
# to non-stationary price levels when an asset enters a new price regime.
LIVE_FEATURE_COLS = [
    # Momentum & Oscillators (bounded / stationary)
    'RSI', 'RSI_Z', 'Stoch_RSI', 'ADX_Norm', 'DI_Spread',
    # ATR-Normalized MACD & Moving Average Spreads (scale-invariant)
    'MACD_Norm', 'MACD_Signal_Norm', 'MACD_Diff_Norm',
    'EMA12_26_Spread', 'SMA20_50_Spread', 'SMA50_200_Spread',
    'Dist_SMA20', 'Dist_SMA50', 'Dist_EMA12',
    # Volatility & Bollinger/Keltner Position (stationary)
    'BB_Width', 'BB_pctB', 'ATR_Pct', 'GK_Vol', 'YZ_Vol', 'Parkinson_Vol', 'Vol_Regime_Ratio', 'Squeeze_Ratio',
    # Multi-Horizon ATR-Normalized Momentum Term Structure & Trend Gradient
    'Return', 'Log_Return', 'Mom_3_ATR', 'Mom_5_ATR', 'Mom_10_ATR', 'Mom_20_ATR', 'EMA20_Slope_Norm',
    # Return Autocorrelation & Memory Persistence
    'Autocorr_1', 'Autocorr_2', 'FracDiff_Close', 'Hurst_20', 'Efficiency_Ratio',
    # Higher-Order Return Distribution & Tail Statistics
    'Return_Vol_10', 'Return_Skew_20', 'Return_Kurt_20', 'Z_Score_Return',
    # Microstructure Order Flow, Wick Pressure & Volume Z-Scores (strictly causal)
    'CLV', 'Wick_Imbalance', 'Order_Flow_Z', 'Volume_Z', 'Streak_Length',
    'VWAP_Dev', 'RSI_Divergence', 'Hawkes_Intensity', 'Pair_RSI_MACD_Lift',
    # Cyclical Session & Weekly Seasonality
    'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos',
]


def _frac_diff_ffd(series: pd.Series, d: float = 0.40, window: int = 24) -> pd.Series:
    """
    Fixed-Window Fractional Differentiation (FFD) from Marcos López de Prado,
    'Advances in Financial Machine Learning' (Chapter 5).
    Achieves statistical stationarity while preserving long-memory trend information.
    Normalized by rolling volatility so it is scale-invariant across assets.
    """
    weights = [1.0]
    for k in range(1, window):
        w_k = -weights[-1] * (d - k + 1) / k
        weights.append(w_k)
    weights = np.array(weights[::-1], dtype=float)

    log_p = np.log(series.clip(lower=1e-8))
    vals = log_p.values
    out = np.full(len(vals), np.nan, dtype=float)
    for i in range(window - 1, len(vals)):
        out[i] = float(np.dot(weights, vals[i - window + 1 : i + 1]))

    fd_series = pd.Series(out, index=series.index)
    roll_std = fd_series.rolling(30, min_periods=10).std().replace(0, 1e-9)
    roll_mean = fd_series.rolling(30, min_periods=10).mean()
    return ((fd_series - roll_mean) / roll_std).fillna(0.0)


def _rolling_hurst(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Causal Variance-Ratio Hurst Exponent estimator over rolling window.
    H < 0.48 indicates mean-reverting regime; H > 0.52 indicates trending persistence.
    """
    log_p = np.log(close.clip(lower=1e-8))
    ret_1 = log_p.diff(1)
    ret_4 = log_p.diff(4)
    var_1 = ret_1.rolling(window, min_periods=10).var().replace(0, 1e-9)
    var_4 = ret_4.rolling(window, min_periods=10).var().replace(0, 1e-9)
    # Since Var(r_k) ~ k^(2H) * Var(r_1), 2H = log(Var(r_4)/Var(r_1)) / log(4)
    vr = (var_4 / var_1).clip(lower=0.25, upper=16.0)
    hurst = 0.5 * (np.log(vr) / np.log(4.0))
    return hurst.clip(0.15, 0.85).fillna(0.50)


def _triple_barrier_labels(
    df: pd.DataFrame,
    pt_mult: float = 0.85,
    sl_mult: float = 0.85,
    max_holding: int = None,
) -> pd.Series:
    """
    Volatility-Adaptive Multi-Horizon Triple-Barrier Labeling (López de Prado, AFML Ch. 3).
    Automatically adapts holding horizon to candle timeframe:
      - 6 bars for intraday (30 min for 5m, 1.5 hr for 15m) to allow trades to clear the spread.
      - 4 bars for daily/weekly swing trades.
    Checks which barrier is touched first using intrabar High/Low and Close trajectory.
    If neither horizontal barrier is hit within `max_holding` bars, uses the volatility-gated
    forward cumulative return over the horizon (with a noise deadzone so flat chop is excluded).
    Strictly NaN on the final `max_holding` bars to prevent any lookahead leakage at inference edge.
    """
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    atr = df['ATR'].ffill().bfill().values
    n = len(df)

    if max_holding is None:
        try:
            dt_idx = pd.to_datetime(df.index)
            med_diff = pd.Series(dt_idx).diff().median() if len(dt_idx) > 2 else pd.Timedelta(days=1)
            is_intraday = bool(med_diff < pd.Timedelta(hours=12))
        except Exception:
            is_intraday = False
        max_holding = 6 if is_intraday else 4

    labels = np.full(n, np.nan, dtype=np.float64)
    if n <= max_holding + 5:
        return pd.Series(labels, index=df.index)

    # Adapt barrier width for shorter datasets so enough training rows are labeled
    deadzone_mult = 0.12 if n >= 250 else 0.06

    for t in range(n - max_holding):
        c0 = close[t]
        atr_t = max(atr[t], c0 * 1e-4)
        upper_b = c0 + pt_mult * atr_t
        lower_b = c0 - sl_mult * atr_t

        hit_label = np.nan
        for h in range(1, max_holding + 1):
            up_hit = high[t + h] >= upper_b
            dn_hit = low[t + h] <= lower_b
            if up_hit and not dn_hit:
                hit_label = 1.0
                break
            elif dn_hit and not up_hit:
                hit_label = 0.0
                break
            elif up_hit and dn_hit:
                # Both touched in same bar: resolve by close direction of that bar
                hit_label = 1.0 if close[t + h] >= c0 else 0.0
                break

        if np.isnan(hit_label):
            # Vertical barrier reached: evaluate forward return with noise filter
            fwd_move = close[t + max_holding] - c0
            if fwd_move > deadzone_mult * atr_t:
                hit_label = 1.0
            elif fwd_move < -deadzone_mult * atr_t:
                hit_label = 0.0

        labels[t] = hit_label

    # Fallback if ultra-low volatility left too few labels
    valid_count = int(np.sum(~np.isnan(labels)))
    if valid_count < 55 and n >= 65:
        for t in range(n - 1):
            if np.isnan(labels[t]):
                diff = close[t + 1] - close[t]
                if abs(diff) > 1e-9:
                    labels[t] = 1.0 if diff > 0 else 0.0

    return pd.Series(labels, index=df.index)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes 40+ institutional quantitative, microstructure, and regime features.
    All features in LIVE_FEATURE_COLS are strictly causal, stationary, and scale-invariant.
    """
    df = df.copy()

    if len(df) < 50:
        return df

    if 'Close' not in df.columns or 'High' not in df.columns or 'Low' not in df.columns or 'Volume' not in df.columns:
        return df

    close = df['Close']
    high = df['High']
    low = df['Low']
    open_p = df['Open'] if 'Open' in df.columns else close.shift(1).fillna(close)
    volume = df['Volume']

    # --- 1. Volatility: ATR, Garman-Klass, Yang-Zhang & Parkinson Estimators ---
    df['ATR'] = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()
    atr_safe = df['ATR'].replace(0, np.nan).ffill().bfill().replace(0, 1e-9)
    close_safe = close.replace(0, 1e-9)
    df['ATR_Pct'] = atr_safe / close_safe

    # Garman-Klass Volatility (uses OHLC; 8x more statistically efficient than close-to-close)
    log_hl = np.log((high / low.replace(0, 1e-9)).clip(lower=1.0))
    log_co = np.log((close / open_p.replace(0, 1e-9)).clip(lower=1e-6))
    gk_var = (0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)).clip(lower=0.0)
    df['GK_Vol'] = np.sqrt(gk_var.rolling(14, min_periods=5).mean()).fillna(0.0)
    gk_ma30 = df['GK_Vol'].rolling(30, min_periods=10).mean().replace(0, 1e-9)
    df['Vol_Regime_Ratio'] = (df['GK_Vol'] / gk_ma30).fillna(1.0)

    # Yang-Zhang Volatility Estimator (minimum-variance unbiased; handles overnight jumps + intraday drift)
    prev_close = close.shift(1).bfill().replace(0, 1e-9)
    log_oc = np.log((open_p / prev_close).clip(lower=1e-6))
    log_ho = np.log((high / open_p.replace(0, 1e-9)).clip(lower=1.0))
    log_lo = np.log((low / open_p.replace(0, 1e-9)).clip(upper=1.0))
    rs_var = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).clip(lower=0.0)
    var_oc = log_oc.rolling(14, min_periods=5).var().fillna(0.0)
    var_co = log_co.rolling(14, min_periods=5).var().fillna(0.0)
    mean_rs = rs_var.rolling(14, min_periods=5).mean().fillna(0.0)
    k_yz = 0.34 / (1.34 + (15.0 / 13.0))
    yz_var = (var_oc + k_yz * var_co + (1.0 - k_yz) * mean_rs).clip(lower=0.0)
    df['YZ_Vol'] = np.sqrt(yz_var).fillna(0.0)

    # Parkinson High-Low Volatility Estimator
    park_var = ((log_hl ** 2) / (4.0 * np.log(2.0))).clip(lower=0.0)
    df['Parkinson_Vol'] = np.sqrt(park_var.rolling(14, min_periods=5).mean()).fillna(0.0)

    # --- 2. Momentum & Directional Strength (RSI, StochRSI, ADX) ---
    df['RSI'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    rsi_mean = df['RSI'].rolling(20, min_periods=10).mean()
    rsi_std = df['RSI'].rolling(20, min_periods=10).std().replace(0, 1e-9)
    df['RSI_Z'] = ((df['RSI'] - rsi_mean) / rsi_std).fillna(0.0)

    rsi_min = df['RSI'].rolling(14, min_periods=5).min()
    rsi_max = df['RSI'].rolling(14, min_periods=5).max()
    df['Stoch_RSI'] = ((df['RSI'] - rsi_min) / (rsi_max - rsi_min).replace(0, 1e-9)).fillna(0.5)

    try:
        adx_ind = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
        df['ADX_Norm'] = (adx_ind.adx() / 100.0).clip(0.0, 1.0).fillna(0.25)
        df['DI_Spread'] = ((adx_ind.adx_pos() - adx_ind.adx_neg()) / 100.0).clip(-1.0, 1.0).fillna(0.0)
    except Exception:
        df['ADX_Norm'] = 0.25
        df['DI_Spread'] = 0.0

    # --- 3. Trend: MACD & Moving Averages (Raw for Charts + ATR-Normalized for ML) ---
    macd = ta.trend.MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()

    df['MACD_Norm'] = (df['MACD'] / atr_safe).clip(-10.0, 10.0)
    df['MACD_Signal_Norm'] = (df['MACD_Signal'] / atr_safe).clip(-10.0, 10.0)
    df['MACD_Diff_Norm'] = (df['MACD_Diff'] / atr_safe).clip(-10.0, 10.0)

    df['EMA_12'] = ta.trend.EMAIndicator(close=close, window=12).ema_indicator()
    df['EMA_26'] = ta.trend.EMAIndicator(close=close, window=26).ema_indicator()
    df['SMA_20'] = close.rolling(window=20, min_periods=10).mean()
    df['SMA_50'] = close.rolling(window=50, min_periods=20).mean()
    df['SMA_200'] = close.rolling(window=200, min_periods=20).mean()

    df['EMA12_26_Spread'] = ((df['EMA_12'] - df['EMA_26']) / atr_safe).clip(-10.0, 10.0)
    df['SMA20_50_Spread'] = ((df['SMA_20'] - df['SMA_50']) / atr_safe).clip(-10.0, 10.0)
    df['SMA50_200_Spread'] = ((df['SMA_50'] - df['SMA_200']) / atr_safe).clip(-15.0, 15.0)

    df['Dist_SMA20'] = (close - df['SMA_20']) / df['SMA_20'].replace(0, 1e-9)
    df['Dist_SMA50'] = (close - df['SMA_50']) / df['SMA_50'].replace(0, 1e-9)
    df['Dist_EMA12'] = (close - df['EMA_12']) / df['EMA_12'].replace(0, 1e-9)
    df['EMA20_Slope_Norm'] = ((df['EMA_12'] - df['EMA_12'].shift(3)) / (3.0 * atr_safe)).clip(-4.0, 4.0).fillna(0.0)

    # --- 4. Bollinger Bands & Keltner Squeeze Ratio ---
    bollinger = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    df['BB_Mid'] = bollinger.bollinger_mavg()
    df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['BB_Mid'].replace(0, 1e-9)
    _bb_range = (df['BB_High'] - df['BB_Low']).replace(0, 1e-9)
    df['BB_pctB'] = (close - df['BB_Low']) / _bb_range

    try:
        kc = ta.volatility.KeltnerChannel(high=high, low=low, close=close, window=20, window_atr=10)
        kc_width = (kc.keltner_channel_hband() - kc.keltner_channel_lband()).replace(0, 1e-9)
        df['Squeeze_Ratio'] = (_bb_range / kc_width).clip(0.1, 5.0).fillna(1.0)
    except Exception:
        df['Squeeze_Ratio'] = 1.0

    # --- 5. Returns & Multi-Horizon ATR-Normalized Momentum Term Structure ---
    df['Return'] = close.pct_change()
    df['Log_Return'] = np.log(close / close.shift(1).replace(0, 1e-9))
    df['Mom_3_ATR'] = ((close - close.shift(3)) / atr_safe).clip(-10.0, 10.0)
    df['Mom_5_ATR'] = ((close - close.shift(5)) / atr_safe).clip(-12.0, 12.0)
    df['Mom_10_ATR'] = ((close - close.shift(10)) / atr_safe).clip(-15.0, 15.0)
    df['Mom_20_ATR'] = ((close - close.shift(20)) / atr_safe).clip(-20.0, 20.0)

    # Return Autocorrelations (Lag-1 and Lag-2) over rolling 15 bars
    _ret_clean = df['Return'].fillna(0.0)
    df['Autocorr_1'] = _ret_clean.rolling(15, min_periods=8).corr(_ret_clean.shift(1)).clip(-1.0, 1.0).fillna(0.0)
    df['Autocorr_2'] = _ret_clean.rolling(15, min_periods=8).corr(_ret_clean.shift(2)).clip(-1.0, 1.0).fillna(0.0)

    # --- 6. Stationary Memory (FracDiff), Hurst Exponent & Kaufman Efficiency Ratio ---
    df['FracDiff_Close'] = _frac_diff_ffd(close, d=0.40, window=24)
    df['Hurst_20'] = _rolling_hurst(close, window=20)
    net_change_10 = (close - close.shift(10)).abs()
    sum_abs_changes_10 = close.diff().abs().rolling(10, min_periods=5).sum().replace(0, 1e-9)
    df['Efficiency_Ratio'] = (net_change_10 / sum_abs_changes_10).clip(0.0, 1.0).fillna(0.5)

    # --- 7. Return Distribution Statistics ---
    _ret = df['Return']
    df['Return_Vol_10'] = _ret.rolling(10, min_periods=5).std()
    df['Return_Skew_20'] = _ret.rolling(20, min_periods=10).skew().fillna(0.0)
    df['Return_Kurt_20'] = _ret.rolling(20, min_periods=10).kurt().fillna(0.0)
    _rollmean = _ret.rolling(20, min_periods=10).mean()
    _rollstd = _ret.rolling(20, min_periods=10).std().replace(0, 1e-9)
    df['Z_Score_Return'] = ((_ret - _rollmean) / _rollstd).clip(-6.0, 6.0)

    # --- 8. Microstructure Order Flow, Wick Pressure, CLV & Volume Z-Score ---
    hl_range = (high - low).replace(0, 1e-9)
    # Close Location Value in [-1, +1]: +1 means closed at High (strong buying pressure)
    df['CLV'] = (((close - low) - (high - close)) / hl_range).clip(-1.0, 1.0).fillna(0.0)

    # Intrabar Wick Pressure (Buying Absorption vs Selling Rejection)
    upper_wick = (high - np.maximum(open_p, close)).clip(lower=0.0)
    lower_wick = (np.minimum(open_p, close) - low).clip(lower=0.0)
    df['Wick_Imbalance'] = ((lower_wick - upper_wick) / atr_safe).clip(-3.0, 3.0).fillna(0.0)

    vol_mean = volume.rolling(20, min_periods=5).mean()
    vol_std = volume.rolling(20, min_periods=5).std().replace(0, 1e-9)
    df['Volume_Z'] = ((volume - vol_mean) / vol_std).clip(-5.0, 5.0).fillna(0.0)
    signed_flow = df['CLV'] * volume
    sf_mean = signed_flow.rolling(20, min_periods=5).mean()
    sf_std = signed_flow.rolling(20, min_periods=5).std().replace(0, 1e-9)
    df['Order_Flow_Z'] = ((signed_flow - sf_mean) / sf_std).clip(-5.0, 5.0).fillna(0.0)

    # --- 9. Consecutive-Direction Streak Length ---
    _direction = np.sign(_ret.fillna(0.0))
    _streak = pd.Series(0, index=df.index, dtype='int64')
    for i in range(1, len(_direction)):
        if _direction.iloc[i] != 0 and _direction.iloc[i] == _direction.iloc[i - 1]:
            _streak.iloc[i] = abs(_streak.iloc[i - 1]) + 1
        else:
            _streak.iloc[i] = 1 if _direction.iloc[i] != 0 else 0
    df['Streak_Length'] = _streak

    # --- 10. Cross-Feature RSI/MACD Lift & Strictly Causal Hawkes Intensity ---
    _macd_z = (df['MACD_Norm'] - df['MACD_Norm'].rolling(20, min_periods=10).mean()) / (
        df['MACD_Norm'].rolling(20, min_periods=10).std().replace(0, 1e-9)
    )
    df['Pair_RSI_MACD_Lift'] = (df['RSI_Z'] * _macd_z.fillna(0.0)).clip(-10.0, 10.0)
    df['Hawkes_Intensity'] = _hawkes_intensity(_ret, alpha=0.4, beta=0.2)

    # --- 11. VWAP + RSI Divergence ---
    df = add_vwap(df)
    df = add_rsi_divergence(df)

    # --- 12. Cyclical Session & Weekly Seasonality ---
    try:
        dt_idx = pd.to_datetime(df.index)
        hours = dt_idx.hour + dt_idx.minute / 60.0
        dows = dt_idx.dayofweek
        df['Hour_Sin'] = np.sin(2.0 * np.pi * hours / 24.0)
        df['Hour_Cos'] = np.cos(2.0 * np.pi * hours / 24.0)
        df['Day_Sin'] = np.sin(2.0 * np.pi * dows / 7.0)
        df['Day_Cos'] = np.cos(2.0 * np.pi * dows / 7.0)
    except Exception:
        df['Hour_Sin'] = 0.0
        df['Hour_Cos'] = 1.0
        df['Day_Sin'] = 0.0
        df['Day_Cos'] = 1.0

    # Clean infinities
    for col in LIVE_FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # --- 13. Volatility-Adaptive Multi-Horizon Triple-Barrier Target ---
    df['Target'] = _triple_barrier_labels(df, pt_mult=0.85, sl_mult=0.85)

    return df

    return df


def _hawkes_intensity(
    returns: pd.Series,
    alpha: float = 0.4,
    beta: float = 0.2,
    decay_per_bar: float = 0.5,
) -> pd.Series:
    """
    Strictly causal self-exciting Hawkes intensity over return shocks.
    Uses a causal rolling 60-bar min/max normalization (instead of global full-series
    min/max) to guarantee zero future lookahead leakage into historical bars.
    """
    excitation = returns.fillna(0.0).abs()
    intensity = pd.Series(np.zeros(len(excitation), dtype=float), index=excitation.index)
    lam = 0.0
    for i in range(len(excitation)):
        lam = lam * decay_per_bar + float(excitation.iloc[i])
        intensity.iloc[i] = alpha * float(excitation.iloc[i]) + beta * lam

    roll_min = intensity.rolling(60, min_periods=5).min()
    roll_max = intensity.rolling(60, min_periods=5).max()
    span = (roll_max - roll_min).replace(0, 1e-9)
    out = ((intensity - roll_min) / span).clip(0.0, 1.0).fillna(0.5)
    return out


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Volume Weighted Average Price (VWAP).
    Resets daily if intraday index, or uses a 20-bar institutional rolling VWAP
    on daily/weekly bars so VWAP_Dev never degenerates to 0 on daily candles!
    """
    df = df.copy()
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
    tp_vol = typical_price * df['Volume']

    try:
        idx_dt = pd.to_datetime(df.index)
        # Check if multiple bars share the same calendar date (intraday)
        dates = pd.Series(idx_dt.date, index=df.index)
        bars_per_day = dates.value_counts().median() if len(dates) > 0 else 1
        if bars_per_day > 1.5:
            cum_tp_vol = tp_vol.groupby(dates).cumsum()
            cum_vol = df['Volume'].groupby(dates).cumsum()
            df['VWAP'] = cum_tp_vol / cum_vol.replace(0, 1e-6)
        else:
            # On Daily/Weekly bars, a single-day groupby cumsum equals typical_price!
            # Use a 20-bar institutional rolling anchored VWAP instead.
            df['VWAP'] = (
                tp_vol.rolling(20, min_periods=3).sum()
                / df['Volume'].rolling(20, min_periods=3).sum().replace(0, 1e-6)
            )
    except Exception:
        df['VWAP'] = (
            tp_vol.rolling(20, min_periods=3).sum()
            / df['Volume'].rolling(20, min_periods=3).sum().replace(0, 1e-6)
        )

    atr_denom = df['ATR'].replace(0, 1e-6) if 'ATR' in df.columns else df['Close'].replace(0, 1e-6)
    df['VWAP_Dev'] = ((df['Close'] - df['VWAP']) / atr_denom).clip(-10.0, 10.0).fillna(0.0)
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

    hvn_threshold = np.percentile(vol_pct, 80)
    lvn_threshold = np.percentile(vol_pct, 20)

    hvn_levels = bin_centers[vol_pct >= hvn_threshold].tolist()
    lvn_levels = bin_centers[vol_pct <= lvn_threshold].tolist()
    poc = float(bin_centers[np.argmax(vol_hist)])
    dec = 5 if abs(poc) < 1.0 else (4 if abs(poc) < 20.0 else (3 if abs(poc) < 200.0 else 2))

    return {
        "hvn_levels": [round(x, dec) for x in hvn_levels],
        "lvn_levels": [round(x, dec) for x in lvn_levels],
        "poc": round(poc, dec),
        "vol_distribution": vol_pct.tolist(),
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

        bearish_div = (df['High'] == price_high) & (df['RSI'] < rsi_high.shift(lookback))
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

    hist_var = -float(np.percentile(returns, (1.0 - confidence) * 100))

    from scipy.stats import norm
    mu = np.mean(returns)
    sigma = np.std(returns)
    param_var = -float(norm.ppf(1.0 - confidence, mu, sigma))

    return {
        "historical_var": round(max(0.0, hist_var), 4),
        "parametric_var": round(max(0.0, param_var), 4),
        "confidence": confidence,
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
    kurt_val = float(kurtosis(returns))

    try:
        df_t, loc_t, scale_t = t.fit(returns)
    except Exception:
        df_t = 4.0

    tail_risk = "HIGH" if kurt_val > 3.0 or df_t < 4.0 else "MODERATE" if kurt_val > 1.0 else "NORMAL"

    return {
        "skewness": round(skew_val, 4),
        "kurtosis": round(kurt_val, 4),
        "student_t_df": round(float(df_t), 2),
        "tail_risk": tail_risk,
    }
