"""
cot_engine/indicators.py — Mathematical Oscillators & Technical Indicators
from Larry Williams' 'Trade Stocks and Commodities With the Insiders' (Wiley, 2005).

All indicators are normalized, vectorized, and equipped with rigorous numerical guards
(zero-division handling, epsilon bounds, boundary safety [0, 100], and NaN handling).
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# Inside-OI Diagnostic State Definitions (Chapter 9)
INSIDE_OI_STATES: Dict[int, str] = {
    0: "NEUTRAL",
    1: "SPECULATIVE_TOP",
    2: "COMMERCIAL_CAPITULATION",
    3: "DESPAIR_BLASTOFF",
    4: "HOLLOW_RALLY",
}


def _stochastic_normalize(
    series: pd.Series,
    window: int,
    min_periods: Optional[int] = None,
    eps: float = 1e-9,
) -> pd.Series:
    """
    Standard Stochastic %K normalization over rolling window:
    100 * (x - min(x)) / (max(x) - min(x))

    Mathematical specifications:
      - Normalization window includes current bar t.
      - Safe epsilon guard for flat series (max == min): returns 50.0.
      - Output is strictly bounded in [0.0, 100.0].
      - Strict NaN preservation during warm-up and on missing inputs.
    """
    min_p = min_periods if min_periods is not None else window
    rolling_min = series.rolling(window=window, min_periods=min_p).min()
    rolling_max = series.rolling(window=window, min_periods=min_p).max()
    denom = rolling_max - rolling_min

    valid = series.notna() & rolling_min.notna() & rolling_max.notna()
    stoch = pd.Series(np.nan, index=series.index, dtype=float)

    # Flat range epsilon guard: if max == min, output neutral 50.0
    flat_mask = valid & (denom <= eps)
    stoch[flat_mask] = 50.0

    # Normalization where valid range exists
    norm_mask = valid & (denom > eps)
    norm_val = 100.0 * (series[norm_mask] - rolling_min[norm_mask]) / denom[norm_mask]
    stoch[norm_mask] = np.clip(norm_val, 0.0, 100.0)

    series_name = series.name if series.name is not None else "val"
    return pd.Series(stoch, index=series.index, name=f"{series_name}_stoch_{window}")


def commercial_net_position(df: pd.DataFrame) -> pd.Series:
    """
    Commercial Net Position (CP) = Commercial Long - Commercial Short.
    Axiom: Commercials are structurally net short in commodities.
    Zero-line crossovers are rare; relative level matters.
    """
    cols = {c.lower(): c for c in df.columns}
    if "comm_long" in cols and "comm_short" in cols:
        cl = df[cols["comm_long"]]
        cs = df[cols["comm_short"]]
    elif "comm_long" in df.columns and "comm_short" in df.columns:
        cl = df["comm_long"]
        cs = df["comm_short"]
    else:
        raise KeyError("DataFrame must contain 'comm_long' and 'comm_short' columns")
    return (cl - cs).rename("comm_net")


def commercial_share_oi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Commercial concentration as a percentage of Total Open Interest (COT-02 & COT-03):
      - comm_long_pct_oi: Commercial Long / OI * 100
      - comm_short_pct_oi: Commercial Short / OI * 100
      - comm_net_pct_oi: (Commercial Long - Commercial Short) / OI * 100
      - comm_short_extreme_55: Flag when comm_short_pct_oi >= 55% (Williams yen/lumber top alert)
      - comm_short_bottom_30: Flag when comm_short_pct_oi < 30% (Williams bottom alert, Ch 10)
      - comm_long_dominant_50: Flag when comm_long_pct_oi > 50% (Williams wheat dominance, Ch 2)
      - comm_short_dominant_50: Flag when comm_short_pct_oi > 50% (Commercial short dominance)
    """
    cols = {c.lower(): c for c in df.columns}
    oi_col = cols.get("open_interest", "open_interest")
    if oi_col not in df.columns:
        raise KeyError("DataFrame must contain 'open_interest' column")

    # Guard against zero or negative open interest
    oi = df[oi_col].where(df[oi_col] > 0, np.nan)
    cl = df[cols["comm_long"]] if "comm_long" in cols else df["comm_long"]
    cs = df[cols["comm_short"]] if "comm_short" in cols else df["comm_short"]

    cl_pct = (cl / oi) * 100.0
    cs_pct = (cs / oi) * 100.0
    net_pct = ((cl - cs) / oi) * 100.0
    cs_extreme = (cs_pct >= 55.0) & cs_pct.notna()
    cs_bottom = (cs_pct < 30.0) & cs_pct.notna()
    cl_dominant_50 = (cl_pct > 50.0) & cl_pct.notna()
    cs_dominant_50 = (cs_pct > 50.0) & cs_pct.notna()

    return pd.DataFrame({
        "comm_long_pct_oi": cl_pct,
        "comm_short_pct_oi": cs_pct,
        "comm_net_pct_oi": net_pct,
        "comm_short_extreme_55": cs_extreme.astype(int),
        "comm_short_bottom_30": cs_bottom.astype(int),
        "comm_long_dominant_50": cl_dominant_50.astype(int),
        "comm_short_dominant_50": cs_dominant_50.astype(int),
    }, index=df.index)


def commercial_net_flow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Commercial Weekly Net Flow / Velocity (COT-04, Chapters 2 & 4):
    Measures weekly pace and direction of commercial position changes:
      - delta_comm_net: Weekly change in commercial net position (CP_t - CP_{t-1})
      - delta_comm_net_pct_oi: Weekly change normalized by Open Interest ((delta_comm_net / OI_t) * 100)
      - comm_flow_buying: 1 if delta_comm_net > 0 (weekly net commercial accumulation)
      - comm_flow_selling: 1 if delta_comm_net < 0 (weekly net commercial liquidation)
      - comm_exhaustion_divergence: 1 if commercials remain Net Long (CP_t > 0)
        while liquidating on a weekly basis (delta_comm_net < 0) [Momentum Exhaustion]
    """
    cp = commercial_net_position(df)
    delta_cp = cp.diff(1)

    cols = {c.lower(): c for c in df.columns}
    oi_col = cols.get("open_interest", "open_interest")
    if oi_col in df.columns:
        oi = df[oi_col].where(df[oi_col] > 0, np.nan)
        delta_pct = (delta_cp / oi) * 100.0
    else:
        delta_pct = pd.Series(np.nan, index=df.index)

    valid = delta_cp.notna() & cp.notna()
    buying = (valid & (delta_cp > 0)).astype(int)
    selling = (valid & (delta_cp < 0)).astype(int)
    exhaustion = (valid & (cp > 0) & (delta_cp < 0)).astype(int)

    return pd.DataFrame({
        "delta_comm_net": delta_cp,
        "delta_comm_net_pct_oi": delta_pct,
        "comm_flow_buying": buying,
        "comm_flow_selling": selling,
        "comm_exhaustion_divergence": exhaustion,
    }, index=df.index)


commercial_weekly_flow = commercial_net_flow


def williams_cot_index(
    df: pd.DataFrame,
    lookback_weeks: int = 156,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """
    The Classic Larry Williams COT Index:
    COT_Index = 100 * (CP - min_N(CP)) / (max_N(CP) - min_N(CP))
    Lookbacks:
      - 156 weeks (3-year baseline, Williams classic)
      - 52 weeks (1-year intermediate)
      - 26 weeks (6-month swing lookback, Chapters 10 & 13)
    Thresholds:
      >= 80% (or 75%): Bullish Setup (Commercials buying)
      <= 20% (or 25%): Bearish Setup (Commercials selling)
    """
    cp = commercial_net_position(df)
    cp.name = "comm_net"
    index_series = _stochastic_normalize(cp, window=lookback_weeks, min_periods=min_periods)
    index_series.name = f"cot_index_{lookback_weeks}w"
    return index_series


def williams_willco(
    df: pd.DataFrame,
    lookback_weeks: int = 26,
    variant: str = "net",
    min_periods: Optional[int] = None,
) -> pd.Series:
    """
    The WILLCO (Williams Commercial Index) Indicator (Chapter 10):
    Normalizes Commercial positioning as a percentage of Open Interest over an N-week window.
    variant:
      - 'net': Stochastic of (Commercial Net / OI) [Williams default]
      - 'short': Inverted Stochastic of (Commercial Short / OI) [100 - Stoch(CS/OI)]
    Thresholds:
      >= 75% - 80%: Buy Setup Zone (>=90% in Gold May 2004)
      <= 20% - 25%: Sell Setup Zone
    """
    if variant not in ("net", "short"):
        raise ValueError(f"Unknown WILLCO variant: '{variant}'. Must be 'net' or 'short'.")

    cols = {c.lower(): c for c in df.columns}
    oi_col = cols.get("open_interest", "open_interest")
    if oi_col not in df.columns:
        raise KeyError("DataFrame must contain 'open_interest' column")

    oi = df[oi_col].where(df[oi_col] > 0, np.nan)
    cl = df[cols["comm_long"]] if "comm_long" in cols else df["comm_long"]
    cs = df[cols["comm_short"]] if "comm_short" in cols else df["comm_short"]

    if variant == "short":
        ratio = (cs / oi) * 100.0
        ratio.name = "comm_short_pct"
        stoch = _stochastic_normalize(ratio, window=lookback_weeks, min_periods=min_periods)
        willco = np.clip(100.0 - stoch, 0.0, 100.0)
    else:
        ratio = ((cl - cs) / oi) * 100.0
        ratio.name = "comm_net_pct"
        willco = _stochastic_normalize(ratio, window=lookback_weeks, min_periods=min_periods)

    willco.name = f"willco_{lookback_weeks}w_{variant}"
    return willco


def multi_year_extremes(
    df: pd.DataFrame,
    lookback_weeks: Union[int, Sequence[int]] = 208,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    Multi-Year Extremes ("Lesson One", COT-05, Chapter 3):
    Flags when Commercial Net Position reaches a multi-year high or low.
    Excludes the current bar from the historical rolling baseline via shift(1).
    Standard lookbacks:
      - 208 weeks (4-Year Rule of Thumb, Williams default)
      - 156 weeks (3-Year Macro lookback)
      - 260 weeks (5-Year Macro lookback — S&P 1998/1999)
      - 312 weeks (6-Year Macro lookback — Gold 2004 top)
      - 520 weeks (10-Year Macro lookback — British Pound 2000)
      - 728 weeks (14-Year Macro lookback — British Pound 1999)
    Returns binary flags:
      - comm_{N}w_extreme_high: 1 if current net >= max over past N weeks (with non-flat range)
      - comm_{N}w_extreme_low: 1 if current net <= min over past N weeks (with non-flat range)
    """
    cp = commercial_net_position(df)
    windows = [lookback_weeks] if isinstance(lookback_weeks, int) else list(lookback_weeks)
    res = {}

    for w in windows:
        min_p = min_periods if min_periods is not None else w
        prior_max = cp.shift(1).rolling(window=w, min_periods=min_p).max()
        prior_min = cp.shift(1).rolling(window=w, min_periods=min_p).min()

        valid = cp.notna() & prior_max.notna() & prior_min.notna()
        has_range = (prior_max - prior_min) > 1e-9

        is_high = valid & has_range & (cp >= prior_max)
        is_low = valid & has_range & (cp <= prior_min)

        res[f"comm_{w}w_extreme_high"] = is_high.astype(int)
        res[f"comm_{w}w_extreme_low"] = is_low.astype(int)

    return pd.DataFrame(res, index=df.index)


def perpetual_short_regime_breakout(
    df: pd.DataFrame,
    lookback_weeks: int = 260,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    Perpetual Short Regime Breakout ('The Silver Rule', COT-07, Chapter 3):
    Flags a rare secular mega-bull regime trigger when commercial hedgers in a
    structurally net-short market (e.g., Silver, Coffee, Cocoa) flip to net long (CP_t > 0)
    after being continuously non-positive (max(CP_{t-k}) <= 0) over the prior M weeks.

    Mathematical Logic:
      RegimeBreakout_{Long, t} = 1 if (CP_t > 0 and max_{1 <= k <= M}(CP_{t-k}) <= 0 and min_{1 <= k <= M}(CP_{t-k}) < 0) else 0

    Standard lookback M:
      - 260 weeks (5 years benchmark, up to 1000+ weeks for Silver)
    Returns DataFrame:
      - silver_rule_breakout: Binary trigger (1 on bar of breakout)
      - perpetual_short_regime_breakout: Alias for silver_rule_breakout
    """
    cp = commercial_net_position(df)
    min_p = min_periods if min_periods is not None else lookback_weeks

    prior_max = cp.shift(1).rolling(window=lookback_weeks, min_periods=min_p).max()
    prior_min = cp.shift(1).rolling(window=lookback_weeks, min_periods=min_p).min()

    valid = cp.notna() & prior_max.notna() & prior_min.notna()
    # Breakout condition: current net position is strictly positive (CP > 0),
    # while for all prior M weeks commercial net position never exceeded zero (max <= 0),
    # and was actively short at some point (min < 0).
    breakout = valid & (cp > 0) & (prior_max <= 0) & (prior_min < 0)

    breakout_series = breakout.astype(int)
    return pd.DataFrame({
        "silver_rule_breakout": breakout_series,
        "perpetual_short_regime_breakout": breakout_series,
    }, index=df.index)


def cot_swing_velocity(
    df: pd.DataFrame,
    velocity_window: int = 26,
    index_lookback: int = 156,
    rapid_threshold: float = 75.0,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    26-Week COT Swing Velocity (COT-09, Chapters 1 & 4):
    Quantifies the institutional rotation velocity of Commercials over a 26-week horizon:
      - cot_swing_velocity_26w: COT_Index_t(156w) - COT_Index_{t-26}(156w)
      - cot_net_contract_swing_26w: CP_t - CP_{t-26}
      - cot_rapid_swing_alert: 1 if |cot_swing_velocity_26w| >= 75.0%
      - cot_rapid_bull_swing: 1 if cot_swing_velocity_26w >= +75.0% (Bullish rotation)
      - cot_rapid_bear_swing: 1 if cot_swing_velocity_26w <= -75.0% (Bearish rotation)
    """
    cot_idx = williams_cot_index(df, lookback_weeks=index_lookback, min_periods=min_periods)
    cp = commercial_net_position(df)

    swing_vel = cot_idx - cot_idx.shift(velocity_window)
    contract_swing = cp - cp.shift(velocity_window)

    valid = swing_vel.notna()
    rapid_alert = (valid & (swing_vel.abs() >= rapid_threshold)).astype(int)
    rapid_bull = (valid & (swing_vel >= rapid_threshold)).astype(int)
    rapid_bear = (valid & (swing_vel <= -rapid_threshold)).astype(int)

    return pd.DataFrame({
        f"cot_swing_velocity_{velocity_window}w": swing_vel,
        f"cot_net_contract_swing_{velocity_window}w": contract_swing,
        "cot_rapid_swing_alert": rapid_alert,
        "cot_rapid_bull_swing": rapid_bull,
        "cot_rapid_bear_swing": rapid_bear,
    }, index=df.index)


def public_contrary_index(
    df: pd.DataFrame,
    lookback_weeks: int = 156,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """
    Small Speculator / Non-Reportable Contrary Index (Chapter 5):
    Stochastic of (Nonreportable Long - Nonreportable Short).
    Contrarian rules:
      >= 75%: Peak retail optimism -> Short Setup
      <= 25%: Retail capitulation -> Buy Setup (ONLY if commercials agree)
    """
    cols = {c.lower(): c for c in df.columns}
    if "nonrep_long" not in cols or "nonrep_short" not in cols:
        raise KeyError("DataFrame must contain 'nonrep_long' and 'nonrep_short' columns")
    small_net = df[cols["nonrep_long"]] - df[cols["nonrep_short"]]
    small_net.name = "small_net"
    idx = _stochastic_normalize(small_net, window=lookback_weeks, min_periods=min_periods)
    idx.name = f"public_index_{lookback_weeks}w"
    return idx


def large_trader_index(
    df: pd.DataFrame,
    lookback_weeks: int = 156,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """
    Large Trader / Non-Commercial Fund Exhaustion Index (Chapter 6):
    Stochastic of (Noncommercial Long - Noncommercial Short).
    Rules:
      >= 80%: Funds fully long ('car out of gas') -> Top warning / Fade Funds
      <= 20%: Funds peak short -> Bottom warning
    """
    cols = {c.lower(): c for c in df.columns}
    if "noncomm_long" not in cols or "noncomm_short" not in cols:
        raise KeyError("DataFrame must contain 'noncomm_long' and 'noncomm_short' columns")
    large_net = df[cols["noncomm_long"]] - df[cols["noncomm_short"]]
    large_net.name = "large_net"
    idx = _stochastic_normalize(large_net, window=lookback_weeks, min_periods=min_periods)
    idx.name = f"large_index_{lookback_weeks}w"
    return idx


def open_interest_stochastic(
    df: pd.DataFrame,
    lookback_weeks: int = 52,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """
    12-Month Open Interest Stochastic (Chapter 8):
    Stochastic of Open Interest over a 52-week rolling window.
    Interpretation:
      >= 80%: Crowd crowded in -> Speculative Top risk
      <= 20%: Low participation -> Pure commercial base / Bullish bottom
    """
    target_oi = None
    for cand in ["open_interest", "Open_Interest", "Open Interest", "oi", "OI"]:
        if cand in df.columns:
            target_oi = cand
            break
    if target_oi is None:
        raise KeyError("DataFrame must contain 'open_interest' column")

    oi = df[target_oi].where(df[target_oi] > 0, np.nan).copy()
    oi.name = "open_interest"
    stoch = _stochastic_normalize(oi, window=lookback_weeks, min_periods=min_periods)
    stoch.name = f"oi_stoch_{lookback_weeks}w"
    return stoch


def inside_oi_regime(
    df: pd.DataFrame,
    price_col: str = "Close",
) -> pd.Series:
    """
    Inside Open Interest 4-State Diagnostic Matrix (Chapter 9):
    Classifies market state based on weekly delta of Price, OI, and Commercial Net.
    States:
      1 = 'SPECULATIVE_TOP': Price UP, OI UP, Commercial Net DOWN (Crowd chasing)
      2 = 'COMMERCIAL_CAPITULATION': Price DOWN, OI UP, Commercial Net UP (Hedgers absorbing panic)
      3 = 'DESPAIR_BLASTOFF': Price DOWN, OI DOWN, Commercial Net UP (Retail dumping, Hedgers accumulating)
      4 = 'HOLLOW_RALLY': Price UP, OI DOWN, Commercial Net DOWN (Short-covering rally)
      0 = 'NEUTRAL'
    """
    cp = commercial_net_position(df)
    delta_cp = cp.diff(1)

    target_oi = None
    for cand in ["open_interest", "Open_Interest", "Open Interest", "oi", "OI"]:
        if cand in df.columns:
            target_oi = cand
            break
    if target_oi is not None:
        delta_oi = df[target_oi].diff(1)
    else:
        delta_oi = pd.Series(0.0, index=df.index)

    target_price = None
    for cand in [price_col, price_col.lower(), price_col.capitalize(), price_col.upper()]:
        if cand in df.columns:
            target_price = cand
            break
    if target_price is not None:
        delta_p = df[target_price].diff(1)
    else:
        delta_p = pd.Series(0.0, index=df.index)

    p_up = delta_p > 0
    p_dn = delta_p < 0
    oi_up = delta_oi > 0
    oi_dn = delta_oi < 0
    cp_up = delta_cp > 0
    cp_dn = delta_cp < 0

    state = pd.Series(0, index=df.index, name="inside_oi_regime")
    state[p_up & oi_up & cp_dn] = 1   # SPECULATIVE_TOP
    state[p_dn & oi_up & cp_up] = 2   # COMMERCIAL_CAPITULATION
    state[p_dn & oi_dn & cp_up] = 3   # DESPAIR_BLASTOFF
    state[p_up & oi_dn & cp_dn] = 4   # HOLLOW_RALLY

    return state


def progo_oscillator(
    df: pd.DataFrame,
    window: int = 14,
    div_lookback: int = 20,
) -> pd.DataFrame:
    """
    ProGo & Public Index (Chapters 10 & 14):
    ProGo = 14-day average of (Close - Open) [Institutional accumulation/distribution]
    Public = 14-day average of (Open - Prior Close) [Retail emotional opening trades]
    Divergences:
      - progo_bull_divergence: Price making 20-day low while ProGo is rising and above prior low
      - progo_bear_divergence: Price making 20-day high while ProGo is falling and below prior high
    """
    open_s = None
    close_s = None
    for cand in ["Open", "open", "OPEN"]:
        if cand in df.columns:
            open_s = df[cand]
            break
    for cand in ["Close", "close", "CLOSE"]:
        if cand in df.columns:
            close_s = df[cand]
            break
    if open_s is None or close_s is None:
        raise KeyError("DataFrame must contain 'Open' and 'Close' for ProGo calculation")

    pro = close_s - open_s
    public = open_s - close_s.shift(1)

    progo = pro.rolling(window=window).mean()
    public_idx = public.rolling(window=window).mean()

    # Detect divergences over div_lookback lookback
    price_low = close_s.rolling(div_lookback).min()
    price_high = close_s.rolling(div_lookback).max()
    progo_low = progo.rolling(div_lookback).min()
    progo_high = progo.rolling(div_lookback).max()

    valid = close_s.notna() & progo.notna() & price_low.notna() & progo_low.shift(1).notna()
    bull_div = valid & (close_s <= price_low) & (progo > progo_low.shift(1)) & (progo > progo.shift(1))
    bear_div = valid & (close_s >= price_high) & (progo < progo_high.shift(1)) & (progo < progo.shift(1))

    return pd.DataFrame({
        "progo": progo,
        "public_index": public_idx,
        "progo_bull_divergence": bull_div.astype(int),
        "progo_bear_divergence": bear_div.astype(int),
    }, index=df.index)


def jiler_seasonal_deviation(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    lookback_years: int = 6,
    min_prior_obs: int = 3,
) -> pd.DataFrame:
    """
    Bill Jiler CRB Seasonal Deviation (Chapters 1 & 4):
    Deviation of Commercial % of Open Interest from historical month-of-year average.
    Calculates Point-in-Time (PIT) expanding norm strictly using prior calendar years.
    Thresholds:
      - Noise: |dev| < 5%
      - Caution: |dev| > 40%
    """
    cols = {c.lower(): c for c in df.columns}
    oi_col = cols.get("open_interest", "open_interest")
    if oi_col not in df.columns:
        raise KeyError("DataFrame must contain 'open_interest' column")

    oi = df[oi_col].where(df[oi_col] > 0, np.nan)
    cl = df[cols["comm_long"]] if "comm_long" in cols else df["comm_long"]
    cs = df[cols["comm_short"]] if "comm_short" in cols else df["comm_short"]

    net_pct = ((cl - cs) / oi) * 100.0

    if date_col and date_col in df.columns:
        dt_idx = pd.DatetimeIndex(pd.to_datetime(df[date_col]))
    elif isinstance(df.index, pd.DatetimeIndex):
        dt_idx = df.index
    elif "report_date" in df.columns:
        dt_idx = pd.DatetimeIndex(pd.to_datetime(df["report_date"]))
    elif "report_date" in cols:
        dt_idx = pd.DatetimeIndex(pd.to_datetime(df[cols["report_date"]]))
    else:
        dt_idx = pd.date_range("2000-01-01", periods=len(df), freq="W")

    months = dt_idx.month.values
    years = dt_idx.year.values
    net_pct_vals = net_pct.values
    n = len(df)
    monthly_mean = np.full(n, np.nan, dtype=float)

    # Expanding PIT calculation of monthly norm using historical prior years
    for i in range(n):
        cur_m = months[i]
        cur_y = years[i]
        prior_m = months[:i]
        prior_y = years[:i]
        # Only use prior years within [cur_y - lookback_years, cur_y - 1]
        mask = (prior_m == cur_m) & (prior_y < cur_y) & (prior_y >= cur_y - lookback_years)
        valid_mask = mask & ~np.isnan(net_pct_vals[:i])
        if np.sum(valid_mask) >= min_prior_obs:
            monthly_mean[i] = np.mean(net_pct_vals[:i][valid_mask])

    monthly_mean_series = pd.Series(monthly_mean, index=df.index, name="jiler_seasonal_norm")
    dev = net_pct - monthly_mean_series

    # Flags: 0 during warm-up (when dev is NaN)
    valid_dev = dev.notna()
    is_noise = valid_dev & (dev.abs() < 5.0)
    is_caution = valid_dev & (dev.abs() > 40.0)
    is_bullish = valid_dev & (dev > 20.0)
    is_bearish = valid_dev & (dev < -20.0)

    return pd.DataFrame({
        "jiler_seasonal_norm": monthly_mean_series,
        "jiler_deviation": dev,
        "jiler_noise_flag": is_noise.astype(int),
        "jiler_caution_flag": is_caution.astype(int),
        "jiler_bullish_flag": is_bullish.astype(int),
        "jiler_bearish_flag": is_bearish.astype(int),
    }, index=df.index)


def compute_all_cot_features(
    df: pd.DataFrame,
    include_daily_progo: bool = True,
    include_jiler: bool = True,
    multi_year_lookbacks: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """
    Master pipeline computing all Larry Williams COT indicators on a DataFrame.
    Returns the enriched DataFrame with all indicators attached.
    """
    out = df.copy()

    # Core Commercial Metrics (COT-01, COT-02, COT-03)
    out["comm_net"] = commercial_net_position(out)
    comm_share = commercial_share_oi(out)
    for col in comm_share.columns:
        out[col] = comm_share[col]

    # COT-04: Commercial Weekly Net Flow
    flow_df = commercial_net_flow(out)
    for col in flow_df.columns:
        out[col] = flow_df[col]

    # COT-06: Williams COT Indexes (156w, 52w, 26w) & Bull/Bear Setup Zones
    for w in (156, 52, 26):
        idx_series = williams_cot_index(out, lookback_weeks=w)
        out[f"cot_index_{w}w"] = idx_series
        out[f"cot_index_{w}w_bull"] = (idx_series.notna() & (idx_series >= 80.0)).astype(int)
        out[f"cot_index_{w}w_bear"] = (idx_series.notna() & (idx_series <= 20.0)).astype(int)

    # COT-09: 26-Week COT Swing Velocity
    vel_df = cot_swing_velocity(out, velocity_window=26, index_lookback=156)
    for col in vel_df.columns:
        out[col] = vel_df[col]

    # COT-07: The Silver Rule (Perpetual Short Regime Breakout)
    silver_df = perpetual_short_regime_breakout(out, lookback_weeks=260)
    for col in silver_df.columns:
        out[col] = silver_df[col]

    # COT-05: Multi-Year Extremes (208w / 4-yr, plus 3-yr, 5-yr, 6-yr, 10-yr, 14-yr lookbacks)
    my_lookbacks = multi_year_lookbacks or [208, 156, 260, 312, 520, 728]
    extremes_df = multi_year_extremes(out, lookback_weeks=my_lookbacks)
    for col in extremes_df.columns:
        out[col] = extremes_df[col]

    # WILLCO
    out["willco_26w_net"] = williams_willco(out, lookback_weeks=26, variant="net")
    out["willco_26w_short"] = williams_willco(out, lookback_weeks=26, variant="short")

    # Speculator Sentiment
    cols = {c.lower(): c for c in out.columns}
    if "nonrep_long" in cols and "nonrep_short" in cols:
        out["public_index_156w"] = public_contrary_index(out, lookback_weeks=156)
        out["public_index_26w"] = public_contrary_index(out, lookback_weeks=26)

    if "noncomm_long" in cols and "noncomm_short" in cols:
        out["large_index_156w"] = large_trader_index(out, lookback_weeks=156)
        out["large_index_26w"] = large_trader_index(out, lookback_weeks=26)

    # Open Interest Dynamics
    out["oi_stoch_52w"] = open_interest_stochastic(out, lookback_weeks=52)
    out["inside_oi_regime"] = inside_oi_regime(out)

    # ProGo (if OHLC available)
    if include_daily_progo and ("Close" in cols or "close" in cols) and ("Open" in cols or "open" in cols):
        progo_df = progo_oscillator(out, window=14)
        for col in progo_df.columns:
            out[col] = progo_df[col]

    # COT-08: Jiler Seasonal Deviation
    if include_jiler and ("report_date" in cols or isinstance(out.index, pd.DatetimeIndex)):
        try:
            jiler_df = jiler_seasonal_deviation(out)
            for col in jiler_df.columns:
                out[col] = jiler_df[col]
        except Exception:
            pass

    return out


# =============================================================================
# Chapters 11: Equities & Aggregated Commercial Indicators
# =============================================================================

DEFAULT_INDEX_MULTIPLIERS: Dict[str, float] = {
    "ES": 50.0,    # E-mini S&P 500
    "SP": 250.0,   # Big S&P 500
    "NQ": 20.0,    # E-mini Nasdaq 100
    "ND": 100.0,   # Big Nasdaq 100
    "YM": 5.0,     # E-mini Dow Jones
    "DJ": 25.0,    # Big Dow Jones
}


def aggregated_dollar_commercial_position(
    market_data: Dict[str, pd.DataFrame],
    multipliers: Optional[Dict[str, float]] = None,
    price_cols: Optional[Dict[str, str]] = None,
    comm_short_alert_threshold: float = -38e9,
    nonrep_long_alert_threshold: float = 30e9,
) -> pd.DataFrame:
    """
    Feature 11.1: Aggregated Dollar-Weighted Index Commercial Position (Chapter 11, pp. 123-130).
    Aggregates net dollar positioning across equity index futures:
      Dollar Position_i,t = Net Contracts_i,t * Multiplier_i * Price_i,t
      Aggregated Commercial Dollar = sum_i ( Commercial Dollar Position_i )
      Aggregated Non-Reportable Dollar = sum_i ( Non-Reportable Dollar Position_i )

    Historical Extreme Imbalance Threshold (Jason Goepfert & Steve Briese Benchmark):
      - When Commercial traders reach <= -$38 Billion net short across index contracts
      - AND Non-reportable small speculators reach >= +$30 Billion net long
      -> Triggers extreme historical equity imbalance alert (extreme_imbalance_alert = 1).
    """
    mults = DEFAULT_INDEX_MULTIPLIERS.copy()
    if multipliers:
        mults.update({k.upper(): v for k, v in multipliers.items()})

    all_indices = []
    for df in market_data.values():
        if isinstance(df.index, pd.DatetimeIndex):
            all_indices.append(df.index)
        elif "report_date" in df.columns:
            all_indices.append(pd.DatetimeIndex(pd.to_datetime(df["report_date"])))
    if not all_indices:
        raise ValueError("At least one DataFrame must have a DatetimeIndex or 'report_date' column")

    union_idx = all_indices[0]
    for idx in all_indices[1:]:
        union_idx = union_idx.union(idx)
    union_idx = union_idx.sort_values().drop_duplicates()

    res = pd.DataFrame(index=union_idx)
    comm_dollar_total = pd.Series(0.0, index=union_idx)
    nonrep_dollar_total = pd.Series(0.0, index=union_idx)

    for symbol, df in market_data.items():
        sym_clean = symbol.upper()
        mult = mults.get(sym_clean, 1.0)
        p_col = "Close"
        if price_cols and symbol in price_cols:
            p_col = price_cols[symbol]
        elif price_cols and sym_clean in price_cols:
            p_col = price_cols[sym_clean]

        cols = {c.lower(): c for c in df.columns}
        df_aligned = df.reindex(union_idx)

        # Commercial Net Contracts
        if "comm_long" in cols and "comm_short" in cols:
            cl = df_aligned[cols["comm_long"]]
            cs = df_aligned[cols["comm_short"]]
            comm_net = cl - cs
        elif "comm_net" in cols:
            comm_net = df_aligned[cols["comm_net"]]
        else:
            comm_net = pd.Series(0.0, index=union_idx)

        # Non-Reportable Small Spec Net Contracts
        if "nonrep_long" in cols and "nonrep_short" in cols:
            nl = df_aligned[cols["nonrep_long"]]
            ns = df_aligned[cols["nonrep_short"]]
            nonrep_net = nl - ns
        else:
            nonrep_net = pd.Series(0.0, index=union_idx)

        # Price
        if p_col in df_aligned.columns:
            price = df_aligned[p_col].ffill()
        elif p_col.lower() in cols:
            price = df_aligned[cols[p_col.lower()]].ffill()
        else:
            price = pd.Series(1.0, index=union_idx)

        mkt_comm_dollar = (comm_net * mult * price).fillna(0.0)
        mkt_nonrep_dollar = (nonrep_net * mult * price).fillna(0.0)

        res[f"{sym_clean.lower()}_comm_dollar"] = mkt_comm_dollar
        res[f"{sym_clean.lower()}_nonrep_dollar"] = mkt_nonrep_dollar

        comm_dollar_total += mkt_comm_dollar
        nonrep_dollar_total += mkt_nonrep_dollar

    res["aggregated_dollar_comm_position"] = comm_dollar_total
    res["aggregated_dollar_nonrep_position"] = nonrep_dollar_total

    # Goepfert & Briese benchmark alert
    is_comm_extreme = comm_dollar_total <= comm_short_alert_threshold
    is_nonrep_extreme = nonrep_dollar_total >= nonrep_long_alert_threshold
    res["extreme_imbalance_alert"] = (is_comm_extreme & is_nonrep_extreme).astype(int)

    return res


def synthetic_commercial_proxy(
    df: pd.DataFrame,
    lookback_bars: int = 130,  # 26 weeks of daily bars (5 * 26 = 130)
    price_col: str = "Close",
    high_col: str = "High",
    low_col: str = "Low",
    volume_col: Optional[str] = "Volume",
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    Feature 11.2: Synthetic Commercial Proxy Indicator (Chapter 11, pp. 131-134).
    Designed for individual equities (e.g. MSFT, MO, FAST, SBUX, C, INTC)
    and non-reporting markets that lack official CFTC weekly COT reports.

    Mechanism:
      Commercial hedgers signature: 'buy on scale down and sell on scale up'
      (accumulate during price weakness, distribute into price strength).
      Calculates Close Location Value (CLV) and net synthetic flow:
        - Downside days where Close finishes in upper half of daily range reflect institutional accumulation.
        - Upside days where Close finishes in lower half reflect institutional distribution.
      Flow = CLV * Volume (or 2*Close - High - Low when Volume is absent).
      Accumulated synthetic flow is normalized using 26-week (130-bar) Stochastic %K.

    Thresholds:
      - Buy Setup Zone: Synthetic Proxy Index >= 80% (Heavy smart money scale-down accumulation)
      - Sell Setup Zone: Synthetic Proxy Index <= 20% (Heavy smart money scale-up distribution)
    """
    close = df[price_col] if price_col in df.columns else df[price_col.lower()]
    high = df[high_col] if high_col in df.columns else df[high_col.lower()]
    low = df[low_col] if low_col in df.columns else df[low_col.lower()]

    rng = high - low
    # Close Location Value in [-1, +1]
    denom = rng.where(rng > 1e-9, np.nan)
    clv = ((close - low) - (high - close)) / denom
    clv = clv.fillna(0.0)

    has_vol = False
    if volume_col and volume_col in df.columns:
        vol = df[volume_col]
        has_vol = vol.notna().any() and (vol > 0).any()
    elif volume_col and volume_col.lower() in df.columns:
        vol = df[volume_col.lower()]
        has_vol = vol.notna().any() and (vol > 0).any()

    if has_vol:
        flow = clv * vol.fillna(0.0)
    else:
        flow = (2.0 * close - high - low)

    cum_flow = flow.cumsum()
    proxy = _stochastic_normalize(cum_flow, window=lookback_bars, min_periods=min_periods)
    proxy.name = f"synthetic_commercial_proxy_{lookback_bars}"

    buy_setup = (proxy >= 80.0) & proxy.notna()
    sell_setup = (proxy <= 20.0) & proxy.notna()

    return pd.DataFrame({
        "synthetic_proxy_index": proxy,
        "synthetic_proxy_buy_setup": buy_setup.astype(int),
        "synthetic_proxy_sell_setup": sell_setup.astype(int),
        "accumulated_synthetic_flow": cum_flow,
    }, index=df.index)

