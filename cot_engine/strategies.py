"""
cot_engine/strategies.py — Systematic Rule-Based Trading Models & Regimes
from Larry Williams' 'Trade Stocks and Commodities With the Insiders' (Wiley, 2005).

Models & Systems:
  - Feature 10.4: Unified Multi-Condition Trading System (UnifiedCOTStrategy)
  - Feature 10.5: Consecutive Down Closes Mean-Reversion Pattern (ConsecutiveDownClosesStrategy, consecutive_down_closes_pattern)
  - Feature 10.6: 6-Month COT Commercial Index Directional Filter (DirectionalRegimeFilter)
  - Feature 13.1: The 60-Second Weekly Screening Strategy (OneMinuteWeeklyScan)
  - Feature 13.2: Gold / U.S. Dollar Spread Valuation Index (intermarket_gold_dollar_valuation, gold_usd_valuation_signals)
  - Feature 13.3: Gold Seasonal Cycle (gold_seasonal_cycle)
  - Feature 14.1 - 14.3: Classical Chart Patterns & 1-2-3 Reversals (detect_123_reversal, classical_chart_breakouts)
  - Feature 15.1: Tripartite Execution Architecture (TripartiteMacroModel)
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .indicators import (
    williams_cot_index,
    williams_willco,
    open_interest_stochastic,
    progo_oscillator,
)


class OneMinuteWeeklyScan:
    """
    Feature 13.1: 'The One-Minute Commodity Trader' (Chapter 13, Table 13.1).
    Rapid 60-second weekly screening process per chart over weekend; execution Monday open.
    Rules:
      1. Trend Filter: 52-week SMA of weekly closes.
         - Uptrend: SMA_52[t] > SMA_52[t-lag]
         - Downtrend: SMA_52[t] < SMA_52[t-lag]
      2. COT Setup Filter: 26-week COT Index.
         - Long Setup: Uptrend Active AND COT Index >= 80% (Long calls recommended)
         - Short Setup: Downtrend Active AND COT Index <= 20% (Long puts recommended)
      3. Asymmetric Filter: Strictly discard all counter-trend setups.
    """
    def __init__(
        self,
        trend_window: int = 52,
        cot_window: int = 26,
        trend_lag: int = 2,
        bull_threshold: float = 80.0,
        bear_threshold: float = 20.0,
        min_periods: Optional[int] = None,
    ):
        self.trend_window = trend_window
        self.cot_window = cot_window
        self.trend_lag = trend_lag
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
        self.min_periods = min_periods if min_periods is not None else (trend_window // 2)

    def generate_signals(
        self,
        df: pd.DataFrame,
        price_col: str = "Close",
    ) -> pd.DataFrame:
        out = df.copy()

        # Compute 52-week trend
        sma_52 = out[price_col].rolling(window=self.trend_window, min_periods=self.min_periods).mean()
        trend_up = sma_52 > sma_52.shift(self.trend_lag)
        trend_down = sma_52 < sma_52.shift(self.trend_lag)

        # Compute COT Index if not already present
        cot_col = f"cot_index_{self.cot_window}w"
        if cot_col not in out.columns:
            out[cot_col] = williams_cot_index(out, lookback_weeks=self.cot_window)

        cot_series = out[cot_col]

        # Chapter 13 rules:
        # Long Setup: 52-week SMA rising AND 26-week COT Index >= 80
        # Short Setup: 52-week SMA falling AND 26-week COT Index <= 20
        # Discard counter-trend signals strictly
        long_setup = trend_up & (cot_series >= self.bull_threshold)
        short_setup = trend_down & (cot_series <= self.bear_threshold)

        signals = pd.Series(0, index=out.index, name="signal_one_minute_scan")
        signals[long_setup] = 1
        signals[short_setup] = -1

        # Entry triggers: state transition (first bar of setup)
        prev_sig = signals.shift(1).fillna(0)
        triggers = pd.Series(0, index=out.index, name="trigger_one_minute_scan")
        triggers[(signals == 1) & (prev_sig != 1)] = 1
        triggers[(signals == -1) & (prev_sig != -1)] = -1

        out["sma_52w"] = sma_52
        out["trend_up"] = trend_up.astype(int)
        out["trend_down"] = trend_down.astype(int)
        out["one_minute_signal"] = signals
        out["one_minute_trigger"] = triggers

        return out


class UnifiedCOTStrategy:
    """
    Feature 10.4: Unified Multi-Condition Trading System (Williams Master Setup, Chapter 10).
    Systematic Multi-Timeframe Entry Architecture:
      1. Condition 1 (Macro COT Setup): WILLCO 26w >= 75-80% (Long) or <= 20-25% (Short).
      2. Condition 2 (Seasonal Alignment): Optional seasonal window alignment.
      3. Condition 3 (Smart Money Flow Confirmation): Daily ProGo bullish divergence (Long) or bearish divergence (Short).
      4. Condition 4 (Execution Trigger): Price trendline violation / swing breakout (breaks N-bar swing high/low).
    """
    def __init__(
        self,
        willco_window: int = 26,
        willco_bull: float = 75.0,
        willco_bear: float = 25.0,
        trigger_lookback: int = 3,
        willco_variant: str = "net",
        use_seasonal_filter: bool = False,
    ):
        self.willco_window = willco_window
        self.willco_bull = willco_bull
        self.willco_bear = willco_bear
        self.trigger_lookback = trigger_lookback
        self.willco_variant = willco_variant
        self.use_seasonal_filter = use_seasonal_filter

    def generate_signals(
        self,
        df: pd.DataFrame,
        seasonal_long_mask: Optional[pd.Series] = None,
        seasonal_short_mask: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        out = df.copy()

        # WILLCO
        willco_col = f"willco_{self.willco_window}w_{self.willco_variant}"
        if willco_col not in out.columns:
            out[willco_col] = williams_willco(
                out,
                lookback_weeks=self.willco_window,
                variant=self.willco_variant,
            )

        # ProGo
        if "progo_bull_divergence" not in out.columns or "progo_bear_divergence" not in out.columns:
            progo_df = progo_oscillator(out, window=14)
            for c in progo_df.columns:
                out[c] = progo_df[c]

        high = out["High"] if "High" in out.columns else out["high"]
        low = out["Low"] if "Low" in out.columns else out["low"]
        close = out["Close"] if "Close" in out.columns else out["close"]

        # Swing triggers (Trendline / Swing High/Low Breakout)
        swing_high = high.shift(1).rolling(self.trigger_lookback).max()
        swing_low = low.shift(1).rolling(self.trigger_lookback).min()

        willco = out[willco_col]
        setup_long = willco >= self.willco_bull
        setup_short = willco <= self.willco_bear

        # Condition 2: Seasonal alignment
        if self.use_seasonal_filter:
            if seasonal_long_mask is not None:
                setup_long = setup_long & seasonal_long_mask
            if seasonal_short_mask is not None:
                setup_short = setup_short & seasonal_short_mask

        # Condition 3: ProGo Divergence confirmation active if fired in past 10 bars
        confirm_long = out["progo_bull_divergence"].rolling(10).max() > 0
        confirm_short = out["progo_bear_divergence"].rolling(10).max() > 0

        # Condition 4: Execution trigger
        trigger_long = close > swing_high
        trigger_short = close < swing_low

        signal = pd.Series(0, index=out.index, name="unified_cot_signal")
        signal[setup_long & confirm_long & trigger_long] = 1
        signal[setup_short & confirm_short & trigger_short] = -1

        out["unified_cot_signal"] = signal
        return out


def consecutive_down_closes_pattern(
    df: pd.DataFrame,
    min_consecutive: int = 3,
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    Feature 10.5: Consecutive Down Closes Mean-Reversion Pattern (Chapter 10, Table 10.2).
    Larry Williams empirical finding (S&P 500 Daily Bars 1982-2004):
      - N=1, N=2: Negative expectancy.
      - N>=3 (3, 4, 5): Statistically significant positive edge.
    Execution:
      - Trigger at Close of day t when consecutive down closes >= min_consecutive.
      - Buy at Market Open of day t+1; exit at Market Close of day t+1 (1-day hold).
    """
    close = df[price_col] if price_col in df.columns else df[price_col.lower()]

    # Count consecutive daily declines: Close_t < Close_{t-1}
    is_down = (close < close.shift(1)).fillna(False)

    streak = pd.Series(0, index=df.index, dtype=int, name="consecutive_down_count")
    cur_streak = 0
    for i in range(1, len(df)):
        if is_down.iloc[i]:
            cur_streak += 1
        else:
            cur_streak = 0
        streak.iloc[i] = cur_streak

    buy_trigger = (streak >= min_consecutive).astype(int).rename("consecutive_down_trigger")
    # Position active on next bar
    in_position = buy_trigger.shift(1).fillna(0).astype(int).rename("consecutive_down_position")

    return pd.DataFrame({
        "consecutive_down_count": streak,
        "consecutive_down_trigger": buy_trigger,
        "consecutive_down_position": in_position,
    }, index=df.index)


class ConsecutiveDownClosesStrategy:
    """
    Feature 10.5: Class-based Consecutive Down Closes Mean-Reversion Strategy.
    """
    def __init__(self, min_consecutive: int = 3):
        self.min_consecutive = min_consecutive

    def generate_signals(
        self,
        df: pd.DataFrame,
        price_col: str = "Close",
    ) -> pd.DataFrame:
        out = df.copy()
        res = consecutive_down_closes_pattern(out, min_consecutive=self.min_consecutive, price_col=price_col)
        for col in res.columns:
            out[col] = res[col]
        return out


class DirectionalRegimeFilter:
    """
    Feature 10.6: 6-Month COT Commercial Index as a Short-Term Trading Filter (Chapter 10, pp. 120-121).
    Larry Williams: 'Never, ever would I take a long trade when the 6-month
    commercial index was below 50... It is an infallible directional filter.'
    Rules:
      - Long trading systems permitted ONLY when COT Index 26w > 50%.
      - Short trading systems permitted ONLY when COT Index 26w < 50%.
    """
    def __init__(self, cot_window: int = 26):
        self.cot_window = cot_window

    def get_regime(self, df: pd.DataFrame) -> pd.Series:
        cot_col = f"cot_index_{self.cot_window}w"
        if cot_col not in df.columns:
            cot_s = williams_cot_index(df, lookback_weeks=self.cot_window)
        else:
            cot_s = df[cot_col]

        regime = pd.Series(0, index=df.index, name="cot_directional_regime")
        regime[cot_s > 50.0] = 1   # Long-only regime
        regime[cot_s < 50.0] = -1  # Short-only regime
        return regime

    def is_long_permitted(self, cot_value: float) -> bool:
        """Returns True if COT Index > 50%."""
        return cot_value > 50.0

    def is_short_permitted(self, cot_value: float) -> bool:
        """Returns True if COT Index < 50%."""
        return cot_value < 50.0

    def filter_signals(
        self,
        df: pd.DataFrame,
        raw_signals: pd.Series,
    ) -> pd.Series:
        """
        Discards long signals (+1) when COT Index < 50, and short signals (-1) when COT Index > 50.
        """
        regime = self.get_regime(df)
        filtered = raw_signals.copy()
        # Suppress long signal if regime is -1 (short only)
        filtered[(raw_signals > 0) & (regime < 0)] = 0
        # Suppress short signal if regime is 1 (long only)
        filtered[(raw_signals < 0) & (regime > 0)] = 0
        return filtered.rename("filtered_signals")


def intermarket_gold_dollar_valuation(
    gold_series: pd.Series,
    usd_series: pd.Series,
    short_window: int = 3,
    long_window: int = 21,
) -> pd.Series:
    """
    Feature 13.2: Gold / U.S. Dollar Spread Valuation Index (Chapter 13, pp. 154-155).
    Formula:
      Spread = Gold_Close - USD_Close
      Spread Valuation Ratio = (SMA_3(Spread) - SMA_21(Spread)) / 100
    Rule:
      Ratio > 0.30 indicates extreme overvaluation of Gold relative to the Dollar,
      forecasting a major 6-month+ cyclical decline in Gold.
    """
    spread = gold_series - usd_series
    sma_short = spread.rolling(short_window).mean()
    sma_long = spread.rolling(long_window).mean()
    ratio = (sma_short - sma_long) / 100.0
    ratio.name = "gold_usd_spread_ratio"
    return ratio


def gold_usd_valuation_signals(
    gold_series: pd.Series,
    usd_series: pd.Series,
    short_window: int = 3,
    long_window: int = 21,
    overbought_threshold: float = 0.30,
    undervalued_threshold: float = -0.30,
) -> pd.DataFrame:
    """
    Feature 13.2: Full signal generator for Gold/USD Spread Valuation Ratio.
    Returns:
      - gold_usd_spread_ratio
      - gold_overbought_short_alert: 1 when ratio > 0.30 (extreme overvaluation -> professional sell)
      - gold_undervalued_long_alert: 1 when ratio < -0.30 (undervalued -> base for rally)
    """
    ratio = intermarket_gold_dollar_valuation(
        gold_series,
        usd_series,
        short_window=short_window,
        long_window=long_window,
    )
    overbought = (ratio > overbought_threshold) & ratio.notna()
    undervalued = (ratio < undervalued_threshold) & ratio.notna()

    return pd.DataFrame({
        "gold_usd_spread_ratio": ratio,
        "gold_overbought_short_alert": overbought.astype(int),
        "gold_undervalued_long_alert": undervalued.astype(int),
    }, index=gold_series.index)


def gold_seasonal_cycle(
    dates: Union[pd.DataFrame, pd.DatetimeIndex, pd.Series],
    date_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Feature 13.3: Gold Seasonal Cycle (Chapter 13, p. 154).
    Rules:
      - Sowing Window (Buy): Seasonal accumulation launches in mid-year (July).
      - Reaping Window (Sell): Seasonal distribution/peaks occur at year-end (December)
        into early Q1 (January-February).
    """
    if isinstance(dates, pd.DataFrame):
        if date_col and date_col in dates.columns:
            dt_idx = pd.DatetimeIndex(pd.to_datetime(dates[date_col]))
        elif isinstance(dates.index, pd.DatetimeIndex):
            dt_idx = dates.index
        elif "report_date" in dates.columns:
            dt_idx = pd.DatetimeIndex(pd.to_datetime(dates["report_date"]))
        else:
            raise KeyError("DataFrame must have DatetimeIndex or 'report_date' column")
        orig_index = dates.index
    elif isinstance(dates, pd.DatetimeIndex):
        dt_idx = dates
        orig_index = dates
    elif isinstance(dates, pd.Series):
        dt_idx = pd.DatetimeIndex(pd.to_datetime(dates))
        orig_index = dates.index
    else:
        raise TypeError("Input must be DataFrame, DatetimeIndex, or Series of dates")

    months = dt_idx.month

    # July is Month 7
    sowing = (months == 7)
    # Reaping is Dec (12), Jan (1), Feb (2)
    reaping = np.isin(months, [12, 1, 2])

    seasonal_bias = pd.Series(0, index=orig_index, dtype=int, name="gold_seasonal_bias")
    seasonal_bias[sowing] = 1
    seasonal_bias[reaping] = -1

    return pd.DataFrame({
        "month": months,
        "gold_sowing_window": sowing.astype(int),
        "gold_reaping_window": reaping.astype(int),
        "gold_seasonal_bias": seasonal_bias,
    }, index=orig_index)


def detect_123_reversal(
    df: pd.DataFrame,
    lookback: int = 10,
    min_swing_bars: int = 2,
) -> pd.DataFrame:
    """
    Feature 14.3: 1-2-3 (A-B-C) Reversal Pattern Triggers (Chapter 14, Figure 14.11).
    - 1-2-3 Top:
        Point 1: Major swing high.
        Point 2: Pullback swing low.
        Point 3: Lower high (fails to exceed Point 1).
        Execution: Sell short on price breaking below Point 2 low.
    - 1-2-3 Bottom:
        Point 1: Major swing low.
        Point 2: Rebound swing high.
        Point 3: Higher low (fails to break below Point 1).
        Execution: Buy on price breaking above Point 2 high.
    """
    high = df["High"] if "High" in df.columns else df["high"]
    low = df["Low"] if "Low" in df.columns else df["low"]
    close = df["Close"] if "Close" in df.columns else df["close"]

    # Recent local swing extremes excluding current bar
    swing_high = high.shift(min_swing_bars).rolling(window=lookback).max()
    swing_low = low.shift(min_swing_bars).rolling(window=lookback).min()

    # 1-2-3 Top breakdown: close breaks below recent swing low after higher highs
    break_below_pt2 = (close < swing_low) & (close.shift(1) >= swing_low.shift(1))
    # 1-2-3 Bottom breakout: close breaks above recent swing high after lower lows
    break_above_pt2 = (close > swing_high) & (close.shift(1) <= swing_high.shift(1))

    return pd.DataFrame({
        "reversal_123_top_trigger": break_below_pt2.astype(int),
        "reversal_123_bottom_trigger": break_above_pt2.astype(int),
        "swing_pt2_high": swing_high,
        "swing_pt2_low": swing_low,
    }, index=df.index)


def classical_chart_breakouts(
    df: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    Feature 14.3: Classical Chart Patterns Triggers (Trendlines, Channels, Triangles).
    Computes rolling resistance and support boundaries to generate directional triggers.
    """
    high = df["High"] if "High" in df.columns else df["high"]
    low = df["Low"] if "Low" in df.columns else df["low"]
    close = df["Close"] if "Close" in df.columns else df["close"]

    # Resistance (upper channel/trendline) and Support (lower channel/trendline)
    upper_bound = high.shift(1).rolling(window=lookback).max()
    lower_bound = low.shift(1).rolling(window=lookback).min()

    break_up = (close > upper_bound) & (close.shift(1) <= upper_bound.shift(1))
    break_down = (close < lower_bound) & (close.shift(1) >= lower_bound.shift(1))

    return pd.DataFrame({
        "channel_upper_bound": upper_bound,
        "channel_lower_bound": lower_bound,
        "pattern_breakout_long": break_up.astype(int),
        "pattern_breakout_short": break_down.astype(int),
    }, index=df.index)


class TripartiteMacroModel:
    """
    Feature 15.1: Tripartite Execution Architecture (Williams Unified Model, Chapter 15).
    Three-Tier Systematic Hierarchy:
      Tier 1 (Liquidity Exhaustion): 12-month Open Interest Stochastic low (<= 20-25) for buy,
                                     high (>= 75-80) for sell.
      Tier 2 (Commercial Positioning): 26-week COT Index OR WILLCO >= 75-80% (buy) or <= 20-25% (sell).
      Tier 3 (Long-Term Trend Filter): 39-week Moving Average (SMA_39w).
        - Buy permitted if: Price > SMA_39w OR SMA_39w slope > 0.
        - Sell permitted if: Price < SMA_39w OR SMA_39w slope < 0.
    """
    def __init__(
        self,
        oi_window: int = 52,
        cot_window: int = 26,
        trend_window: int = 39,
        oi_bull: float = 25.0,
        oi_bear: float = 75.0,
        cot_bull: float = 75.0,
        cot_bear: float = 25.0,
        cot_metric: str = "cot_index",  # "cot_index" or "willco"
        min_periods: Optional[int] = None,
    ):
        self.oi_window = oi_window
        self.cot_window = cot_window
        self.trend_window = trend_window
        self.oi_bull = oi_bull
        self.oi_bear = oi_bear
        self.cot_bull = cot_bull
        self.cot_bear = cot_bear
        self.cot_metric = cot_metric.lower()
        self.min_periods = min_periods if min_periods is not None else (trend_window // 2)

    def generate_signals(
        self,
        df: pd.DataFrame,
        price_col: str = "Close",
    ) -> pd.DataFrame:
        out = df.copy()

        # Tier 1: OI Stoch
        oi_col = f"oi_stoch_{self.oi_window}w"
        if oi_col not in out.columns:
            out[oi_col] = open_interest_stochastic(out, lookback_weeks=self.oi_window)

        # Tier 2: COT Index or WILLCO
        if self.cot_metric == "willco":
            cot_col = f"willco_{self.cot_window}w_net"
            if cot_col not in out.columns:
                out[cot_col] = williams_willco(out, lookback_weeks=self.cot_window, variant="net")
        else:
            cot_col = f"cot_index_{self.cot_window}w"
            if cot_col not in out.columns:
                out[cot_col] = williams_cot_index(out, lookback_weeks=self.cot_window)

        # Tier 3: 39-Week SMA
        sma_39 = out[price_col].rolling(self.trend_window, min_periods=self.min_periods).mean()
        slope_39 = sma_39.diff(1)

        tier1_buy = out[oi_col] <= self.oi_bull
        tier1_sell = out[oi_col] >= self.oi_bear

        tier2_buy = out[cot_col] >= self.cot_bull
        tier2_sell = out[cot_col] <= self.cot_bear

        tier3_buy = (out[price_col] > sma_39) | (slope_39 > 0)
        tier3_sell = (out[price_col] < sma_39) | (slope_39 < 0)

        signals = pd.Series(0, index=out.index, name="tripartite_signal")
        signals[tier1_buy & tier2_buy & tier3_buy] = 1
        signals[tier1_sell & tier2_sell & tier3_sell] = -1

        # Entry triggers: state transition (first bar of setup)
        prev_sig = signals.shift(1).fillna(0)
        triggers = pd.Series(0, index=out.index, name="tripartite_trigger")
        triggers[(signals == 1) & (prev_sig != 1)] = 1
        triggers[(signals == -1) & (prev_sig != -1)] = -1

        out["sma_39w"] = sma_39
        out["tripartite_signal"] = signals
        out["tripartite_trigger"] = triggers
        return out
