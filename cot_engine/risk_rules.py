"""
cot_engine/risk_rules.py — Risk Management, Dynamic Stops & Execution Filters
from Larry Williams' 'Trade Stocks and Commodities With the Insiders' (Wiley, 2005).

Features implemented:
  - Feature 12.1: Dollar-Based Protective Stop Loss (DollarRiskStop)
  - Feature 12.2: 17-Day Trend Trailing Stop strictly excluding inside days (seventeen_day_trailing_stop)
  - Feature 12.3: 13-Day Trend Re-entry Rule (thirteen_day_reentry_levels)
  - Feature 12.4: Time-Based Trend Exhaustion (TimeExitFilter) & Parabolic Windfall Exit (ParabolicWindfallExit)
  - Feature 12.5: Anti-Plunging & Position Sizing Discipline (AntiPlungingPositionSizer)
  - Feature 14.3: Flag Consolidation Bracket Orders (FlagBracketExecution)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DollarRiskStop:
    """
    Feature 12.1: Fixed Dollar Risk Stop (Chapter 12, p. 138).
    Williams: 'Your risk is your stop... stops protect DOLLARS, not chart patterns.'
    Stop Price Long  = Entry Price - (Max Dollar Risk / Point Value)
    Stop Price Short = Entry Price + (Max Dollar Risk / Point Value)
    """
    max_dollar_risk: float
    point_value: float = 1.0  # Multiplier (e.g. 100 for Gold, 50 for ES, 5000 for Wheat)

    def calculate_stop(self, entry_price: float, side: str = "LONG") -> float:
        points = self.max_dollar_risk / self.point_value
        if side.upper() == "LONG":
            return entry_price - points
        else:
            return entry_price + points

    def check_stop(
        self,
        entry_price: float,
        bar_high: float,
        bar_low: float,
        side: str = "LONG",
    ) -> Tuple[bool, float]:
        """
        Evaluates whether the protective dollar stop was breached during the bar.
        Returns (is_stopped, exit_price).
        """
        stop_level = self.calculate_stop(entry_price, side)
        if side.upper() == "LONG":
            if bar_low <= stop_level:
                return True, stop_level
            return False, entry_price
        else:
            if bar_high >= stop_level:
                return True, stop_level
            return False, entry_price


def is_inside_day(df: pd.DataFrame) -> pd.Series:
    """
    Identifies inside bars using Larry Williams' strict definition (Chapter 10, p. 118):
    High < Prior High AND Low > Prior Low
    """
    high = df["High"] if "High" in df.columns else df["high"]
    low = df["Low"] if "Low" in df.columns else df["low"]

    prior_high = high.shift(1)
    prior_low = low.shift(1)

    inside = (high < prior_high) & (low > prior_low)
    return inside.fillna(False)


def seventeen_day_trailing_stop(
    df: pd.DataFrame,
    side: str = "LONG",
    target_count: int = 17,
    ratchet: bool = False,
) -> pd.Series:
    """
    Feature 12.2: 17-Day Major Trend Trailing Stop (Chapter 12, p. 145).
    - Long stop = Lowest Low of the last 17 trading days.
    - Short stop = Highest High of the last 17 trading days.
    - CRUCIAL MANDATE: Inside days are strictly EXCLUDED from the 17-day count.
      Only true trend-expansion/continuation days count toward the 17 days.
    - ratchet: If True, stop can only tighten in favor of position (monotonically ratcheting).
    """
    high = df["High"] if "High" in df.columns else df["high"]
    low = df["Low"] if "Low" in df.columns else df["low"]
    inside = is_inside_day(df)

    stops = pd.Series(index=df.index, dtype=float)

    # For each bar t, look back at preceding non-inside bars
    for i in range(len(df)):
        if i == 0:
            stops.iloc[i] = low.iloc[i] if side.upper() == "LONG" else high.iloc[i]
            continue

        # Look back up to i bars
        prior_non_inside_idx = []
        for j in range(i - 1, -1, -1):
            if not inside.iloc[j]:
                prior_non_inside_idx.append(j)
                if len(prior_non_inside_idx) >= target_count:
                    break

        if not prior_non_inside_idx:
            raw_stop = low.iloc[i - 1] if side.upper() == "LONG" else high.iloc[i - 1]
        else:
            if side.upper() == "LONG":
                raw_stop = low.iloc[prior_non_inside_idx].min()
            else:
                raw_stop = high.iloc[prior_non_inside_idx].max()

        if ratchet and i > 0:
            prev_stop = stops.iloc[i - 1]
            if not np.isnan(prev_stop):
                if side.upper() == "LONG":
                    stops.iloc[i] = max(prev_stop, raw_stop)
                else:
                    stops.iloc[i] = min(prev_stop, raw_stop)
            else:
                stops.iloc[i] = raw_stop
        else:
            stops.iloc[i] = raw_stop

    stops.name = f"stop_17d_{side.lower()}"
    return stops


def thirteen_day_reentry_levels(
    df: pd.DataFrame,
    lookback_bars: int = 13,
    exclude_inside_days: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    """
    Feature 12.3: 13-Day Trend Re-entry Rule (Chapter 12, p. 145).
    If stopped out of a trending position while macro COT regime is active:
    - Long Re-Entry: Breakout above highest high of the past 13 trading days (strictly excluding inside days).
    - Short Re-Entry: Breakdown below lowest low of the past 13 trading days (strictly excluding inside days).
    """
    high = df["High"] if "High" in df.columns else df["high"]
    low = df["Low"] if "Low" in df.columns else df["low"]

    if not exclude_inside_days:
        reentry_long = high.shift(1).rolling(window=lookback_bars).max()
        reentry_short = low.shift(1).rolling(window=lookback_bars).min()
    else:
        inside = is_inside_day(df)
        reentry_long = pd.Series(index=df.index, dtype=float)
        reentry_short = pd.Series(index=df.index, dtype=float)

        for i in range(len(df)):
            if i == 0:
                reentry_long.iloc[i] = np.nan
                reentry_short.iloc[i] = np.nan
                continue

            prior_non_inside_idx = []
            for j in range(i - 1, -1, -1):
                if not inside.iloc[j]:
                    prior_non_inside_idx.append(j)
                    if len(prior_non_inside_idx) >= lookback_bars:
                        break

            if not prior_non_inside_idx:
                reentry_long.iloc[i] = high.iloc[i - 1]
                reentry_short.iloc[i] = low.iloc[i - 1]
            else:
                reentry_long.iloc[i] = high.iloc[prior_non_inside_idx].max()
                reentry_short.iloc[i] = low.iloc[prior_non_inside_idx].min()

    reentry_long.name = "reentry_13d_high"
    reentry_short.name = "reentry_13d_low"
    return reentry_long, reentry_short


@dataclass
class TimeExitFilter:
    """
    Feature 12.4: Time-Based Exit Horizon (Chapter 13, p. 158).
    Williams: 'Few markets go up or down for more than 15 consecutive weeks
    without an important correction.'
    Triggers mandatory profit exit after 10 to 15 weeks (50 to 75 trading days).
    """
    min_holding_bars: int = 50   # 10 weeks of daily bars (or 10 weekly bars)
    max_holding_bars: int = 75   # 15 weeks of daily bars (or 15 weekly bars)
    timeframe: str = "daily"     # "daily" or "weekly"

    def __post_init__(self):
        if self.timeframe.lower() == "weekly":
            # If user didn't override default daily numbers, scale to weekly bars
            if self.min_holding_bars == 50 and self.max_holding_bars == 75:
                self.min_holding_bars = 10
                self.max_holding_bars = 15

    def should_exit(self, bars_in_trade: int) -> bool:
        return bars_in_trade >= self.max_holding_bars


@dataclass
class ParabolicWindfallExit:
    """
    Feature 12.4: Parabolic Windfall Profit Taking (Chapter 12, p. 158).
    Larry Williams: 'Liquidate immediately upon sudden, massive, explosive
    vertical price surges rather than waiting for trailing stops.'
    Triggers exit when bar range and directional expansion exceeds an ATR multiplier threshold.
    """
    expansion_multiplier: float = 2.5
    atr_window: int = 14

    def check_exit(
        self,
        entry_price: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        atr: float,
        side: str = "LONG",
    ) -> Tuple[bool, str]:
        """
        Returns (should_exit, reason).
        """
        bar_range = bar_high - bar_low
        is_windfall = bar_range >= (self.expansion_multiplier * atr)
        if side.upper() == "LONG":
            profit = bar_close - entry_price
            if is_windfall and profit > 0 and (bar_close - bar_low) >= 0.70 * bar_range:
                return True, "PARABOLIC_WINDFALL_SURGE_LONG"
        else:
            profit = entry_price - bar_close
            if is_windfall and profit > 0 and (bar_high - bar_close) >= 0.70 * bar_range:
                return True, "PARABOLIC_WINDFALL_SURGE_SHORT"
        return False, ""

    def detect_windfall_bars(
        self,
        df: pd.DataFrame,
        side: str = "LONG",
    ) -> pd.Series:
        """
        Vectorized detection of windfall expansion bars.
        """
        high = df["High"] if "High" in df.columns else df["high"]
        low = df["Low"] if "Low" in df.columns else df["low"]
        close = df["Close"] if "Close" in df.columns else df["close"]
        prev_close = close.shift(1).fillna(close)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_window, min_periods=1).mean()
        bar_range = high - low

        windfall = bar_range >= (self.expansion_multiplier * atr)
        if side.upper() == "LONG":
            signal = windfall & (close > prev_close) & ((close - low) >= 0.70 * bar_range)
        else:
            signal = windfall & (close < prev_close) & ((high - close) >= 0.70 * bar_range)

        return signal.astype(int).rename(f"windfall_exit_{side.lower()}")


@dataclass
class AntiPlungingPositionSizer:
    """
    Feature 12.5: Anti-Plunging & Position Sizing Discipline (Chapter 12, pp. 135-138).
    Williams: 'Plunging (overleveraging) is the primary driver of trader ruin.
    Sizing must remain strictly scaled so that stepping up contract size causes
    zero alteration in execution rules or psychological detachment.'
    Calculates position contracts strictly bounded by fixed account equity risk fraction.
    """
    account_equity: float
    risk_fraction: float = 0.01  # Default 1% equity risk per trade
    contract_point_value: float = 1.0  # Point value / contract multiplier
    max_contracts_cap: Optional[int] = None
    min_contracts: int = 1

    def calculate_max_risk_dollars(self) -> float:
        """Max allowable dollar loss on the trade."""
        return self.account_equity * self.risk_fraction

    def calculate_contracts(
        self,
        stop_distance_points: float,
        protective_stop_dollars: Optional[float] = None,
    ) -> int:
        """
        Computes safe number of contracts based on stop distance in points or fixed dollar stop.
        Formula: Contracts = floor( MaxDollarRisk / LossPerContract )
        """
        max_risk_usd = self.calculate_max_risk_dollars()
        if protective_stop_dollars is not None and protective_stop_dollars > 0:
            loss_per_contract = protective_stop_dollars
        else:
            if stop_distance_points <= 0:
                return self.min_contracts
            loss_per_contract = stop_distance_points * self.contract_point_value

        if loss_per_contract <= 0:
            return self.min_contracts

        contracts = int(max_risk_usd // loss_per_contract)
        if contracts < self.min_contracts:
            contracts = self.min_contracts
        if self.max_contracts_cap is not None:
            contracts = min(contracts, self.max_contracts_cap)
        return contracts


class FlagBracketExecution:
    """
    Feature 14.3: Flag Consolidation Bracket Orders (Chapter 14, Figure 14.11).
    Places stop buy order above flag resistance and stop sell order below flag support.
    Whichever order executes, the unfilled opposite bracket order is left active
    as the initial protective stop loss.
    """
    def __init__(self, high_trigger: float, low_trigger: float):
        self.high_trigger = high_trigger
        self.low_trigger = low_trigger

    def evaluate_bar(self, bar_high: float, bar_low: float) -> Optional[Dict[str, float]]:
        if bar_high >= self.high_trigger:
            return {
                "action": "BUY_ENTER",
                "entry_price": self.high_trigger,
                "initial_stop": self.low_trigger,  # Opposite bracket is stop
                "side": "LONG",
            }
        elif bar_low <= self.low_trigger:
            return {
                "action": "SELL_ENTER",
                "entry_price": self.low_trigger,
                "initial_stop": self.high_trigger,  # Opposite bracket is stop
                "side": "SHORT",
            }
        return None
