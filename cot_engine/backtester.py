"""
cot_engine/backtester.py — Event-Driven & Vectorized Backtesting Engine
Tailored for Larry Williams Commitments of Traders (COT) Systems.

Features:
  - Weekly scan with Monday open execution.
  - Integration with 17-Day Trailing Stops (excluding inside bars).
  - Fixed Dollar Risk Stops (Rule R1).
  - 10-15 Week Time Exits (Rule R4).
  - Comprehensive performance metrics (Win Rate, Profit Factor, Sharpe, Max Drawdown).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .risk_rules import DollarRiskStop, seventeen_day_trailing_stop, is_inside_day


@dataclass
class TradeRecord:
    entry_idx: int
    entry_date: Union[str, pd.Timestamp]
    entry_price: float
    side: str
    exit_idx: Optional[int] = None
    exit_date: Optional[Union[str, pd.Timestamp]] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_points: float = 0.0
    pnl_dollars: float = 0.0
    pnl_pct: float = 0.0
    bars_held: int = 0


class COTBacktester:
    """
    Backtesting engine for Larry Williams COT Strategies.
    Supports fixed dollar risk stops, 17-day trailing stops, and time exits.
    """
    def __init__(
        self,
        point_value: float = 1.0,         # $ per point (e.g. 100 for Gold, 50 for ES)
        max_dollar_risk: float = 3500.0,   # Williams typical stop size per contract
        use_trailing_stop: bool = True,
        trailing_stop_count: int = 17,
        max_holding_bars: int = 65,        # ~13 weeks
        commission_per_trade: float = 10.0,
        slippage_points: float = 0.0,
        entry_mode: str = "regime",        # "regime" (any active signal when flat) or "trigger" (signal transitions)
    ):
        self.point_value = point_value
        self.max_dollar_risk = max_dollar_risk
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_count = trailing_stop_count
        self.max_holding_bars = max_holding_bars
        self.commission = commission_per_trade
        self.slippage_points = slippage_points
        self.entry_mode = entry_mode

    def run(
        self,
        df: pd.DataFrame,
        signal_col: str,
        open_col: str = "Open",
        high_col: str = "High",
        low_col: str = "Low",
        close_col: str = "Close",
    ) -> Dict[str, Union[float, int, List[TradeRecord], pd.DataFrame]]:
        """
        Executes backtest over dataframe with precomputed signal_col (1 = Buy, -1 = Sell, 0 = Hold).
        Signals execute on NEXT bar's Open (standard Monday Open execution).
        """
        if df is None or len(df) == 0:
            return {"error": "Empty DataFrame"}

        data = df.copy()
        opens = data[open_col].values
        highs = data[high_col].values
        lows = data[low_col].values
        closes = data[close_col].values
        signals = data[signal_col].values
        dates = data.index if isinstance(data.index, pd.DatetimeIndex) else data.get("report_date", range(len(data)))

        # Precompute 17-day trailing stops if enabled
        if self.use_trailing_stop:
            trailing_long = seventeen_day_trailing_stop(data, side="LONG", target_count=self.trailing_stop_count).values
            trailing_short = seventeen_day_trailing_stop(data, side="SHORT", target_count=self.trailing_stop_count).values
        else:
            trailing_long = np.zeros(len(data))
            trailing_short = np.zeros(len(data))

        trades: List[TradeRecord] = []
        active_trade: Optional[TradeRecord] = None
        equity_curve = [0.0]
        initial_stop_price = 0.0

        n = len(data)
        for i in range(1, n):
            trade_was_active_at_open = (active_trade is not None)
            exit_price = None
            exit_reason = None

            # Check active trade exits first
            if active_trade is not None:
                side = active_trade.side
                bars_held = i - active_trade.entry_idx
                cur_high = highs[i]
                cur_low = lows[i]
                cur_open = opens[i]

                # 1. Check Fixed Dollar Stop Loss (Intrabar breach)
                if side == "LONG" and cur_low <= initial_stop_price:
                    exit_price = min(cur_open, initial_stop_price) - self.slippage_points
                    exit_reason = "DOLLAR_STOP"
                elif side == "SHORT" and cur_high >= initial_stop_price:
                    exit_price = max(cur_open, initial_stop_price) + self.slippage_points
                    exit_reason = "DOLLAR_STOP"

                # 2. Check 17-Day Trailing Stop (Intrabar breach, using point-in-time trailing_stop[i])
                elif self.use_trailing_stop and (
                    (side == "LONG" and cur_low <= trailing_long[i]) or
                    (side == "SHORT" and cur_high >= trailing_short[i])
                ):
                    if side == "LONG":
                        exit_price = min(cur_open, trailing_long[i]) - self.slippage_points
                    else:
                        exit_price = max(cur_open, trailing_short[i]) + self.slippage_points
                    exit_reason = "TRAILING_17D_STOP"

                # 3. Check Time Exit (10-15 weeks / max_holding_bars)
                elif bars_held >= self.max_holding_bars:
                    exit_price = cur_open - (self.slippage_points if side == "LONG" else -self.slippage_points)
                    exit_reason = "TIME_EXIT_15W"

                # 4. Check Reverse Signal Exit
                elif (side == "LONG" and signals[i - 1] == -1) or (side == "SHORT" and signals[i - 1] == 1):
                    exit_price = cur_open - (self.slippage_points if side == "LONG" else -self.slippage_points)
                    exit_reason = "REVERSE_SIGNAL"

                # 5. Final bar closeout
                elif i == n - 1:
                    exit_price = closes[i]
                    exit_reason = "END_OF_DATA"

                # Process Exit
                if exit_price is not None:
                    active_trade.exit_idx = i
                    active_trade.exit_date = dates[i]
                    active_trade.exit_price = exit_price
                    active_trade.exit_reason = exit_reason
                    active_trade.bars_held = bars_held

                    if side == "LONG":
                        points = exit_price - active_trade.entry_price
                    else:
                        points = active_trade.entry_price - exit_price

                    dollars = (points * self.point_value) - (self.commission * 2)
                    pct = (points / active_trade.entry_price) * 100.0 if active_trade.entry_price > 0 else 0.0

                    active_trade.pnl_points = points
                    active_trade.pnl_dollars = dollars
                    active_trade.pnl_pct = pct

                    trades.append(active_trade)
                    equity_curve.append(equity_curve[-1] + dollars)
                    active_trade = None

            # Check new entries on NEXT open:
            # Prevents same-bar re-entry if stopped out or time-exited during bar i.
            # Only allows entry on bar i if flat at open of bar i OR if it was a position reversal.
            can_enter = (not trade_was_active_at_open) or (exit_reason == "REVERSE_SIGNAL")
            if can_enter and active_trade is None and i < n - 1:
                prev_sig = signals[i - 1]
                sig_fired = False
                if self.entry_mode == "trigger":
                    prev_prev_sig = signals[i - 2] if i >= 2 else 0
                    sig_fired = (prev_sig != 0 and prev_sig != prev_prev_sig)
                else:
                    sig_fired = (prev_sig != 0)

                if sig_fired:
                    if prev_sig == 1:
                        entry_price = opens[i] + self.slippage_points
                        active_trade = TradeRecord(
                            entry_idx=i,
                            entry_date=dates[i],
                            entry_price=entry_price,
                            side="LONG",
                        )
                        # Set Dollar Stop
                        risk_pts = self.max_dollar_risk / self.point_value
                        initial_stop_price = entry_price - risk_pts

                    elif prev_sig == -1:
                        entry_price = opens[i] - self.slippage_points
                        active_trade = TradeRecord(
                            entry_idx=i,
                            entry_date=dates[i],
                            entry_price=entry_price,
                            side="SHORT",
                        )
                        risk_pts = self.max_dollar_risk / self.point_value
                        initial_stop_price = entry_price + risk_pts

        # Compute summary metrics
        total_trades = len(trades)
        if total_trades == 0:
            return {
                "total_trades": 0,
                "net_profit_dollars": 0.0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown_dollars": 0.0,
                "trades": [],
            }

        pnls = [t.pnl_dollars for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = (len(wins) / total_trades) * 100.0
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
        net_profit = sum(pnls)

        # Drawdown calculation
        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        drawdowns = peak - eq
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Sharpe ratio approximation
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)
        sharpe = float((mean_pnl / std_pnl) * np.sqrt(52)) if std_pnl > 0 else 0.0

        return {
            "total_trades": total_trades,
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": round(win_rate, 2),
            "net_profit_dollars": round(net_profit, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_dollars": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe, 2),
            "avg_trade_dollars": round(float(mean_pnl), 2),
            "trades": trades,
        }

