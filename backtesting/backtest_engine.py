"""
Backtesting engine for strategy evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Tuple
from datetime import datetime
import json
import os
import sys

# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.settings import BACKTEST_CONFIG, BACKTEST_DIR
from utils.logging_setup import get_logger
from metrics.validation import BacktestMetrics

logger = get_logger(__name__)

class Trade:
    """Represents a single trade."""
    
    def __init__(self, entry_index: int, entry_price: float, entry_signal: str):
        self.entry_index = entry_index
        self.entry_price = entry_price
        self.entry_signal = entry_signal
        self.entry_time = None
        
        self.exit_index = None
        self.exit_price = None
        self.exit_signal = None
        self.exit_time = None
        
        self.pnl = None
        self.pnl_pct = None
        self.bars_held = None
    
    def close(self, exit_index: int, exit_price: float, exit_signal: str):
        """Close the trade."""
        self.exit_index = exit_index
        self.exit_price = exit_price
        self.exit_signal = exit_signal
        
        self.bars_held = exit_index - self.entry_index
        self.pnl = (exit_price - self.entry_price)
        self.pnl_pct = (self.pnl / self.entry_price) * 100
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "entry_index": self.entry_index,
            "exit_index": self.exit_index,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "entry_signal": self.entry_signal,
            "exit_signal": self.exit_signal,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "bars_held": self.bars_held
        }


class Backtest:
    """Backtesting engine."""
    
    def __init__(self, 
                 symbol: str,
                 interval: str,
                 initial_capital: float = None,
                 commission: float = None,
                 slippage: float = None):
        """
        Initialize backtest.
        
        Args:
            symbol: Asset symbol
            interval: Timeframe
            initial_capital: Starting capital (from config if None)
            commission: Commission per trade (from config if None)
            slippage: Slippage per trade (from config if None)
        """
        self.symbol = symbol
        self.interval = interval
        
        self.initial_capital = initial_capital or BACKTEST_CONFIG["initial_capital"]
        self.commission = commission or BACKTEST_CONFIG["commission"]
        self.slippage = slippage or BACKTEST_CONFIG["slippage"]
        
        self.df = None
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.cash = self.initial_capital
        self.position = None  # Current open trade
        self.results = None
    
    def run(self, 
            df: pd.DataFrame,
            signal_col: str,
            entry_threshold: float = 0.56,
            exit_threshold: float = 0.44,
            use_stops: bool = True,
            stop_loss_pct: float = 0.05) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            df: DataFrame with OHLC + signals
            signal_col: Column name with signal probabilities (0-1)
            entry_threshold: Probability threshold for entry (default: 0.56 = BUY)
            exit_threshold: Threshold for exit (default: 0.44 = SELL)
            use_stops: Whether to use stop losses
            stop_loss_pct: Stop loss percentage
        
        Returns:
            Backtest results dictionary
        """
        self.df = df.copy().reset_index(drop=True)
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.cash = self.initial_capital
        self.position = None
        
        logger.info(f"Starting backtest for {self.symbol} {self.interval}")
        logger.info(f"Period: {df.index[0]} to {df.index[-1]}")
        
        has_hl = ("High" in df.columns) and ("Low" in df.columns)
        has_atr = "ATR" in df.columns
        has_exh = "Exit_Exhaustion_Score" in df.columns
        peak_since_entry = 0.0
        tp1_reached = False
        active_stop_price = 0.0
        tp1_price = 0.0
        tp2_price = 0.0

        for idx in range(1, len(df)):
            price = float(df['Close'].iloc[idx])
            high_p = float(df['High'].iloc[idx]) if has_hl else price
            low_p = float(df['Low'].iloc[idx]) if has_hl else price
            atr_val = float(df['ATR'].iloc[idx]) if has_atr and pd.notna(df['ATR'].iloc[idx]) else (price * stop_loss_pct * 0.5)
            exh_val = float(df['Exit_Exhaustion_Score'].iloc[idx]) if has_exh and pd.notna(df['Exit_Exhaustion_Score'].iloc[idx]) else 0.0
            signal_prob = float(df[signal_col].iloc[idx])

            if self.position:
                peak_since_entry = max(peak_since_entry, high_p)
                if high_p >= tp1_price > 0:
                    tp1_reached = True
                    # Ratchet stop to breakeven + Chandelier trail once TP1 is reached
                    chandelier = peak_since_entry - 1.35 * max(atr_val, 1e-6)
                    active_stop_price = max(active_stop_price, self.position.entry_price * 1.001, chandelier)

                # 1. Check Full Take-Profit (TP2)
                if use_stops and tp2_price > 0 and high_p >= tp2_price:
                    blended_tp = 0.5 * tp1_price + 0.5 * tp2_price if tp1_reached else tp2_price
                    self._close_position(idx, blended_tp, "TAKE_PROFIT")

                # 2. Check Exhaustion Profit-Lock once trade is in profit
                elif self.position and tp1_reached and exh_val >= 0.45 and price > self.position.entry_price:
                    self._close_position(idx, price, "PROFIT_LOCK_EXHAUSTION")

                # 3. Check Stop-Loss or Ratcheted Trailing Stop (Intrabar Low)
                elif self.position and use_stops and low_p <= active_stop_price:
                    exit_lbl = "TRAILING_PROFIT_STOP" if tp1_reached else "STOP_LOSS"
                    self._close_position(idx, active_stop_price, exit_lbl)

            # Check exit signal (AI conviction flip)
            if self.position and signal_prob < exit_threshold:
                self._close_position(idx, price, "EXIT_SIGNAL")

            # Check entry signal
            if not self.position and signal_prob > entry_threshold:
                self._open_position(idx, price, "ENTRY_SIGNAL")
                if self.position:
                    peak_since_entry = self.position.entry_price
                    tp1_reached = False
                    risk_dist = min(self.position.entry_price * stop_loss_pct, max(1.6 * atr_val, self.position.entry_price * 0.008))
                    active_stop_price = self.position.entry_price - risk_dist
                    tp1_price = self.position.entry_price + 1.05 * risk_dist
                    tp2_price = self.position.entry_price + 1.95 * risk_dist

            # Update equity
            current_equity = self.cash
            if self.position:
                position_value = self.position.entry_price * (price / self.position.entry_price - 1)
                current_equity += position_value

            self.equity_curve.append(current_equity)
        
        # Close any open position at the end
        if self.position:
            last_price = df['Close'].iloc[-1]
            self._close_position(len(df) - 1, last_price, "END_OF_TEST")
        
        self.results = self._compile_results()
        logger.info(f"Backtest completed. Results: {self.results}")
        
        return self.results
    
    def _open_position(self, idx: int, price: float, signal: str):
        """Open a long position."""
        if self.position:
            logger.warning("Trying to open position while one is already open")
            return
        
        # Apply slippage and commission
        entry_price = price * (1 + self.slippage)
        cost = entry_price  # For simplicity, trade 1 unit
        
        if self.cash >= cost:
            self.position = Trade(idx, entry_price, signal)
            self.cash -= cost
            logger.debug(f"Opened position at {idx}: ${entry_price:.2f}")
    
    def _close_position(self, idx: int, price: float, signal: str):
        """Close the open position."""
        if not self.position:
            return
        
        # Apply slippage and commission
        exit_price = price * (1 - self.slippage)
        proceeds = exit_price
        commission_cost = proceeds * self.commission
        
        self.position.close(idx, exit_price, signal)
        
        # Update cash
        self.cash += proceeds - commission_cost
        self.trades.append(self.position)
        
        logger.debug(f"Closed position at {idx}: ${exit_price:.2f}, PnL: ${self.position.pnl:.2f}")
        self.position = None
    
    def _compile_results(self) -> Dict:
        """Compile backtest results."""
        results = {
            "symbol": self.symbol,
            "interval": self.interval,
            "backtest_date": datetime.now().isoformat(),
            "initial_capital": self.initial_capital,
            "final_capital": self.cash + (self.position.entry_price if self.position else 0),
            "total_trades": len(self.trades),
            "commission": self.commission,
            "slippage": self.slippage
        }
        
        # Add trade metrics
        trade_list = [t.to_dict() for t in self.trades]
        results["trades"] = trade_list
        
        # Add returns metrics
        returns_metrics = BacktestMetrics.calculate_returns(trade_list)
        results["returns"] = returns_metrics
        
        # Add risk metrics
        risk_metrics = BacktestMetrics.calculate_risk_metrics(self.equity_curve)
        results["risk"] = risk_metrics
        
        results["equity_curve"] = [float(e) for e in self.equity_curve]
        
        return results
    
    def save_results(self, filename: str = None) -> str:
        """Save backtest results to file."""
        if self.results is None:
            logger.warning("No backtest results to save")
            return ""
        
        if filename is None:
            safe_sym = (
                str(self.symbol)
                .replace("=", "_")
                .replace("^", "_")
                .replace("/", "_")
                .replace(":", "_")
                .replace("\\", "_")
            )
            filename = f"{safe_sym}_{self.interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(BACKTEST_DIR, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=4, default=str)
            logger.info(f"Saved backtest results to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save backtest results: {e}")
            return ""
    
    def get_summary(self) -> str:
        """Get human-readable backtest summary."""
        if self.results is None:
            return "No backtest results available"
        
        r = self.results
        ret = r["returns"]
        risk = r["risk"]
        
        summary = f"""
{'='*60}
BACKTEST SUMMARY: {r['symbol']} ({r['interval']})
{'='*60}

CAPITAL:
  Initial Capital:      ${r['initial_capital']:,.2f}
  Final Capital:        ${r['final_capital']:,.2f}
  Net Return:           {risk['total_return']*100:+.2f}%

TRADING STATISTICS:
  Total Trades:         {r['total_trades']}
  Winning Trades:       {ret['win_count']}
  Losing Trades:        {ret['loss_count']}
  Win Rate:             {ret['win_rate']*100:.2f}%
  Average Win:          ${ret['avg_win']:,.2f}
  Average Loss:         ${ret['avg_loss']:,.2f}
  Profit Factor:        {ret['profit_factor']:.2f}

RISK METRICS:
  Max Drawdown:         {risk['max_drawdown']*100:.2f}%
  Sharpe Ratio:         {risk['sharpe_ratio']:.2f}
  Sortino Ratio:        {risk['sortino_ratio']:.2f}
  Neutral Cost (%):     {ret.get('neutral_cost_pct', 0.0):.4f}% (Break-even fee hurdle)
  
Best Trade:            ${ret['best_trade']:,.2f}
Worst Trade:           ${ret['worst_trade']:,.2f}

{'='*60}
"""
        return summary


class MarketNeutralPortfolioBacktest:
    """
    Multi-Asset Market-Neutral Portfolio Backtester.
    Direct implementation of Fu et al. (MIT / Preferred Networks 2024, arXiv:2412.09880):
      1. Predicts direction across an asset universe over horizon h.
      2. Computes basic binary directional positions (+1 for up, -1 for down).
      3. Cross-sectional demeaning: pos_i = pos_i - mean(pos) across all assets in basket.
         Removes systemic market drift (beta) and isolates true alpha.
      4. Normalizes L1-norm of daily exposure to 1 / (h - 1), guaranteeing cumulative
         capital exposure across the holding period <= 1.0 (100% portfolio budget).
      5. Evaluates Realized PnL, Annualized Sharpe, Max Drawdown, and Neutral Cost (%).
    """

    def __init__(
        self,
        symbols: List[str],
        horizon: int = 128,
        cost_ratio: float = 0.0005,
        rebalance_daily: bool = True,
    ):
        self.symbols = symbols
        self.horizon = max(2, horizon)
        self.cost_ratio = cost_ratio
        self.rebalance_daily = rebalance_daily
        self.results: Dict = {}

    def run_with_predictions(
        self,
        prices_df: pd.DataFrame,
        signals_df: pd.DataFrame,
        market_neutral: bool = True,
        continuous_weighting: bool = False,
    ) -> Dict:
        """
        Runs portfolio backtest given aligned historical prices and model signal scores.

        Args:
            prices_df: DataFrame of asset close prices (Index: datetime, Columns: symbols)
            signals_df: DataFrame of model predictions (values > 0.5 or > 0 indicates predicted UP,
                        or continuous Information-Ratio alpha scores)
            market_neutral: Whether to apply cross-sectional demeaning (recommended: True)
            continuous_weighting: Whether to use continuous uncertainty-adjusted alpha weighting (TimesFM Intraday Blueprint)
                                 or classical binary +/- 1 directional allocation.

        Returns:
            Dictionary with cumulative PnL, Sharpe, Max Drawdown, and Neutral Cost (%)
        """
        aligned_idx = prices_df.index.intersection(signals_df.index)
        if len(aligned_idx) < self.horizon + 10:
            # Fallback: date-level floor alignment if timestamps had mismatched hour offsets (e.g. 04:00 futures vs 00:00 crypto)
            try:
                p_copy = prices_df.copy()
                s_copy = signals_df.copy()
                p_copy.index = pd.to_datetime(p_copy.index, utc=True).floor('D')
                s_copy.index = pd.to_datetime(s_copy.index, utc=True).floor('D')
                p_copy = p_copy[~p_copy.index.duplicated(keep='last')]
                s_copy = s_copy[~s_copy.index.duplicated(keep='last')]
                common = p_copy.index.intersection(s_copy.index)
                if len(common) >= self.horizon + 10:
                    prices_df = p_copy
                    signals_df = s_copy
                    aligned_idx = common
            except Exception:
                pass

        if len(aligned_idx) < self.horizon + 10:
            logger.warning("Insufficient aligned data for market neutral backtest.")
            return {}

        prices = prices_df.loc[aligned_idx].ffill().bfill()
        signals = signals_df.loc[aligned_idx].ffill().fillna(0.5)

        # Forward multi-horizon return: R_{t, h} = (P_{t + h - 1} - P_t) / P_t
        h_step = self.horizon - 1
        returns = (prices.shift(-h_step) - prices) / np.clip(prices, 1e-8, None)
        returns = returns.iloc[:-h_step] if h_step > 0 else returns

        # Raw position: binary or continuous Information-Ratio weighted
        if continuous_weighting:
            # Continuous Information Ratio Alpha (TimesFM Intraday Blueprint, arXiv:2412.09880 / TimesFM-3)
            base_score = 0.50 if signals.min().min() >= 0.0 else 0.0
            pos = signals.loc[returns.index] - base_score
        else:
            # Binary directional signals: +1 if signal > threshold, else -1
            threshold = 0.50 if signals.min().min() >= 0.0 else 0.0
            pos_arr = np.where(signals.loc[returns.index] > threshold, 1.0, -1.0)
            pos = pd.DataFrame(pos_arr, index=returns.index, columns=prices.columns)

        # Step 1: Cross-Sectional Mean Subtraction (Market-Neutral Demeaning)
        if market_neutral and pos.shape[1] > 1:
            pos = pos.sub(pos.mean(axis=1), axis=0)

        # Step 2: L1-Norm Sizing Budget: sum(|pos|) = 1 / (h - 1)
        l1_sum = pos.abs().sum(axis=1).replace(0.0, np.nan)
        scaling_factor = 1.0 / max(1, self.horizon - 1)
        norm_pos = pos.div(l1_sum, axis=0) * scaling_factor
        norm_pos = norm_pos.fillna(0.0)

        # Trading friction & transaction costs
        trading_cost = norm_pos.abs().sum(axis=1) * self.cost_ratio
        total_trades_volume = float(norm_pos.abs().sum().sum())

        # Realized portfolio PnL = sum(pos_i * return_i)
        asset_pnls = norm_pos * returns
        daily_pnl = asset_pnls.sum(axis=1) - trading_cost

        # Shift PnL by horizon to align with trade realization date
        daily_pnl = daily_pnl.shift(h_step).fillna(0.0)

        cum_pnl = daily_pnl.cumsum()
        total_pnl_val = float(daily_pnl.sum())

        # Neutral Cost (%) = (Total PnL / Total Traded Volume) * 100 / 2
        neutral_cost_pct = (
            (total_pnl_val / total_trades_volume * 100.0 / 2.0)
            if total_trades_volume > 1e-9
            else 0.0
        )

        # Annualized metrics (assuming 252 trading days)
        periods_per_year = 252
        n_periods = max(1, len(daily_pnl))
        ann_return = float(daily_pnl.sum() * periods_per_year / n_periods)
        daily_std = float(np.std(daily_pnl))
        ann_vol = daily_std * np.sqrt(periods_per_year)
        ann_sharpe = float(ann_return / ann_vol) if ann_vol > 1e-8 else 0.0

        # Max Drawdown
        cum_equity = 1.0 + cum_pnl
        peak = cum_equity.cummax()
        dd = (cum_equity - peak) / peak
        max_dd = float(dd.min())

        # Accuracy
        trade_accuracies = (asset_pnls > 0).sum().sum() / max(1, asset_pnls.count().sum())

        self.results = {
            "horizon": self.horizon,
            "market_neutral": market_neutral,
            "annualized_sharpe": round(ann_sharpe, 3),
            "annualized_return_pct": round(ann_return * 100.0, 2),
            "annualized_volatility_pct": round(ann_vol * 100.0, 2),
            "max_drawdown_pct": round(max_dd * 100.0, 2),
            "neutral_cost_pct": round(neutral_cost_pct, 4),
            "directional_accuracy": round(float(trade_accuracies), 4),
            "total_trades_volume": round(total_trades_volume, 2),
            "daily_pnl": daily_pnl,
            "cumulative_pnl": cum_pnl,
        }

        return self.results
