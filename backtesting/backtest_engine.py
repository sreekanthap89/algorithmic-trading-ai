"""
Backtesting engine for strategy evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Tuple
from datetime import datetime
import json
import os

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
        
        for idx in range(1, len(df)):
            price = df['Close'].iloc[idx]
            signal_prob = df[signal_col].iloc[idx]
            prev_price = df['Close'].iloc[idx - 1]
            
            # Check stop loss
            if self.position and use_stops:
                stop_price = self.position.entry_price * (1 - stop_loss_pct)
                if price < stop_price:
                    # Stop loss triggered
                    self._close_position(idx, stop_price, "STOP_LOSS")
            
            # Check exit signal
            if self.position and signal_prob < exit_threshold:
                self._close_position(idx, price, "EXIT_SIGNAL")
            
            # Check entry signal
            if not self.position and signal_prob > entry_threshold:
                self._open_position(idx, price, "ENTRY_SIGNAL")
            
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
            filename = f"{self.symbol}_{self.interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
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
  
Best Trade:            ${ret['best_trade']:,.2f}
Worst Trade:           ${ret['worst_trade']:,.2f}

{'='*60}
"""
        return summary
