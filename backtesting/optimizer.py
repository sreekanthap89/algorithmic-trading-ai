"""
Parameter optimization for backtesting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable
from itertools import product
import json
import os

from backtesting.backtest_engine import Backtest
from config.settings import BACKTEST_DIR
from utils.logging_setup import get_logger

logger = get_logger(__name__)

class ParameterOptimizer:
    """Optimize trading parameters using grid search."""
    
    def __init__(self, symbol: str, interval: str):
        self.symbol = symbol
        self.interval = interval
        self.optimization_results = []
    
    def grid_search(self,
                   df: pd.DataFrame,
                   signal_col: str,
                   entry_thresholds: List[float],
                   exit_thresholds: List[float],
                   stop_loss_pcts: List[float],
                   metric: str = "sharpe_ratio") -> Dict:
        """
        Grid search over parameter combinations.
        
        Args:
            df: Historical data
            signal_col: Signal column name
            entry_thresholds: List of entry probability thresholds to test
            exit_thresholds: List of exit probability thresholds to test
            stop_loss_pcts: List of stop loss percentages to test
            metric: Metric to optimize (sharpe_ratio, total_return, etc.)
        
        Returns:
            Best parameters and results
        """
        logger.info(f"Starting grid search with {len(entry_thresholds)} * {len(exit_thresholds)} * {len(stop_loss_pcts)} combinations")
        
        best_result = None
        best_metric_value = -np.inf if metric == "sharpe_ratio" else -np.inf
        
        for entry, exit_th, stop_loss in product(entry_thresholds, exit_thresholds, stop_loss_pcts):
            if entry <= exit_th:
                continue  # Invalid combination
            
            try:
                backtest = Backtest(self.symbol, self.interval)
                results = backtest.run(
                    df,
                    signal_col=signal_col,
                    entry_threshold=entry,
                    exit_threshold=exit_th,
                    use_stops=True,
                    stop_loss_pct=stop_loss
                )
                
                # Extract metric value
                if metric in results["risk"]:
                    metric_value = results["risk"][metric]
                elif metric in results["returns"]:
                    metric_value = results["returns"][metric]
                else:
                    metric_value = results["final_capital"]
                
                # Update best if better
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_result = {
                        "params": {
                            "entry_threshold": entry,
                            "exit_threshold": exit_th,
                            "stop_loss_pct": stop_loss
                        },
                        "results": results,
                        "metric_value": metric_value
                    }
                
                self.optimization_results.append({
                    "entry": entry,
                    "exit": exit_th,
                    "stop_loss": stop_loss,
                    metric: metric_value,
                    "total_trades": results["total_trades"]
                })
                
            except Exception as e:
                logger.error(f"Error in grid search: {e}")
                continue
        
        logger.info(f"Grid search complete. Best {metric}: {best_metric_value:.4f}")
        return best_result
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get optimization results as DataFrame."""
        return pd.DataFrame(self.optimization_results)
    
    def save_results(self, filename: str = None) -> str:
        """Save optimization results."""
        if not self.optimization_results:
            return ""
        
        if filename is None:
            from datetime import datetime
            filename = f"optimization_{self.symbol}_{self.interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = os.path.join(BACKTEST_DIR, filename)
        df = self.get_results_dataframe()
        
        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Saved optimization results to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save optimization results: {e}")
            return ""
