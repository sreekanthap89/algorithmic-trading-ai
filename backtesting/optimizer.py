"""
Parameter optimization for backtesting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable
from itertools import product, combinations
import json
import os

from backtesting.backtest_engine import Backtest
from config.settings import BACKTEST_DIR
from utils.logging_setup import get_logger

logger = get_logger(__name__)


class CombinatorialPurgedCV:
    """
    Module 10 - Part IV (Marcos López de Prado - Advances in Financial Machine Learning):
    Combinatorial Purged Cross-Validation (CPCV).
    Divides historical bars into N groups, generating C(N, k) combinations of test groups.
    Enforces Purging and Embargo around train/test boundaries to eliminate data leakage.
    Allows constructing multiple out-of-sample backtest paths and calculating the exact
    Probability of Backtest Overfitting (PBO).
    """

    def __init__(self, n_groups: int = 5, k_test: int = 2, embargo_pct: float = 0.01):
        self.n_groups = n_groups
        self.k_test = k_test
        self.embargo_pct = embargo_pct

    def split(self, n_samples: int):
        group_bounds = np.linspace(0, n_samples, self.n_groups + 1, dtype=int)
        groups = [list(range(group_bounds[i], group_bounds[i + 1])) for i in range(self.n_groups)]
        embargo = max(1, int(n_samples * self.embargo_pct))

        for test_combo in combinations(range(self.n_groups), self.k_test):
            test_indices = []
            for g in test_combo:
                test_indices.extend(groups[g])

            train_indices = []
            for g in range(self.n_groups):
                if g in test_combo:
                    continue
                g_indices = groups[g]
                if (g - 1) in test_combo:
                    g_indices = g_indices[embargo:]
                train_indices.extend(g_indices)

            yield np.array(train_indices, dtype=int), np.array(test_indices, dtype=int)


class ParameterOptimizer:
    """Optimize trading parameters using grid search and Combinatorial Purged Cross-Validation (CPCV)."""
    
    def __init__(self, symbol: str, interval: str):
        self.symbol = symbol
        self.interval = interval
        self.optimization_results = []
    
    def optimize_with_cpcv(
        self,
        df: pd.DataFrame,
        signal_col: str,
        entry_thresholds: List[float],
        exit_thresholds: List[float],
        stop_loss_pcts: List[float],
        n_groups: int = 5,
        k_test: int = 2,
    ) -> Dict:
        """
        Runs Combinatorial Purged Cross-Validation (CPCV) to optimize trading rules
        while computing Marcos López de Prado's Probability of Backtest Overfitting (PBO).
        """
        param_candidates = []
        for entry, exit_th, stop_loss in product(entry_thresholds, exit_thresholds, stop_loss_pcts):
            if entry > exit_th:
                param_candidates.append({
                    "entry_threshold": entry,
                    "exit_threshold": exit_th,
                    "stop_loss_pct": stop_loss
                })

        if not param_candidates:
            return {}

        cpcv = CombinatorialPurgedCV(n_groups=n_groups, k_test=k_test)
        splits = list(cpcv.split(len(df)))
        n_splits = len(splits)

        is_matrix = np.zeros((len(param_candidates), n_splits), dtype=float)
        oos_matrix = np.zeros((len(param_candidates), n_splits), dtype=float)

        backtester = Backtest(self.symbol, self.interval)

        for s_idx, (tr_idx, te_idx) in enumerate(splits):
            if len(tr_idx) < 20 or len(te_idx) < 10:
                continue
            df_tr = df.iloc[tr_idx]
            df_te = df.iloc[te_idx]

            for c_idx, params in enumerate(param_candidates):
                try:
                    res_tr = backtester.run(
                        df_tr,
                        signal_col=signal_col,
                        entry_threshold=params["entry_threshold"],
                        exit_threshold=params["exit_threshold"],
                        use_stops=True,
                        stop_loss_pct=params["stop_loss_pct"]
                    )
                    res_te = backtester.run(
                        df_te,
                        signal_col=signal_col,
                        entry_threshold=params["entry_threshold"],
                        exit_threshold=params["exit_threshold"],
                        use_stops=True,
                        stop_loss_pct=params["stop_loss_pct"]
                    )
                    is_matrix[c_idx, s_idx] = float(res_tr["risk"].get("sharpe_ratio", 0.0))
                    oos_matrix[c_idx, s_idx] = float(res_te["risk"].get("sharpe_ratio", 0.0))
                except Exception:
                    is_matrix[c_idx, s_idx] = -1.0
                    oos_matrix[c_idx, s_idx] = -1.0

        oos_ranks_of_best_is = []
        for s_idx in range(n_splits):
            best_is_idx = int(np.argmax(is_matrix[:, s_idx]))
            oos_scores = oos_matrix[:, s_idx]
            rank_oos = float(np.mean(oos_scores <= oos_scores[best_is_idx]))
            oos_ranks_of_best_is.append(rank_oos)

        pbo = float(np.mean([1.0 if r <= 0.50 else 0.0 for r in oos_ranks_of_best_is]))

        median_oos_sharpes = np.median(oos_matrix, axis=1)
        best_cand_idx = int(np.argmax(median_oos_sharpes))
        best_params = param_candidates[best_cand_idx]

        oos_dist = oos_matrix[best_cand_idx, :]
        from scipy.stats import spearmanr
        mean_is = np.mean(is_matrix, axis=1)
        mean_oos = np.mean(oos_matrix, axis=1)
        rank_ic, _ = spearmanr(mean_is, mean_oos)
        rank_ic = 0.0 if np.isnan(rank_ic) else float(rank_ic)

        overfit_badge = (
            "LOW RISK (ROBUST)" if pbo < 0.25
            else "MODERATE OVERFITTING" if pbo < 0.50
            else "HIGH OVERFITTING RISK"
        )

        return {
            "best_params": best_params,
            "pbo": round(pbo, 3),
            "overfit_risk_badge": overfit_badge,
            "is_oos_rank_correlation": round(rank_ic, 3),
            "oos_sharpe_distribution": {
                "median": round(float(np.median(oos_dist)), 3),
                "mean": round(float(np.mean(oos_dist)), 3),
                "q10": round(float(np.percentile(oos_dist, 10)), 3),
                "q90": round(float(np.percentile(oos_dist, 90)), 3),
            },
            "cpcv_combinations_tested": n_splits,
            "total_candidates": len(param_candidates),
        }
    
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
            safe_sym = (
                str(self.symbol)
                .replace("=", "_")
                .replace("^", "_")
                .replace("/", "_")
                .replace(":", "_")
                .replace("\\", "_")
            )
            filename = f"optimization_{safe_sym}_{self.interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = os.path.join(BACKTEST_DIR, filename)
        df = self.get_results_dataframe()
        
        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Saved optimization results to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save optimization results: {e}")
            return ""
