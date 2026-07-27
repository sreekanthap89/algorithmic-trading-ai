"""
Model validation metrics and performance tracking.
Tracks prediction accuracy, model drift, and trading performance.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    confusion_matrix, classification_report
)
from typing import Dict, Tuple
from datetime import datetime
import json
import os

from config.settings import METRICS_DIR
from utils.logging_setup import get_logger

logger = get_logger(__name__)

class ModelValidator:
    """Validate model performance on test data."""
    
    @staticmethod
    def validate_model(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict:
        """
        Comprehensive model validation metrics.
        
        Args:
            y_true: True labels (0, 1)
            y_pred: Predicted labels (0, 1)
            y_pred_proba: Predicted probabilities for class 1
        
        Returns:
            Dict with all validation metrics
        """
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_pred_proba)) if len(np.unique(y_true)) > 1 else 0.5,
            "timestamp": datetime.now().isoformat()
        }
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["confusion_matrix"] = {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
        
        # Additional metrics
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        
        logger.debug(f"Validation metrics: {metrics}")
        return metrics


class PredictionTracker:
    """Track predictions over time for drift detection."""
    
    def __init__(self, symbol: str, interval: str):
        self.symbol = symbol
        self.interval = interval
        self.predictions_list = []
        self.tracking_file = os.path.join(
            METRICS_DIR, 
            f"{symbol}_{interval}_predictions.jsonl"
        )
    
    def log_prediction(self, 
                       signal: str, 
                       prob_up: float, 
                       actual_result: bool = None,
                       metadata: Dict = None):
        """Log a single prediction."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "signal": signal,
            "prob_up": float(prob_up),
            "actual_result": actual_result,
            "metadata": metadata or {}
        }
        
        self.predictions_list.append(record)
        
        # Append to file
        try:
            with open(self.tracking_file, 'a') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
    
    def calculate_drift(self, window: int = 100) -> Dict:
        """
        Calculate prediction drift metrics.
        Tracks whether model predictions are degrading over time.
        """
        if len(self.predictions_list) < window:
            return {"drift_detected": False, "samples": len(self.predictions_list)}
        
        recent = self.predictions_list[-window:]
        
        # Calculate mean probability over time windows
        mid_point = len(recent) // 2
        first_half = np.array([p["prob_up"] for p in recent[:mid_point]])
        second_half = np.array([p["prob_up"] for p in recent[mid_point:]])
        
        mean_first = np.mean(first_half)
        mean_second = np.mean(second_half)
        drift_magnitude = abs(mean_second - mean_first)
        
        drift_metrics = {
            "drift_detected": drift_magnitude > 0.1,
            "drift_magnitude": float(drift_magnitude),
            "mean_prob_first_half": float(mean_first),
            "mean_prob_second_half": float(mean_second),
            "samples_analyzed": len(recent)
        }
        
        logger.info(f"Drift analysis for {self.symbol}_{self.interval}: {drift_metrics}")
        return drift_metrics
    
    def load_historical_predictions(self) -> pd.DataFrame:
        """Load all logged predictions from file."""
        if not os.path.exists(self.tracking_file):
            return pd.DataFrame()
        
        records = []
        try:
            with open(self.tracking_file, 'r') as f:
                for line in f:
                    records.append(json.loads(line))
            return pd.DataFrame(records)
        except Exception as e:
            logger.error(f"Failed to load predictions: {e}")
            return pd.DataFrame()


class BacktestMetrics:
    """Calculate backtesting performance metrics."""
    
    @staticmethod
    def calculate_returns(trades: list) -> Dict:
        """
        Calculate returns metrics from trades.
        
        Args:
            trades: List of trade dicts with 'entry', 'exit', 'pnl' keys
        
        Returns:
            Dict with returns metrics
        """
        if not trades:
            return {
                "total_return": 0.0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0
            }
        
        pnls = [t.get("pnl", 0) for t in trades]
        total_pnl = sum(pnls)
        
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        metrics = {
            "total_return": float(total_pnl),
            "trade_count": len(trades),
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": float(len(wins) / len(trades)) if trades else 0.0,
            "avg_win": float(np.mean(wins)) if wins else 0.0,
            "avg_loss": float(np.mean(losses)) if losses else 0.0,
            "avg_trade": float(np.mean(pnls)) if pnls else 0.0,
            "profit_factor": float(sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 0.0,
            "best_trade": float(max(pnls)) if pnls else 0.0,
            "worst_trade": float(min(pnls)) if pnls else 0.0
        }
        
        return metrics
    
    @staticmethod
    def calculate_risk_metrics(equity_curve: list) -> Dict:
        """Calculate risk metrics from equity curve."""
        if len(equity_curve) < 2:
            return {"max_drawdown": 0.0, "sharpe_ratio": 0.0, "sortino_ratio": 0.0}
        
        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]
        
        # Maximum Drawdown
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax) / cummax
        max_drawdown = float(np.min(drawdown))
        
        # Sharpe Ratio (assuming 252 trading days, 0% risk-free rate)
        sharpe_ratio = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0.0
        
        # Sortino Ratio (penalize downside only)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
        sortino_ratio = float(np.mean(returns) / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0
        
        return {
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "total_return": float((equity[-1] - equity[0]) / equity[0])
        }
