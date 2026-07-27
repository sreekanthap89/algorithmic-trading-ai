"""
Feature importance analysis using multiple techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import json
import os
from datetime import datetime

from config.settings import METRICS_DIR
from utils.logging_setup import get_logger

logger = get_logger(__name__)

class FeatureImportanceAnalyzer:
    """Analyze feature importance using multiple methods."""
    
    def __init__(self, symbol: str, interval: str):
        self.symbol = symbol
        self.interval = interval
        self.importance_file = os.path.join(
            METRICS_DIR,
            f"{symbol}_{interval}_feature_importance.json"
        )
    
    @staticmethod
    def tree_importance(model_dict: Dict) -> Dict[str, float]:
        """
        Extract feature importance from tree-based models.
        
        Args:
            model_dict: Dict with trained models
        
        Returns:
            Dict with feature importance scores
        """
        importances = {}
        feature_counts = {}
        
        for model_name, model in model_dict.items():
            if hasattr(model, 'feature_importances_'):
                importances[model_name] = model.feature_importances_
                feature_counts[model_name] = len(model.feature_importances_)
        
        if not importances:
            return {}
        
        # Average importances across models
        feature_count = max(feature_counts.values())
        avg_importance = np.zeros(feature_count)
        
        for imp in importances.values():
            avg_importance[:len(imp)] += np.array(imp)
        
        avg_importance /= len(importances)
        return {f"feature_{i}": float(imp) for i, imp in enumerate(avg_importance)}
    
    @staticmethod
    def permutation_importance(X: pd.DataFrame, y: np.ndarray, model, n_repeats: int = 10) -> Dict:
        """
        Calculate permutation importance.
        Measures drop in model performance when a feature is permuted.
        """
        baseline_score = model.score(X, y)
        importances = {}
        
        for col in X.columns:
            X_permuted = X.copy()
            scores = []
            
            for _ in range(n_repeats):
                X_permuted[col] = np.random.permutation(X_permuted[col].values)
                permuted_score = model.score(X_permuted, y)
                scores.append(baseline_score - permuted_score)
            
            importances[col] = float(np.mean(scores))
            X_permuted[col] = X[col]  # Reset
        
        return importances
    
    @staticmethod
    def correlation_importance(X: pd.DataFrame, y: np.ndarray) -> Dict:
        """Calculate correlation of features with target."""
        importances = {}
        
        for col in X.columns:
            try:
                corr = np.corrcoef(X[col].fillna(X[col].mean()), y)[0, 1]
                importances[col] = float(abs(corr)) if not np.isnan(corr) else 0.0
            except:
                importances[col] = 0.0
        
        return importances
    
    def analyze_all_methods(self, 
                           X: pd.DataFrame, 
                           y: np.ndarray, 
                           model_dict: Dict,
                           meta_stacker=None) -> Dict:
        """Comprehensive feature importance analysis."""
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "symbol": self.symbol,
            "interval": self.interval,
            "methods": {}
        }
        
        # Tree-based importance
        try:
            tree_imp = self.tree_importance(model_dict)
            if tree_imp:
                analysis["methods"]["tree_importance"] = tree_imp
                logger.info(f"Computed tree importance for {len(tree_imp)} features")
        except Exception as e:
            logger.error(f"Tree importance failed: {e}")
        
        # Permutation importance (on meta-stacker if available)
        try:
            if meta_stacker:
                perm_imp = self.permutation_importance(X, y, meta_stacker, n_repeats=5)
                analysis["methods"]["permutation_importance"] = perm_imp
                logger.info(f"Computed permutation importance for {len(perm_imp)} features")
        except Exception as e:
            logger.error(f"Permutation importance failed: {e}")
        
        # Correlation importance
        try:
            corr_imp = self.correlation_importance(X, y)
            if corr_imp:
                analysis["methods"]["correlation_importance"] = corr_imp
                logger.info(f"Computed correlation importance for {len(corr_imp)} features")
        except Exception as e:
            logger.error(f"Correlation importance failed: {e}")
        
        # Save analysis
        try:
            with open(self.importance_file, 'w') as f:
                json.dump(analysis, f, indent=4)
            logger.info(f"Saved feature importance to {self.importance_file}")
        except Exception as e:
            logger.error(f"Failed to save importance analysis: {e}")
        
        return analysis
    
    def get_top_features(self, method: str = "tree_importance", top_n: int = 10) -> List[tuple]:
        """Get top N most important features."""
        if not os.path.exists(self.importance_file):
            logger.warning(f"Importance file not found: {self.importance_file}")
            return []
        
        try:
            with open(self.importance_file, 'r') as f:
                analysis = json.load(f)
            
            importances = analysis.get("methods", {}).get(method, {})
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            return sorted_features[:top_n]
        except Exception as e:
            logger.error(f"Failed to get top features: {e}")
            return []
    
    def generate_importance_report(self) -> str:
        """Generate human-readable feature importance report."""
        if not os.path.exists(self.importance_file):
            return "No feature importance data available"
        
        try:
            with open(self.importance_file, 'r') as f:
                analysis = json.load(f)
            
            report = f"\n{'='*60}\n"
            report += f"Feature Importance Report: {self.symbol} ({self.interval})\n"
            report += f"Timestamp: {analysis['timestamp']}\n"
            report += f"{'='*60}\n\n"
            
            for method, importances in analysis.get("methods", {}).items():
                report += f"\n{method.upper()}\n"
                report += f"{'-'*40}\n"
                
                sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                for i, (feature, importance) in enumerate(sorted_features[:15], 1):
                    report += f"{i:2d}. {feature:30s} {importance:8.4f}\n"
            
            report += f"\n{'='*60}\n"
            return report
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return "Error generating report"
