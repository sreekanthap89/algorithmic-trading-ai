"""
=============================================================
 ULTRA STACKING ENSEMBLE & DEEP LEARNING MODEL
 Step 8 & Step 9 Implementation (Trading App)
=============================================================
"""

import numpy as np
import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb

MODELS_DIR = "models"

class SequenceFeatureBuilder:
    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def transform(self, df: pd.DataFrame, feature_cols: list) -> tuple:
        """
        Converts 2D DataFrame features into 3D sequence matrices:
        Output shape: (num_samples, lookback, num_features)
        """
        valid_df = df.dropna(subset=feature_cols + ['Target']).copy()
        if len(valid_df) < self.lookback + 5:
            return np.array([]), np.array([])

        data_matrix = valid_df[feature_cols].values
        targets = valid_df['Target'].values

        X_seq = []
        y_seq = []

        for i in range(self.lookback, len(data_matrix)):
            X_seq.append(data_matrix[i - self.lookback:i])
            y_seq.append(targets[i])

        return np.array(X_seq), np.array(y_seq)

class UltraStackingEnsemble:
    def __init__(self, symbol: str, interval: str):
        self.symbol = symbol
        self.interval = interval
        self.safe_symbol = symbol.replace("=", "_").replace("^", "_")
        self.model_path = os.path.join(MODELS_DIR, f"{self.safe_symbol}_{self.interval}_stacking.pkl")
        
        self.feature_cols = [
            'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'BB_Mid', 
            'ATR', 'SMA_20', 'SMA_50', 'Return', 'VWAP_Dev', 'RSI_Divergence'
        ]
        
        self.base_models = {}
        self.meta_learner = None

        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)

    def train_or_load(self, df: pd.DataFrame, force_retrain: bool = False):
        """
        Trains Level-0 Base Learners (XGBoost, LightGBM, RF, ET, MLP) and
        Level-1 Meta-Learner (Logistic Regression) via 5-fold TimeSeriesSplit.
        """
        available_cols = [c for c in self.feature_cols if c in df.columns]
        valid_df = df.dropna(subset=available_cols + ['Target']).copy()

        if not force_retrain and os.path.exists(self.model_path):
            try:
                saved_dict = joblib.load(self.model_path)
                self.base_models = saved_dict.get("base_models", {})
                self.meta_learner = saved_dict.get("meta_learner", None)
                print(f"[{self.symbol}] Loaded Stacking Ensemble successfully.")
                return
            except Exception:
                print(f"[{self.symbol}] Retraining Stacking Ensemble...")

        X = valid_df[available_cols].values
        y = valid_df['Target'].values

        if len(X) < 40:
            print(f"[{self.symbol}] Insufficient data for Stacking Ensemble training.")
            return

        # Initialize Base Learners
        self.base_models = {
            "xgb": xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42, eval_metric='logloss'),
            "lgb": lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42, verbose=-1),
            "rf": RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42),
            "et": ExtraTreesClassifier(n_estimators=50, max_depth=4, random_state=42),
            "mlp": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=150, random_state=42, early_stopping=True)
        }

        # Generate Out-Of-Fold (OOF) predictions via TimeSeriesSplit
        ts_split = TimeSeriesSplit(n_splits=5)
        oof_preds = np.zeros((len(X), len(self.base_models)))

        model_names = list(self.base_models.keys())

        for train_idx, val_idx in ts_split.split(X):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va = X[val_idx]

            for idx, name in enumerate(model_names):
                m = self.base_models[name]
                try:
                    m.fit(X_tr, y_tr)
                    oof_preds[val_idx, idx] = m.predict_proba(X_va)[:, 1]
                except Exception:
                    oof_preds[val_idx, idx] = 0.5

        # Fit all base models on full data
        for name, m in self.base_models.items():
            try:
                m.fit(X, y)
            except Exception:
                pass

        # Train Level-1 Meta-Learner on OOF predictions
        oof_valid_mask = ~np.all(oof_preds == 0, axis=1)
        if oof_valid_mask.sum() >= 20:
            self.meta_learner = LogisticRegression()
            self.meta_learner.fit(oof_preds[oof_valid_mask], y[oof_valid_mask])
        else:
            self.meta_learner = LogisticRegression()
            self.meta_learner.fit(X[:, :len(model_names)], y)

        # Save model dictionary
        joblib.dump({"base_models": self.base_models, "meta_learner": self.meta_learner}, self.model_path)
        print(f"[{self.symbol}] Saved Ultra Stacking Ensemble to {self.model_path}")

    def predict_next(self, df: pd.DataFrame) -> dict:
        """
        Predicts final conviction score from Stacking Ensemble.
        """
        available_cols = [c for c in self.feature_cols if c in df.columns]
        if not self.base_models or df is None or len(df) == 0:
            return {"stacking_conviction": 0.5, "base_votes": {}}

        latest_X = df[available_cols].iloc[-1:].values

        if np.isnan(latest_X).any():
            return {"stacking_conviction": 0.5, "base_votes": {}}

        base_preds = []
        base_votes = {}

        for name, m in self.base_models.items():
            try:
                prob = float(m.predict_proba(latest_X)[0, 1])
            except Exception:
                prob = 0.5
            base_preds.append(prob)
            base_votes[name] = round(prob, 4)

        if self.meta_learner is not None:
            try:
                meta_input = np.array([base_preds])
                conviction = float(self.meta_learner.predict_proba(meta_input)[0, 1])
            except Exception:
                conviction = float(np.mean(base_preds))
        else:
            conviction = float(np.mean(base_preds))

        return {
            "stacking_conviction": round(conviction, 4),
            "base_votes": base_votes
        }
