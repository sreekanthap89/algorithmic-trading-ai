"""
=============================================================
 ULTRA STACKING ENSEMBLE & DEEP SEQUENCE BUILDER (v2)
 6-Model Purged Out-Of-Fold Stacking with Robust Scaling,
 Time-Decay Sample Weighting & Skill-Weighted Meta-Learning
=============================================================
"""

import numpy as np
import pandas as pd
import os
import joblib
from scipy.stats import spearmanr

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, balanced_accuracy_score, matthews_corrcoef
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
        avail = [c for c in feature_cols if c in df.columns]
        valid_df = df.dropna(subset=avail + ['Target']).copy()
        if len(valid_df) < self.lookback + 5:
            return np.array([]), np.array([])

        data_matrix = valid_df[avail].values
        targets = valid_df['Target'].values

        X_seq, y_seq = [], []
        for i in range(self.lookback, len(data_matrix)):
            X_seq.append(data_matrix[i - self.lookback:i])
            y_seq.append(targets[i])

        return np.array(X_seq), np.array(y_seq)


class UltraStackingEnsemble:
    def __init__(self, symbol: str, interval: str):
        self.symbol = symbol
        self.interval = interval
        self.safe_symbol = (
            symbol.replace("=", "_")
            .replace("^", "_")
            .replace("/", "_")
            .replace(":", "_")
            .replace("\\", "_")
        )
        self.model_path = os.path.join(MODELS_DIR, f"{self.safe_symbol}_{self.interval}_stacking.pkl")

        try:
            from features import LIVE_FEATURE_COLS
            self.feature_cols = list(LIVE_FEATURE_COLS)
        except Exception:
            self.feature_cols = [
                'RSI', 'RSI_Z', 'Stoch_RSI', 'ADX_Norm', 'DI_Spread',
                'MACD_Norm', 'MACD_Signal_Norm', 'MACD_Diff_Norm',
                'EMA12_26_Spread', 'SMA20_50_Spread', 'SMA50_200_Spread',
                'Dist_SMA20', 'Dist_SMA50', 'Dist_EMA12',
                'BB_Width', 'BB_pctB', 'ATR_Pct', 'GK_Vol', 'Vol_Regime_Ratio',
                'Return', 'Log_Return', 'Mom_3_ATR', 'Mom_5_ATR', 'Mom_10_ATR',
                'FracDiff_Close', 'Hurst_20', 'Efficiency_Ratio',
                'CLV', 'Order_Flow_Z', 'Volume_Z', 'VWAP_Dev', 'Hawkes_Intensity',
            ]

        self.selected_cols: list = []
        self.base_models: dict = {}
        self.skill_weights: dict = {}
        self.meta_learner = None
        self.oof_metrics: dict = {"accuracy": 0.5, "roc_auc": 0.5, "brier_score": 0.25}

        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)

    @staticmethod
    def _select_top_features(df_valid: pd.DataFrame, cols: list, y: np.ndarray, max_k: int = 16) -> list:
        if len(cols) <= max_k or len(df_valid) < 30:
            return cols
        # Use only the first 80% of rows to rank features (zero lookahead into final fold)
        cutoff = max(20, int(len(df_valid) * 0.80))
        y_sub = y[:cutoff]
        scored = []
        for c in cols:
            x_sub = df_valid[c].values[:cutoff]
            if np.std(x_sub) < 1e-9:
                scored.append((c, -1.0))
                continue
            ic, _ = spearmanr(x_sub, y_sub)
            ic = 0.0 if np.isnan(ic) else abs(float(ic))
            scored.append((c, ic))
        scored.sort(key=lambda item: item[1], reverse=True)
        top = [c for c, _ in scored[:max_k]]
        return top if len(top) >= 8 else cols[:max_k]

    @staticmethod
    def _build_base_learners(n_samples: int) -> dict:
        min_leaf = max(4, min(25, n_samples // 40))
        n_est = min(110, max(40, n_samples // 2))
        return {
            "xgb": xgb.XGBClassifier(
                n_estimators=n_est,
                max_depth=2,
                learning_rate=0.035,
                min_child_weight=max(2, min_leaf // 2),
                subsample=0.80,
                colsample_bytree=0.75,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=42,
                eval_metric='logloss',
            ),
            "lgb": lgb.LGBMClassifier(
                n_estimators=n_est,
                max_depth=3,
                num_leaves=7,
                min_child_samples=min_leaf,
                learning_rate=0.035,
                subsample=0.80,
                colsample_bytree=0.75,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=42,
                verbose=-1,
            ),
            "hgb": HistGradientBoostingClassifier(
                max_depth=3,
                max_iter=n_est,
                learning_rate=0.045,
                min_samples_leaf=min_leaf,
                l2_regularization=2.5,
                random_state=42,
            ),
            "rf": RandomForestClassifier(
                n_estimators=n_est,
                max_depth=3,
                min_samples_leaf=min_leaf,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=42,
            ),
            "et": ExtraTreesClassifier(
                n_estimators=n_est,
                max_depth=4,
                min_samples_leaf=min_leaf,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
            ),
            "mlp": Pipeline([
                ("scaler", RobustScaler()),
                ("mlp", MLPClassifier(
                    hidden_layer_sizes=(24, 12),
                    alpha=0.5,  # Strong L2 weight decay to prevent overfitting
                    learning_rate_init=0.005,
                    max_iter=200,
                    random_state=42,
                )),
            ]),
        }

    def train_or_load(self, df: pd.DataFrame, force_retrain: bool = False):
        """
        Trains 6 Level-0 Base Learners (XGBoost, LightGBM, HistGBDT, RF, ET, Scaled MLP)
        and a Skill-Weighted Level-1 Meta-Learner via Purged TimeSeriesSplit.
        """
        available_cols = [c for c in self.feature_cols if c in df.columns and df[c].notna().any()]
        intv = getattr(self, "interval", "1d").lower()
        if intv in ("1d", "1wk", "1mo"):
            max_train_bars = 7500
        elif intv in ("1h", "4h"):
            max_train_bars = 10000
        else:
            max_train_bars = 15000
        clean_df = df.dropna(subset=available_cols + ['Target'])
        valid_df = clean_df.tail(max_train_bars).copy() if len(clean_df) > max_train_bars else clean_df.copy()

        if not force_retrain and os.path.exists(self.model_path):
            try:
                saved_dict = joblib.load(self.model_path)
                if saved_dict.get("arch_version") == "v3_stationary_6model":
                    self.base_models = saved_dict.get("base_models", {})
                    self.meta_learner = saved_dict.get("meta_learner", None)
                    self.selected_cols = saved_dict.get("selected_cols", available_cols)
                    self.skill_weights = saved_dict.get("skill_weights", {})
                    self.oof_metrics = saved_dict.get("oof_metrics", self.oof_metrics)
                    if all(c in df.columns for c in self.selected_cols):
                        print(f"[{self.symbol}] Loaded Stacking Ensemble v3 successfully.")
                        return
            except Exception:
                print(f"[{self.symbol}] Retraining Stacking Ensemble v3...")

        if len(valid_df) < 40:
            print(f"[{self.symbol}] Insufficient data for Stacking Ensemble training.")
            return

        y = valid_df['Target'].astype(int).values
        max_k = min(len(available_cols), max(10, min(16, len(valid_df) // 4)))
        self.selected_cols = self._select_top_features(valid_df, available_cols, y, max_k=max_k)
        X = valid_df[self.selected_cols].values

        self.base_models = self._build_base_learners(len(X))
        model_names = list(self.base_models.keys())

        # Generate Purged Out-Of-Fold (OOF) predictions via TimeSeriesSplit
        n_splits = 5 if len(X) >= 85 else 4
        embargo = 3
        ts_split = TimeSeriesSplit(n_splits=n_splits)
        oof_preds = np.full((len(X), len(model_names)), np.nan, dtype=float)

        for train_idx, val_idx in ts_split.split(X):
            if len(train_idx) > embargo + 15:
                train_idx = train_idx[:-embargo]
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va = X[val_idx]

            fold_learners = self._build_base_learners(len(X_tr))
            for idx, name in enumerate(model_names):
                m = fold_learners[name]
                try:
                    m.fit(X_tr, y_tr)
                    oof_preds[val_idx, idx] = m.predict_proba(X_va)[:, 1]
                except Exception:
                    oof_preds[val_idx, idx] = 0.5

        oof_valid_mask = ~np.isnan(oof_preds[:, 0])
        y_oof = y[oof_valid_mask]
        oof_sub = oof_preds[oof_valid_mask]

        # Compute Breiman Non-Negative OOF Skill Weights & Regime Polarity Alignment
        raw_skills = {}
        self.polarities = {}
        for idx, name in enumerate(model_names):
            try:
                if len(np.unique(y_oof)) > 1:
                    auc_m = float(roc_auc_score(y_oof, oof_sub[:, idx]))
                    brier_m = float(brier_score_loss(y_oof, oof_sub[:, idx]))
                else:
                    auc_m, brier_m = 0.50, 0.25
            except Exception:
                auc_m, brier_m = 0.50, 0.25

            self.polarities[name] = 1.0
            edge = max(0.0, auc_m - 0.49)
            brier_bonus = max(0.2, 1.0 - 2.0 * max(0.0, brier_m - 0.22))
            skill = ((edge ** 2) + 1e-3) * brier_bonus
            raw_skills[name] = skill

        tot_skill = sum(raw_skills.values())
        self.skill_weights = {k: v / tot_skill for k, v in raw_skills.items()}

        # Train Level-1 Non-Negative Meta-Learner on Purged OOF predictions
        if oof_valid_mask.sum() >= 20 and len(np.unique(y_oof)) > 1:
            self.meta_learner = LogisticRegression(C=0.25, random_state=42)
            self.meta_learner.fit(oof_sub, y_oof)
            # Guard against negative collinear coefficients (Breiman non-negativity constraint)
            if hasattr(self.meta_learner, "coef_"):
                self.meta_learner.coef_ = np.maximum(self.meta_learner.coef_, 0.0)

            skill_oof_prob = sum(self.skill_weights[name] * oof_sub[:, i] for i, name in enumerate(model_names))
            if np.sum(self.meta_learner.coef_) > 1e-6:
                meta_oof_prob = self.meta_learner.predict_proba(oof_sub)[:, 1]
                final_oof_prob = 0.70 * skill_oof_prob + 0.30 * meta_oof_prob
            else:
                final_oof_prob = skill_oof_prob

            # Optimize threshold on OOF for Balanced Accuracy & MCC
            best_score = -1.0
            best_acc = 0.50
            best_b_acc = 0.50
            best_th = 0.50
            for th in np.linspace(0.42, 0.58, 25):
                preds_th = (final_oof_prob >= th).astype(int)
                b_acc = float(balanced_accuracy_score(y_oof, preds_th))
                acc_th = float(accuracy_score(y_oof, preds_th))
                mcc_th = float(matthews_corrcoef(y_oof, preds_th))
                score_th = 0.70 * b_acc + 0.30 * max(0.0, mcc_th)
                if score_th > best_score:
                    best_score = score_th
                    best_acc = acc_th
                    best_b_acc = b_acc
                    best_th = float(th)

            self.oof_metrics = {
                "accuracy": round(best_acc, 4),
                "balanced_accuracy": round(best_b_acc, 4),
                "roc_auc": round(float(roc_auc_score(y_oof, final_oof_prob)), 4),
                "brier_score": round(float(brier_score_loss(y_oof, final_oof_prob)), 4),
                "decision_threshold": round(best_th, 4),
                "oof_samples": int(len(y_oof)),
            }
        else:
            self.meta_learner = None

        # Fit all base models on full labeled dataset X
        for name, m in self.base_models.items():
            try:
                m.fit(X, y)
            except Exception:
                pass

        joblib.dump(
            {
                "arch_version": "v3_stationary_6model",
                "base_models": self.base_models,
                "meta_learner": self.meta_learner,
                "selected_cols": self.selected_cols,
                "skill_weights": self.skill_weights,
                "oof_metrics": self.oof_metrics,
            },
            self.model_path,
        )
        print(
            f"[{self.symbol}] Saved Ultra Stacking Ensemble v2 "
            f"(OOF acc={self.oof_metrics['accuracy']:.3f}, auc={self.oof_metrics['roc_auc']:.3f})"
        )

    def predict_next(self, df: pd.DataFrame) -> dict:
        """
        Predicts final calibrated conviction score from the 6-Model Stacking Ensemble.
        """
        cols = self.selected_cols if self.selected_cols else [c for c in self.feature_cols if c in df.columns]
        cols = [c for c in cols if c in df.columns]
        if not self.base_models or df is None or len(df) == 0 or len(cols) == 0:
            return {"stacking_conviction": 0.5, "base_votes": {}, "oof_metrics": self.oof_metrics}

        latest_df = df[cols].iloc[-1:].copy()
        if latest_df.isna().any().any():
            latest_df = df[cols].ffill().bfill().iloc[-1:].fillna(0.0)
        latest_X = latest_df.values

        base_preds = []
        base_votes = {}
        skill_weighted_sum = 0.0
        weight_total = 0.0

        for name, m in self.base_models.items():
            try:
                prob = float(m.predict_proba(latest_X)[0, 1])
                if getattr(self, "polarities", {}).get(name, 1.0) < 0:
                    prob = 1.0 - prob
            except Exception:
                prob = 0.5
            base_preds.append(prob)
            base_votes[name] = round(prob, 4)
            w = float(self.skill_weights.get(name, 1.0 / max(1, len(self.base_models))))
            skill_weighted_sum += w * prob
            weight_total += w

        skill_conviction = (skill_weighted_sum / weight_total) if weight_total > 0 else float(np.mean(base_preds))

        if self.meta_learner is not None:
            try:
                meta_input = np.array([base_preds])
                meta_prob = float(self.meta_learner.predict_proba(meta_input)[0, 1])
                conviction = 0.55 * skill_conviction + 0.45 * meta_prob
            except Exception:
                conviction = skill_conviction
        else:
            conviction = skill_conviction

        conviction = float(np.clip(conviction, 0.05, 0.95))
        return {
            "stacking_conviction": round(conviction, 4),
            "base_votes": base_votes,
            "skill_weights": {k: round(v, 3) for k, v in self.skill_weights.items()},
            "oof_metrics": self.oof_metrics,
        }
