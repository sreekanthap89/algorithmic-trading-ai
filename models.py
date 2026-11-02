import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import os
import warnings
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
from scipy.stats import spearmanr

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, brier_score_loss,
    balanced_accuracy_score, matthews_corrcoef,
)
import lightgbm as lgb

warnings.filterwarnings('ignore')

MODELS_DIR = "models"


class CalibratedBoostedEnsemble:
    """
    Multi-capacity regularized gradient-boosted ensemble combining:
      1. Shallow High-Stability XGBoost (max_depth=2, strong L1/L2 regularization)
      2. Medium Interaction XGBoost (max_depth=3, colsample/subsample regularized)
      3. Binned Histogram Gradient Boosting (HistGradientBoostingClassifier)
    Weighted by their internal purged validation log-loss / AUC skill and
    temperature-calibrated so predicted probabilities reflect true empirical win rates.
    Picklable and drop-in compatible with scikit-learn / XGBoost `.predict` & `.predict_proba`.
    """

    def __init__(
        self,
        estimators: List[Any],
        weights: List[float],
        feature_names: List[str],
        polarities: Optional[List[float]] = None,
        calibration_temp: float = 1.0,
        calibration_bias: float = 0.0,
        decision_threshold: float = 0.50,
    ):
        self.estimators = estimators
        self.weights = np.array(weights, dtype=float)
        self.weights = self.weights / np.sum(self.weights)
        self.polarities = np.array(polarities if polarities is not None else [1.0] * len(estimators), dtype=float)
        self.feature_names_in_ = np.array(feature_names)
        self.calibration_temp = float(calibration_temp)
        self.calibration_bias = float(calibration_bias)
        self.decision_threshold = float(decision_threshold)

    def predict_proba(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X_mat = X[list(self.feature_names_in_)].values
        else:
            X_mat = np.asarray(X)

        p_up = np.zeros(len(X_mat), dtype=float)
        for est, w, pol in zip(self.estimators, self.weights, self.polarities):
            prob = est.predict_proba(X_mat)[:, 1]
            if pol < 0:
                prob = 1.0 - prob
            p_up += w * prob

        p_up = np.clip(p_up, 1e-5, 1.0 - 1e-5)
        logits = np.log(p_up / (1.0 - p_up))
        cal_logits = logits * self.calibration_temp + self.calibration_bias
        cal_p_up = 1.0 / (1.0 + np.exp(-np.clip(cal_logits, -15.0, 15.0)))
        cal_p_up = np.clip(cal_p_up, 0.03, 0.97)
        return np.column_stack([1.0 - cal_p_up, cal_p_up])

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.decision_threshold).astype(int)


class TradingModel:
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
        self.model_path = os.path.join(MODELS_DIR, f"{self.safe_symbol}_{self.interval}_xgb.pkl")
        self.meta_path = os.path.join(MODELS_DIR, f"{self.safe_symbol}_{self.interval}_xgb.meta.json")
        self.model = None
        self.model_metrics: dict = {}
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
                'Return', 'Log_Return', 'Mom_3_ATR', 'Mom_5_ATR', 'Mom_10_ATR', 'Mom_20_ATR',
                'FracDiff_Close', 'Hurst_20', 'Efficiency_Ratio',
            ]

        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)

    # ------------------------------------------------------------------
    def _load_json(self, path) -> dict:
        import json
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    @staticmethod
    def _select_stationary_features(
        X_train: pd.DataFrame, y_train: np.ndarray, max_features: int = 18
    ) -> List[str]:
        """
        Walk-forward regime-consistent feature selection strictly within X_train.
        Scores each candidate feature by its Spearman Rank Information Coefficient (IC)
        across both the first half and second half of X_train, selecting features that
        have strong, sign-consistent predictive power without looking at X_test.
        """
        cols = list(X_train.columns)
        if len(cols) <= max_features or len(X_train) < 30:
            return cols

        mid = len(X_train) // 2
        scores = []
        for col in cols:
            vals = X_train[col].values
            if np.std(vals) < 1e-9:
                scores.append((col, -1.0))
                continue
            ic_full, _ = spearmanr(vals, y_train)
            ic_h1, _ = spearmanr(vals[:mid], y_train[:mid])
            ic_h2, _ = spearmanr(vals[mid:], y_train[mid:])
            ic_full = 0.0 if np.isnan(ic_full) else float(ic_full)
            ic_h1 = 0.0 if np.isnan(ic_h1) else float(ic_h1)
            ic_h2 = 0.0 if np.isnan(ic_h2) else float(ic_h2)
            # Reward sign consistency between older and newer halves of training history
            sign_bonus = 1.35 if (ic_h1 * ic_h2 > 0) else 0.75
            composite_ic = (0.5 * abs(ic_full) + 0.3 * abs(ic_h2) + 0.2 * abs(ic_h1)) * sign_bonus
            scores.append((col, composite_ic))

        scores.sort(key=lambda x: x[1], reverse=True)
        selected = [c for c, s in scores[:max_features] if s >= 0.0]
        return selected if len(selected) >= 8 else cols[:max_features]

    @staticmethod
    def _compute_time_decay_weights(y: np.ndarray, half_life_frac: float = 0.65) -> np.ndarray:
        """
        Computes exponential time-decay sample weights combined with inverse class
        frequency balancing so recent market regimes carry higher weight and class
        imbalance is neutralized.
        """
        n = len(y)
        if n == 0:
            return np.array([])
        half_life = max(15.0, n * half_life_frac)
        decay_rate = np.log(2.0) / half_life
        ages = np.arange(n - 1, -1, -1, dtype=float)
        time_w = np.exp(-decay_rate * ages)

        w_pos = max(1e-6, float(np.sum(time_w[y == 1])))
        w_neg = max(1e-6, float(np.sum(time_w[y == 0])))
        w = np.where(y == 1, time_w * (n / (2.0 * w_pos)), time_w * (n / (2.0 * w_neg)))
        return w / np.mean(w)

    def _fit_calibrated_ensemble(
        self, X_tr: pd.DataFrame, y_tr: np.ndarray, feature_names: List[str]
    ) -> CalibratedBoostedEnsemble:
        """
        Fits a 6-model multi-capacity gradient-boosted ensemble with Purged Out-of-Fold
        weighting and probability calibration strictly inside (X_tr, y_tr).
        """
        n_samples = len(X_tr)
        X_mat = X_tr[feature_names].values
        sample_w = self._compute_time_decay_weights(y_tr)

        min_leaf = max(4, min(25, n_samples // 40))
        min_child = max(2, min(10, n_samples // 40))
        n_est_shallow = min(110, max(40, n_samples // 2))
        n_est_med = min(140, max(50, int(n_samples * 0.7)))

        def _make_models():
            return [
                xgb.XGBClassifier(
                    n_estimators=n_est_shallow,
                    max_depth=2,
                    learning_rate=0.035,
                    min_child_weight=min_child,
                    subsample=0.80,
                    colsample_bytree=0.75,
                    reg_alpha=0.6,
                    reg_lambda=2.5,
                    random_state=42,
                    eval_metric='logloss',
                    objective='binary:logistic',
                ),
                xgb.XGBClassifier(
                    n_estimators=n_est_med,
                    max_depth=3,
                    learning_rate=0.04,
                    min_child_weight=min_child,
                    subsample=0.85,
                    colsample_bytree=0.80,
                    reg_alpha=0.3,
                    reg_lambda=1.8,
                    random_state=101,
                    eval_metric='logloss',
                    objective='binary:logistic',
                ),
                lgb.LGBMClassifier(
                    n_estimators=n_est_shallow,
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
                HistGradientBoostingClassifier(
                    max_depth=3,
                    max_iter=min(90, max(35, n_samples // 2)),
                    learning_rate=0.045,
                    min_samples_leaf=min_leaf,
                    l2_regularization=2.5,
                    random_state=42,
                ),
                RandomForestClassifier(
                    n_estimators=n_est_shallow,
                    max_depth=3,
                    min_samples_leaf=min_leaf,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=42,
                ),
                ExtraTreesClassifier(
                    n_estimators=n_est_shallow,
                    max_depth=4,
                    min_samples_leaf=min_leaf,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=42,
                ),
            ]

        n_models = 6
        # Purged Out-Of-Fold validation inside X_tr to learn non-negative Breiman skill weights & regime polarity
        n_splits = 4 if n_samples >= 80 else 3
        embargo = 3
        ts = TimeSeriesSplit(n_splits=n_splits)
        oof_preds = np.full((n_samples, n_models), np.nan, dtype=float)

        for tr_idx, va_idx in ts.split(X_mat):
            if len(tr_idx) > embargo + 15:
                tr_idx = tr_idx[:-embargo]
            X_f_tr, y_f_tr = X_mat[tr_idx], y_tr[tr_idx]
            w_f_tr = self._compute_time_decay_weights(y_f_tr)
            X_f_va = X_mat[va_idx]

            fold_models = _make_models()
            for m_i, m in enumerate(fold_models):
                try:
                    if isinstance(m, Pipeline):
                        m.fit(X_f_tr, y_f_tr)
                    else:
                        m.fit(X_f_tr, y_f_tr, sample_weight=w_f_tr)
                    oof_preds[va_idx, m_i] = m.predict_proba(X_f_va)[:, 1]
                except Exception:
                    oof_preds[va_idx, m_i] = 0.5

        valid_mask = ~np.isnan(oof_preds[:, 0])
        weights = [1.0 / n_models] * n_models
        polarities = [1.0] * n_models
        cal_temp = 1.0
        cal_bias = 0.0
        best_thresh = 0.50

        if valid_mask.sum() >= 15 and len(np.unique(y_tr[valid_mask])) > 1:
            y_oof = y_tr[valid_mask]
            rec_start = max(0, int(len(y_oof) * 0.45))
            y_rec = y_oof[rec_start:] if len(np.unique(y_oof[rec_start:])) > 1 else y_oof
            oof_rec = oof_preds[valid_mask][rec_start:] if len(np.unique(y_oof[rec_start:])) > 1 else oof_preds[valid_mask]

            skills = []
            for m_i in range(n_models):
                try:
                    auc_full = float(roc_auc_score(y_oof, oof_preds[valid_mask, m_i]))
                    auc_rec = float(roc_auc_score(y_rec, oof_rec[:, m_i]))
                except Exception:
                    auc_full, auc_rec = 0.50, 0.50

                polarities[m_i] = 1.0
                eff_auc = 0.50 * auc_full + 0.50 * auc_rec
                edge = max(0.0, eff_auc - 0.49)
                skills.append((edge ** 2) + 1e-3)
            weights = [s / sum(skills) for s in skills]

            blended_oof = np.zeros(len(y_oof), dtype=float)
            for i, (w, pol) in enumerate(zip(weights, polarities)):
                col_p = oof_preds[valid_mask, i]
                blended_oof += w * ((1.0 - col_p) if pol < 0 else col_p)

            # Platt calibration via LogisticRegression on OOF logits
            p_clipped = np.clip(blended_oof, 1e-4, 1.0 - 1e-4)
            logits_oof = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
            try:
                calibrator = LogisticRegression(C=1.0, solver='lbfgs', max_iter=200)
                calibrator.fit(logits_oof, y_oof)
                cal_temp = float(np.clip(calibrator.coef_[0][0], 0.35, 2.5))
                cal_bias = float(np.clip(calibrator.intercept_[0], -1.5, 1.5))
            except Exception:
                cal_temp = 1.0
                cal_bias = 0.0

            cal_logits_oof = logits_oof.ravel() * cal_temp + cal_bias
            cal_oof_probs = 1.0 / (1.0 + np.exp(-np.clip(cal_logits_oof, -15.0, 15.0)))

            # Optimize threshold for Balanced Accuracy & Youden's J statistic
            # Treats UP and DOWN with equal importance, eliminating recall skew!
            best_score = -1.0
            best_thresh = 0.50
            for cand_t in np.linspace(0.42, 0.58, 25):
                preds_t = (cal_oof_probs >= cand_t).astype(int)
                b_acc = balanced_accuracy_score(y_oof, preds_t)
                mcc = matthews_corrcoef(y_oof, preds_t)
                score_t = 0.70 * b_acc + 0.30 * max(0.0, mcc)
                if score_t > best_score:
                    best_score = score_t
                    best_thresh = float(cand_t)

        # Fit final estimators on full X_tr with time-decay sample weights
        final_models = _make_models()
        for m in final_models:
            if isinstance(m, Pipeline):
                m.fit(X_mat, y_tr)
            else:
                m.fit(X_mat, y_tr, sample_weight=sample_w)

        return CalibratedBoostedEnsemble(
            estimators=final_models,
            weights=weights,
            feature_names=feature_names,
            polarities=polarities,
            calibration_temp=cal_temp,
            calibration_bias=cal_bias,
            decision_threshold=best_thresh,
        )

    def train_or_load(
        self,
        df: pd.DataFrame,
        force_retrain: bool = False,
        test_frac: float = 0.20,
    ) -> dict:
        """
        Trains a purged, feature-selected, calibrated multi-depth boosted ensemble:
          1. Holds out the final `test_frac` of rows with a 3-bar purge/embargo gap
             for 100% leak-free Out-of-Sample (OOS) evaluation.
          2. Selects stationary features strictly inside `X_train`.
          3. Evaluates honest OOS accuracy, ROC-AUC, and Brier Score on `X_test`.
          4. Refits the production ensemble on 100% of labeled history (with exponential
             time-decay weights) so live `T+1` predictions incorporate the latest month of data.
        """
        cols = [c for c in self.feature_cols if c in df.columns and df[c].notna().any()]
        # Adaptive training window: Use all available multi-year history for daily/weekly (20+ years),
        # and deep multi-thousand bar history for intraday, preventing artificial data starvation.
        intv = getattr(self, "interval", "1d").lower()
        if intv in ("1d", "1wk", "1mo"):
            max_train_bars = 7500
        elif intv in ("1h", "4h"):
            max_train_bars = 10000
        else:
            max_train_bars = 15000
        clean_df = df.dropna(subset=cols + ['Target'])
        train_df = clean_df.tail(max_train_bars).copy() if len(clean_df) > max_train_bars else clean_df.copy()

        if len(train_df) < 40:
            print(f"[{self.symbol}] Only {len(train_df)} usable rows; keeping existing model.")
            if os.path.exists(self.model_path):
                self._load()
            return self.model_metrics

        X = train_df[cols]
        y = train_df['Target'].astype(float).values

        should_train = force_retrain or not os.path.exists(self.model_path)

        meta = self._load_json(self.meta_path)
        if os.path.exists(self.model_path) and not force_retrain:
            saved_cols = meta.get('candidate_cols', [])
            if saved_cols != cols or meta.get("model_arch") != "CalibratedBoostedEnsemble_v3":
                print(f"[{self.symbol}] Upgrading to Stationary Calibrated Ensemble v3 → retraining.")
                should_train = True

        if not should_train and os.path.exists(self.model_path):
            print(f"[{self.symbol}] Loading existing model...")
            self._load()
            return self.model_metrics

        n_test = max(10, int(len(X) * test_frac))
        embargo = min(3, max(1, len(X) // 30))
        split = max(25, len(X) - n_test)
        train_end = max(20, split - embargo)

        X_train, X_test = X.iloc[:train_end], X.iloc[split:]
        y_train, y_test = y[:train_end], y[split:]

        max_k = min(len(cols), max(10, min(18, len(X_train) // 4)))
        selected_cols = self._select_stationary_features(X_train, y_train, max_features=max_k)

        print(
            f"[{self.symbol}] Training Purged Calibrated Ensemble on {len(X)} rows "
            f"({len(X_train)} train / {embargo} purge / {len(X_test)} OOS test | {len(selected_cols)} stationary features)..."
        )

        # 1. Purged Out-of-Sample (OOS) Evaluation across X_test
        # Strictly separates X_train and X_test by `embargo` bars so multi-bar labels
        # never overlap across the train/test boundary, evaluating under a consistent
        # feature set and probability calibration.
        oos_model = self._fit_calibrated_ensemble(X_train, y_train, selected_cols)
        self.model_metrics = self._evaluate(oos_model, X_test[selected_cols], y_test)

        # 2. Refit production model on full labeled X so live T+1 inference has latest data
        selected_cols = self._select_stationary_features(X, y, max_features=max_k)
        self.model = self._fit_calibrated_ensemble(X, y, selected_cols)

        joblib.dump(self.model, self.model_path)
        self._write_meta(selected_cols, cols, len(X_train))
        print(
            f"[{self.symbol}] Saved model. OOS "
            f"acc={self.model_metrics.get('accuracy', 0):.3f} "
            f"auc={self.model_metrics.get('roc_auc', 0):.3f} "
            f"brier={self.model_metrics.get('brier_score', 0.25):.3f} "
            f"win-rate-class1={self.model_metrics.get('precision', 0):.3f}"
        )
        return self.model_metrics

    # ------------------------------------------------------------------
    def _evaluate(self, model, X_test, y_test) -> dict:
        if len(X_test) == 0 or len(np.unique(y_test)) < 2:
            return {
                "accuracy": 0.5,
                "balanced_accuracy": 0.5,
                "roc_auc": 0.5,
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
                "brier_score": 0.25,
                "mcc": 0.0,
                "note": "insufficient test data",
            }
        proba = model.predict_proba(X_test)
        p_up = proba[:, 1]
        y_pred = model.predict(X_test)
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, p_up)),
            "brier_score": float(brier_score_loss(y_test, p_up)),
            "base_rate": float(np.mean(y_test)),
            "test_rows": int(len(X_test)),
        }

    def _write_meta(self, selected_cols: List[str], candidate_cols: List[str], split: int):
        import json
        meta = {
            "symbol": self.symbol,
            "interval": self.interval,
            "model_arch": "CalibratedBoostedEnsemble_v3",
            "feature_cols": selected_cols,
            "candidate_cols": candidate_cols,
            "train_rows": int(split),
            "xgboost_version": xgb.__version__,
            "trained_at": datetime.now().isoformat(),
            "metrics": self.model_metrics,
        }
        with open(self.meta_path, 'w') as f:
            json.dump(meta, f, indent=4, default=str)

    def _load(self):
        self.model = joblib.load(self.model_path)
        meta = self._load_json(self.meta_path)
        self.model_metrics = meta.get("metrics", {})
        saved_feats = meta.get("feature_cols")
        if saved_feats:
            self.feature_cols = saved_feats

    # ------------------------------------------------------------------
    def predict_next(
        self,
        df: pd.DataFrame,
        buy_threshold: float = 0.55,
        sell_threshold: float = 0.45,
    ) -> dict:
        """
        Predicts the calibrated probability of the next move going UP.
        """
        if self.model is None:
            return {"error": "Model not trained", "prob_up": 0.5, "signal": "N/A"}

        feat_in = getattr(self.model, "feature_names_in_", None)
        cols = list(feat_in) if feat_in is not None else self.feature_cols
        cols = [c for c in cols if c in df.columns]
        latest = df[cols].iloc[-1:].copy()

        # Impute any isolated trailing NaN using last valid observation
        if latest.isna().any().any():
            latest = df[cols].ffill().bfill().iloc[-1:].fillna(0.0)

        prob = self.model.predict_proba(latest)[0]
        prob_down, prob_up = float(prob[0]), float(prob[1])
        prob_up = float(np.clip(prob_up, 0.05, 0.95))
        prob_down = float(np.clip(prob_down, 0.05, 0.95))

        if prob_up >= buy_threshold:
            signal = "BUY (UP)"
        elif prob_up <= sell_threshold:
            signal = "SELL (DOWN)"
        else:
            signal = "NEUTRAL"

        return {
            "prob_up": prob_up,
            "prob_down": prob_down,
            "signal": signal,
            "metrics": self.model_metrics,
        }


def run_monte_carlo(
    df: pd.DataFrame,
    days: int = 30,
    simulations: int = 1000,
    seed: Optional[int] = None,
    prob_up: Optional[float] = None,
) -> Optional[dict]:
    """
    Rigorously models future asset price trajectories using:
    1. Exact closed-form Geometric Brownian Motion (GBM) log-normal quantiles
       derived from Itô calculus (zero random sampling error, perfectly continuous and deterministic).
    2. Bayesian drift conditioning incorporating AI model directional probability (prob_up).
    3. Variance-reduced Antithetic Variates Monte Carlo simulation with deterministic
       bar-timestamp seeding for empirical path distributions and VaR computation.
    """
    if df is None or len(df) < 2:
        return None

    working_df = df
    returns = (
        working_df['Return'].dropna().values
        if 'Return' in working_df.columns
        else working_df['Close'].pct_change().dropna().values
    )
    if len(returns) == 0:
        return None

    mu_hist = float(np.mean(returns))
    std = float(np.std(returns))
    std = std if std > 0 else 1e-6

    if prob_up is not None and not np.isnan(prob_up):
        kappa = 1.0 / np.sqrt(max(days, 1))
        mu = mu_hist * 0.2 + (float(prob_up) - 0.5) * std * kappa
    else:
        mu = mu_hist

    last_price = float(working_df['Close'].iloc[-1])
    if np.isnan(last_price) or last_price <= 0:
        return None

    t = np.arange(1, days + 1, dtype=float)
    drift_Per_step = mu - 0.5 * (std ** 2)
    log_mean = drift_Per_step * t
    log_std = std * np.sqrt(t)

    z_10 = -1.2815515655446004
    z_50 = 0.0
    z_90 = 1.2815515655446004
    z_05 = -1.6448536269514722

    p10 = last_price * np.exp(log_mean + z_10 * log_std)
    p50 = last_price * np.exp(log_mean + z_50 * log_std)
    p90 = last_price * np.exp(log_mean + z_90 * log_std)
    expected_path = last_price * np.exp(mu * t)

    if seed is None:
        try:
            last_ts = int(pd.Timestamp(working_df.index[-1]).timestamp())
        except Exception:
            last_ts = len(working_df)
        seed = abs(hash((last_ts, round(last_price, 4), days))) % (2**31 - 1)

    rng = np.random.default_rng(seed)
    half_sims = max(1, simulations // 2)
    z_half = rng.standard_normal((days, half_sims))
    z_full = np.concatenate([z_half, -z_half], axis=1)

    step_log_returns = drift_Per_step + std * z_full
    cum_log_returns = np.cumsum(step_log_returns, axis=0)
    price_paths = last_price * np.exp(cum_log_returns)
    sim_df = pd.DataFrame(price_paths)

    final_prices = price_paths[-1, :]
    var_95_analytical = float(last_price * np.exp(log_mean[-1] + z_05 * log_std[-1]))
    var_95_empirical = float(np.percentile(final_prices, 5))
    var_95 = 0.5 * (var_95_analytical + var_95_empirical)

    from config.market_catalog import smart_round
    return {
        "simulations": sim_df,
        "mean_path": [smart_round(x) for x in expected_path],
        "p50": [smart_round(x) for x in p50],
        "p90": [smart_round(x) for x in p90],
        "p10": [smart_round(x) for x in p10],
        "var_95": smart_round(var_95),
        "expected_price": smart_round(float(p50[-1])),
        "drift_mu": float(mu),
        "volatility_sigma": float(std),
    }


def get_markov_regime(df: pd.DataFrame, window: int = 300) -> dict:
    """
    3-state Markov regime detector (Bull / Bear / Choppy) with Laplace-smoothed
    transition probabilities and persistence scoring.
    """
    fallback = {
        "regime": "UNKNOWN",
        "state_index": 2,
        "transition_matrix": [[0.33, 0.33, 0.34]] * 3,
        "state_probs": [0.33, 0.33, 0.34],
        "markov_score": 0.5,
    }
    if df is None or len(df) < 30 or 'Return' not in df.columns:
        return fallback

    recent_df = df.tail(window).copy()
    returns = recent_df['Return'].dropna().values
    if len(returns) < 15:
        return fallback

    std_ret = np.std(returns) if np.std(returns) > 0 else 0.01
    states = []
    for r in returns:
        if r > 0.45 * std_ret:
            states.append(0)   # Bull
        elif r < -0.45 * std_ret:
            states.append(1)   # Bear
        else:
            states.append(2)   # Choppy

    # Laplace smoothing (0.5 prior) prevents zero-probability transition cells
    trans_counts = np.full((3, 3), 0.5, dtype=float)
    for i in range(len(states) - 1):
        trans_counts[states[i]][states[i + 1]] += 1.0

    row_sums = trans_counts.sum(axis=1, keepdims=True)
    p_matrix = trans_counts / row_sums

    current_state = states[-1]
    state_probs = p_matrix[current_state].tolist()

    regime_names = {0: "BULL TREND", 1: "BEAR TREND", 2: "CHOPPY / RANGE"}
    state_label = regime_names[current_state]
    conviction = max(state_probs)

    return {
        "regime": state_label,
        "state_index": current_state,
        "transition_matrix": p_matrix.tolist(),
        "state_probs": [round(p, 4) for p in state_probs],
        "markov_score": round(float(conviction), 4),
    }


def predict_next_move(
    df_feat: pd.DataFrame,
    symbol: str = "BTC-USD",
    interval: str = "1d",
    buy_threshold: float = 0.56,
    sell_threshold: float = 0.44,
) -> dict:
    """
    Convenience helper that loads/trains TradingModel and returns a hybrid
    XGBoost + TimesFM-3 prediction payload.
    """
    model = TradingModel(symbol, interval)
    model.train_or_load(df_feat, force_retrain=False)
    xgb_pred = model.predict_next(df_feat, buy_threshold=buy_threshold, sell_threshold=sell_threshold)
    try:
        from ml.timesfm_engine import run_timesfm_forecast
        tfm = run_timesfm_forecast(df_feat, symbol=symbol, interval=interval, horizon=15)
        xgb_pred["timesfm"] = tfm
    except Exception:
        pass
    return xgb_pred
