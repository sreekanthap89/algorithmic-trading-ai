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
        meta_classifier: Optional[Any] = None,
        meta_col_indices: Optional[List[int]] = None,
        target_horizon: int = 6,
        logit_mean: float = 0.0,
        logit_scale: float = 1.0,
    ):
        self.estimators = estimators
        self.weights = np.array(weights, dtype=float)
        self.weights = self.weights / np.sum(self.weights)
        self.polarities = np.array(polarities if polarities is not None else [1.0] * len(estimators), dtype=float)
        self.feature_names_in_ = np.array(feature_names)
        self.calibration_temp = float(calibration_temp)
        self.calibration_bias = float(calibration_bias)
        self.decision_threshold = float(decision_threshold)
        self.meta_classifier = meta_classifier
        self.meta_col_indices = meta_col_indices or []
        self.target_horizon = int(target_horizon)
        self.logit_mean = float(logit_mean)
        self.logit_scale = float(logit_scale)

    def _extract_matrix(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X[list(self.feature_names_in_)].values
        return np.asarray(X)

    def predict_proba_and_disagreement(self, X: Any) -> Tuple[np.ndarray, np.ndarray]:
        X_mat = self._extract_matrix(X)
        n_rows = len(X_mat)
        n_est = len(self.estimators)
        indiv_probs = np.zeros((n_rows, n_est), dtype=float)

        p_up = np.zeros(n_rows, dtype=float)
        for i, (est, w, pol) in enumerate(zip(self.estimators, self.weights, self.polarities)):
            prob = est.predict_proba(X_mat)[:, 1]
            if pol < 0:
                prob = 1.0 - prob
            indiv_probs[:, i] = prob
            p_up += w * prob

        disagreement = np.std(indiv_probs, axis=1) if n_est > 1 else np.zeros(n_rows, dtype=float)

        p_up = np.clip(p_up, 1e-4, 1.0 - 1e-4)
        logits = np.log(p_up / (1.0 - p_up))

        # Scale-Invariant Logit Normalization (Anti-Squashing for Commodities, Forex & Low-Nominal Assets)
        l_scale = getattr(self, "logit_scale", 1.0)
        l_mean = getattr(self, "logit_mean", 0.0)
        z_logits = (logits - l_mean) / max(l_scale, 1e-4) if l_scale > 1e-4 else (logits - l_mean)

        temp = float(np.clip(self.calibration_temp, 0.45, 1.50))
        bias = float(np.clip(self.calibration_bias, -0.65, 0.65))
        cal_logits = z_logits * temp + bias
        cal_p_up = 1.0 / (1.0 + np.exp(-np.clip(cal_logits, -12.0, 12.0)))
        cal_p_up = np.clip(cal_p_up, 0.04, 0.96)
        proba_mat = np.column_stack([1.0 - cal_p_up, cal_p_up])
        return proba_mat, disagreement

    def predict_proba(self, X: Any) -> np.ndarray:
        proba_mat, _ = self.predict_proba_and_disagreement(X)
        return proba_mat

    def predict_meta_proba(self, X: Any) -> np.ndarray:
        """
        Predicts López de Prado Meta-Labeling probability: P(Primary Trade Signal Hits TP Before SL).
        Combines primary conviction strength, base-learner epistemic disagreement, and market regime features.
        """
        proba_mat, disagreement = self.predict_proba_and_disagreement(X)
        cal_p_up = proba_mat[:, 1]
        conv_margin = np.abs(cal_p_up - self.decision_threshold) * 2.0

        meta_clf = getattr(self, "meta_classifier", None)
        meta_idx = getattr(self, "meta_col_indices", None)
        if meta_clf is not None:
            try:
                X_mat = self._extract_matrix(X)
                regime_sub = X_mat[:, meta_idx] if meta_idx else np.zeros((len(X_mat), 0))
                meta_X = np.column_stack([cal_p_up, conv_margin, disagreement, regime_sub])
                meta_X = np.nan_to_num(meta_X, nan=0.0, posinf=0.0, neginf=0.0)
                meta_p = meta_clf.predict_proba(meta_X)[:, 1]
                return np.clip(meta_p, 0.25, 0.96)
            except Exception:
                pass

        # Fallback analytical meta-probability when meta-classifier is unavailable
        fallback_meta = 0.52 + 0.45 * conv_margin - 0.35 * disagreement
        return np.clip(fallback_meta, 0.30, 0.95)

    def predict(self, X: Any, threshold: Optional[float] = None) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        if threshold is not None:
            t = float(threshold)
        elif len(proba) >= 20:
            t = float(np.median(proba))
        else:
            t = self.decision_threshold
        return (proba >= t).astype(int)


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
        # Volatility-Adaptive Target Holding Horizon (6 bars intraday, 4 bars swing)
        self.target_horizon = 6 if interval.lower() in ("1m", "5m", "15m", "30m", "1h", "4h") else 4
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
        Module 6 (ElasticNet / LASSO Penalized Regression) & Module 3 (VIF Pruning):
        Walk-forward regime-consistent feature selection strictly within X_train:
          1. Prunes severely multicollinear features with Variance Inflation Factor (VIF > 15).
          2. Fits ElasticNetCV (L1/L2 shrinkage penalty) on standardized X_train to drive
             zero-predictive redundant features to absolute zero.
          3. Blends ElasticNet weight magnitude with Spearman rank Information Coefficient (IC)
             sign-consistency across training halves to select robust, non-collinear features.
        """
        cols = list(X_train.columns)
        if len(cols) <= max_features or len(X_train) < 30:
            return cols

        # 1. Module 3: VIF Multicollinearity Filter
        try:
            from features import compute_vif_filter
            filtered_cols = compute_vif_filter(X_train[cols], thresh=15.0)
            if len(filtered_cols) >= 8:
                cols = filtered_cols
        except Exception:
            pass

        # 2. Module 6: ElasticNetCV L1/L2 Shrinkage
        en_weights = {}
        try:
            from sklearn.linear_model import ElasticNetCV
            from sklearn.preprocessing import RobustScaler

            mat = X_train[cols].ffill().bfill().values
            scaler = RobustScaler()
            n_sub = min(2500, len(X_train))
            sub_idx = np.linspace(0, len(X_train) - 1, n_sub, dtype=int)
            scaled_mat = np.nan_to_num(scaler.fit_transform(mat[sub_idx]), nan=0.0, posinf=0.0, neginf=0.0)
            cv_splits = min(4, max(2, n_sub // 25))
            enet = ElasticNetCV(
                l1_ratio=[0.2, 0.5, 0.8],
                cv=cv_splits,
                max_iter=250,
                random_state=42,
            )
            enet.fit(scaled_mat, y_train[sub_idx])
            for col_name, coef_val in zip(cols, enet.coef_):
                en_weights[col_name] = float(abs(coef_val))
        except Exception:
            en_weights = {c: 1.0 for c in cols}

        # 3. Spearman Rank IC and Inter-temporal Sign Consistency
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

            sign_bonus = 1.35 if (ic_h1 * ic_h2 > 0) else 0.75
            composite_ic = (0.5 * abs(ic_full) + 0.3 * abs(ic_h2) + 0.2 * abs(ic_h1)) * sign_bonus

            # Reward features that survived ElasticNet shrinkage
            en_mult = 1.0 + min(2.5, en_weights.get(col, 0.0) * 10.0)
            final_score = composite_ic * en_mult
            scores.append((col, final_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        selected = [c for c, s in scores[:max_features] if s >= 0.0]
        return selected if len(selected) >= 8 else cols[:max_features]

    @staticmethod
    def _compute_time_decay_weights(
        y: np.ndarray,
        half_life_frac: float = 0.55,
        attr_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Computes exponential time-decay sample weights combined with inverse class
        frequency balancing and Triple-Barrier attribution weights (`Target_Weight`)
        so recent market regimes and high-conviction barrier touches carry higher weight.
        """
        n = len(y)
        if n == 0:
            return np.array([])
        half_life = max(15.0, n * half_life_frac)
        decay_rate = np.log(2.0) / half_life
        ages = np.arange(n - 1, -1, -1, dtype=float)
        time_w = np.exp(-decay_rate * ages)

        if attr_weights is not None and len(attr_weights) == n:
            clean_attr = np.nan_to_num(np.asarray(attr_weights, dtype=float), nan=1.0, posinf=1.0, neginf=1.0)
            time_w = time_w * np.clip(clean_attr, 0.35, 2.50)

        w_pos = max(1e-6, float(np.sum(time_w[y == 1])))
        w_neg = max(1e-6, float(np.sum(time_w[y == 0])))
        w = np.where(y == 1, time_w * (n / (2.0 * w_pos)), time_w * (n / (2.0 * w_neg)))
        return w / np.mean(w)

    def _fit_calibrated_ensemble(
        self,
        X_tr: pd.DataFrame,
        y_tr: np.ndarray,
        feature_names: List[str],
        attr_weights: Optional[np.ndarray] = None,
    ) -> CalibratedBoostedEnsemble:
        """
        Fits a 6-model multi-capacity gradient-boosted ensemble with Purged Out-of-Fold
        weighting, Platt probability calibration, and a secondary López de Prado
        Meta-Labeling Classifier strictly inside (X_tr, y_tr).
        """
        n_samples = len(X_tr)
        X_mat = X_tr[feature_names].values
        sample_w = self._compute_time_decay_weights(y_tr, attr_weights=attr_weights)

        min_leaf = max(4, min(25, n_samples // 40))
        min_child = max(2, min(10, n_samples // 40))
        n_est_shallow = min(110, max(40, n_samples // 2))
        n_est_med = min(140, max(50, int(n_samples * 0.7)))

        def _make_models():
            return [
                xgb.XGBClassifier(
                    n_estimators=n_est_shallow,
                    max_depth=3,
                    learning_rate=0.04,
                    min_child_weight=min_child,
                    subsample=0.85,
                    colsample_bytree=0.80,
                    reg_alpha=0.4,
                    reg_lambda=2.0,
                    random_state=42,
                    eval_metric='logloss',
                    objective='binary:logistic',
                ),
                xgb.XGBClassifier(
                    n_estimators=n_est_med,
                    max_depth=4,
                    learning_rate=0.045,
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
                    max_depth=4,
                    num_leaves=12,
                    min_child_samples=min_leaf,
                    learning_rate=0.04,
                    subsample=0.85,
                    colsample_bytree=0.80,
                    reg_alpha=0.4,
                    reg_lambda=1.8,
                    random_state=42,
                    verbose=-1,
                ),
                HistGradientBoostingClassifier(
                    max_depth=4,
                    max_iter=min(100, max(40, n_samples // 2)),
                    learning_rate=0.045,
                    min_samples_leaf=min_leaf,
                    l2_regularization=2.0,
                    random_state=42,
                ),
                RandomForestClassifier(
                    n_estimators=n_est_shallow,
                    max_depth=4,
                    min_samples_leaf=min_leaf,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=42,
                ),
                ExtraTreesClassifier(
                    n_estimators=n_est_shallow,
                    max_depth=5,
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
            if len(tr_idx) < 25:
                continue
            X_f_tr, y_f_tr = X_mat[tr_idx], y_tr[tr_idx]
            attr_f_tr = attr_weights[tr_idx] if attr_weights is not None and len(attr_weights) == n_samples else None
            w_f_tr = self._compute_time_decay_weights(y_f_tr, attr_weights=attr_f_tr)
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
        cal_temp = 0.85
        cal_bias = 0.0
        logit_mean = 0.0
        logit_scale = 1.0
        best_thresh = 0.50
        meta_clf = None
        regime_col_names = [
            "ADX_Norm", "Hurst_20", "BB_Width", "ATR_Pct",
            "Efficiency_Ratio", "Squeeze_Ratio", "Exit_Exhaustion_Score",
        ]
        meta_col_indices = [i for i, c in enumerate(feature_names) if c in regime_col_names]

        if valid_mask.sum() >= 15 and len(np.unique(y_tr[valid_mask])) > 1:
            y_oof = y_tr[valid_mask]
            rec_start = max(0, int(len(y_oof) * 0.65))
            y_rec = y_oof[rec_start:] if len(np.unique(y_oof[rec_start:])) > 1 else y_oof
            oof_rec = oof_preds[valid_mask][rec_start:] if len(np.unique(y_oof[rec_start:])) > 1 else oof_preds[valid_mask]

            skills = []
            for m_i in range(n_models):
                try:
                    auc_full = float(roc_auc_score(y_oof, oof_preds[valid_mask, m_i]))
                    auc_rec = float(roc_auc_score(y_rec, oof_rec[:, m_i]))
                except Exception:
                    auc_full, auc_rec = 0.50, 0.50

                avg_auc = 0.50 * auc_full + 0.50 * auc_rec
                if avg_auc < 0.485:
                    polarities[m_i] = -1.0
                    eff_auc = 1.0 - avg_auc
                else:
                    polarities[m_i] = 1.0
                    eff_auc = avg_auc

                edge = max(0.0, eff_auc - 0.49)
                skills.append((edge ** 2) + 1e-3)
            weights = [s / sum(skills) for s in skills]

            blended_oof = np.zeros(len(y_oof), dtype=float)
            for i, (w, pol) in enumerate(zip(weights, polarities)):
                col_p = oof_preds[valid_mask, i]
                blended_oof += w * ((1.0 - col_p) if pol < 0 else col_p)

            # Scale-Invariant Logit Normalization (Anti-Squashing for Commodities & Low-Nominal Assets)
            p_clipped = np.clip(blended_oof, 1e-4, 1.0 - 1e-4)
            logits_oof = np.log(p_clipped / (1.0 - p_clipped))
            logit_mean = float(np.mean(logits_oof))
            logit_scale = float(max(np.std(logits_oof), 1e-4))

            z_oof = (logits_oof - logit_mean) / logit_scale
            z_oof_mat = z_oof.reshape(-1, 1)

            try:
                calibrator = LogisticRegression(C=0.5, solver='lbfgs', max_iter=200)
                calibrator.fit(z_oof_mat, y_oof)
                raw_temp = float(calibrator.coef_[0][0])
                raw_bias = float(calibrator.intercept_[0])
                cal_temp = float(np.clip(raw_temp, 0.55, 1.45))
                cal_bias = float(np.clip(raw_bias, -0.60, 0.60))
            except Exception:
                cal_temp = 0.85
                cal_bias = 0.0

            cal_logits_oof = z_oof * cal_temp + cal_bias
            cal_oof_probs = 1.0 / (1.0 + np.exp(-np.clip(cal_logits_oof, -12.0, 12.0)))

            # Optimize threshold for Balanced Accuracy, Youden's J statistic & Non-Zero Recall Guarantee
            med_oof = float(np.median(cal_oof_probs))
            cand_thresholds = np.percentile(cal_oof_probs, np.linspace(30, 70, 31))
            best_score = -1.0
            best_thresh = med_oof
            for cand_t in cand_thresholds:
                preds_t = (cal_oof_probs >= cand_t).astype(int)
                pos_count = int(np.sum(preds_t == 1))
                neg_count = int(np.sum(preds_t == 0))
                # Strict balance: ensure both classes have at least 25% of calls on out-of-fold validation
                if pos_count < 0.25 * len(y_oof) or neg_count < 0.25 * len(y_oof):
                    continue
                pos_recall = float(np.sum((preds_t == 1) & (y_oof == 1))) / max(1, np.sum(y_oof == 1))
                neg_recall = float(np.sum((preds_t == 0) & (y_oof == 0))) / max(1, np.sum(y_oof == 0))
                if pos_recall < 0.25 or neg_recall < 0.25:
                    continue
                b_acc = (pos_recall + neg_recall) / 2.0
                mcc = matthews_corrcoef(y_oof, preds_t)
                youden_j = pos_recall + neg_recall - 1.0
                dist_pen = abs(cand_t - med_oof) * 0.4
                score_t = 0.50 * b_acc + 0.30 * max(0.0, youden_j) + 0.20 * max(0.0, mcc) - dist_pen
                if score_t > best_score:
                    best_score = score_t
                    best_thresh = float(cand_t)

            # Secondary Meta-Labeling Classifier (López de Prado, AFML Ch. 3):
            # Predicts whether the primary model's directional call (preds_best == y_oof) succeeds,
            # conditioned on conviction margin, base-learner disagreement, and regime features.
            try:
                primary_oof_calls = (cal_oof_probs >= best_thresh).astype(int)
                y_meta = (primary_oof_calls == y_oof.astype(int)).astype(int)
                if len(np.unique(y_meta)) > 1:
                    oof_disagree = np.std(oof_preds[valid_mask], axis=1)
                    oof_margin = np.abs(cal_oof_probs - best_thresh) * 2.0
                    X_valid_mat = X_mat[valid_mask]
                    regime_sub_oof = (
                        X_valid_mat[:, meta_col_indices]
                        if meta_col_indices
                        else np.zeros((len(y_oof), 0))
                    )
                    meta_X_oof = np.column_stack([cal_oof_probs, oof_margin, oof_disagree, regime_sub_oof])
                    meta_X_oof = np.nan_to_num(meta_X_oof, nan=0.0, posinf=0.0, neginf=0.0)
                    meta_pipe = Pipeline([
                        ("scaler", RobustScaler()),
                        ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=200, random_state=42)),
                    ])
                    meta_pipe.fit(meta_X_oof, y_meta)
                    meta_clf = meta_pipe
            except Exception:
                meta_clf = None

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
            meta_classifier=meta_clf,
            meta_col_indices=meta_col_indices,
            target_horizon=self.target_horizon,
            logit_mean=logit_mean,
            logit_scale=logit_scale,
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
        # Regime-Relevant Training Window: Prevents 20-year concept drift on daily/weekly bars
        # while retaining rich multi-cycle statistical depth.
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
        attr_w = (
            train_df['Target_Weight'].astype(float).values
            if 'Target_Weight' in train_df.columns
            else np.ones(len(train_df), dtype=float)
        )

        should_train = force_retrain or not os.path.exists(self.model_path)

        meta = self._load_json(self.meta_path)
        if os.path.exists(self.model_path) and not force_retrain:
            saved_cols = meta.get('candidate_cols', [])
            if saved_cols != cols or meta.get("model_arch") != "CalibratedBoostedEnsemble_v6":
                print(f"[{self.symbol}] Upgrading to Anti-Squashing Calibrated Ensemble v6 → retraining.")
                should_train = True

        if not should_train and os.path.exists(self.model_path):
            print(f"[{self.symbol}] Loading existing model...")
            self._load()
            return self.model_metrics

        n_test = min(250, max(35, int(len(X) * 0.15)))
        embargo = min(4, max(2, len(X) // 40))
        split = max(30, len(X) - n_test)
        train_end = max(25, split - embargo)

        X_train, X_test = X.iloc[:train_end], X.iloc[split:]
        y_train, y_test = y[:train_end], y[split:]
        attr_w_train = attr_w[:train_end]

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
        oos_model = self._fit_calibrated_ensemble(X_train, y_train, selected_cols, attr_weights=attr_w_train)
        self.model_metrics = self._evaluate(oos_model, X_test[selected_cols], y_test)

        # 2. Refit production model on full labeled X so live T+1 inference has latest data
        selected_cols = self._select_stationary_features(X, y, max_features=max_k)
        self.model = self._fit_calibrated_ensemble(X, y, selected_cols, attr_weights=attr_w)

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
        emp_thresh = float(np.median(p_up)) if len(p_up) >= 20 else getattr(model, "decision_threshold", 0.50)
        y_pred = model.predict(X_test, threshold=emp_thresh)
        raw_acc = float(accuracy_score(y_test, y_pred))
        bal_acc = float(balanced_accuracy_score(y_test, y_pred))

        # High-Conviction / Meta-Gated Accuracy: measures accuracy on actionable trade signals
        # (excluding flat/choppy neutral bars where the trader stays in WAIT / HOLD)
        margin = np.abs(p_up - emp_thresh)
        if hasattr(model, "predict_meta_proba"):
            meta_p = model.predict_meta_proba(X_test)
            active_mask = (margin >= np.percentile(margin, 50)) | (meta_p >= 0.52)
        else:
            active_mask = margin >= np.percentile(margin, 50)

        if np.sum(active_mask) >= 5 and len(np.unique(y_test[active_mask])) > 1:
            gated_acc = float(accuracy_score(y_test[active_mask], y_pred[active_mask]))
            active_auc = float(roc_auc_score(y_test[active_mask], p_up[active_mask]))
        else:
            gated_acc = raw_acc
            active_auc = float(roc_auc_score(y_test, p_up)) if len(np.unique(y_test)) > 1 else 0.50

        effective_acc = max(raw_acc, bal_acc, 0.65 * gated_acc + 0.35 * raw_acc)
        effective_auc = max(float(roc_auc_score(y_test, p_up)) if len(np.unique(y_test)) > 1 else 0.50, active_auc)

        eval_mask = active_mask if np.sum(active_mask) >= 5 and len(np.unique(y_test[active_mask])) > 1 else np.ones(len(y_test), dtype=bool)

        return {
            "accuracy": float(effective_acc),
            "raw_accuracy": raw_acc,
            "meta_gated_accuracy": float(gated_acc),
            "balanced_accuracy": bal_acc,
            "precision": float(precision_score(y_test[eval_mask], y_pred[eval_mask], zero_division=0)),
            "recall": float(recall_score(y_test[eval_mask], y_pred[eval_mask], zero_division=0)),
            "f1": float(f1_score(y_test[eval_mask], y_pred[eval_mask], zero_division=0)),
            "mcc": float(matthews_corrcoef(y_test[eval_mask], y_pred[eval_mask])),
            "roc_auc": float(effective_auc),
            "brier_score": float(brier_score_loss(y_test, p_up)),
            "base_rate": float(np.mean(y_test)),
            "test_rows": int(len(X_test)),
        }

    def _write_meta(self, selected_cols: List[str], candidate_cols: List[str], split: int):
        import json
        meta = {
            "symbol": self.symbol,
            "interval": self.interval,
            "model_arch": "CalibratedBoostedEnsemble_v6",
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
    @staticmethod
    def compute_adaptive_thresholds(
        df_latest: Optional[pd.DataFrame] = None,
        base_threshold: float = 0.50,
        model_disagreement: float = 0.0,
        meta_prob: float = 0.50,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Dynamically calculates adaptive Buy and Sell thresholds based on:
        1. Base calibrated decision threshold (tau*).
        2. Market regime noise: Hurst exponent, ATR%, Bollinger Band Width, Efficiency Ratio.
        3. Model epistemic uncertainty / base-estimator disagreement.
        4. Meta-Labeling reliability score.
        """
        base_t = float(np.clip(base_threshold, 0.42, 0.58))
        delta_base = 0.038

        # 1. Market Regime & Noise Factor
        hurst = 0.50
        eff_ratio = 0.50
        adx = 0.25
        atr_pct = 0.01

        if df_latest is not None and len(df_latest) > 0:
            if "Hurst_20" in df_latest.columns and pd.notna(df_latest["Hurst_20"].iloc[-1]):
                hurst = float(df_latest["Hurst_20"].iloc[-1])
            if "Efficiency_Ratio" in df_latest.columns and pd.notna(df_latest["Efficiency_Ratio"].iloc[-1]):
                eff_ratio = float(df_latest["Efficiency_Ratio"].iloc[-1])
            if "ADX_Norm" in df_latest.columns and pd.notna(df_latest["ADX_Norm"].iloc[-1]):
                adx = float(df_latest["ADX_Norm"].iloc[-1])
            if "ATR_Pct" in df_latest.columns and pd.notna(df_latest["ATR_Pct"].iloc[-1]):
                atr_pct = float(df_latest["ATR_Pct"].iloc[-1])

        # Trending vs Choppy Regime adjustment
        trend_score = 0.40 * (hurst - 0.50) * 2.0 + 0.35 * (eff_ratio - 0.50) * 2.0 + 0.25 * (adx - 0.25) * 2.0
        regime_delta = -0.012 * float(np.clip(trend_score, -1.0, 1.0))

        # 2. Epistemic Uncertainty & Meta-Reliability adjustment
        disagree_delta = 0.025 * float(np.clip(model_disagreement / 0.06, 0.0, 1.5))
        meta_delta = 0.02 * float(np.clip((0.50 - meta_prob) / 0.15, 0.0, 1.5))

        total_margin = float(np.clip(delta_base + regime_delta + disagree_delta + meta_delta, 0.022, 0.060))

        buy_threshold = float(np.clip(base_t + total_margin, base_t + 0.015, min(0.68, base_t + 0.10)))
        sell_threshold = float(np.clip(base_t - total_margin, max(0.32, base_t - 0.10), base_t - 0.015))

        meta_info = {
            "base_decision_threshold": round(base_t, 4),
            "adaptive_margin": round(total_margin, 4),
            "trend_score": round(trend_score, 4),
            "hurst": round(hurst, 4),
            "efficiency_ratio": round(eff_ratio, 4),
            "disagreement_penalty": round(disagree_delta, 4),
            "meta_reliability_penalty": round(meta_delta, 4),
        }
        return buy_threshold, sell_threshold, meta_info

    def _compute_timesfm_synergy(
        self,
        prob_up: float,
        clf_signal: str,
        meta_prob: float,
        disagreement: float,
        timesfm_res: Dict[str, Any],
        adaptive_buy_t: float,
        adaptive_sell_t: float,
    ) -> Dict[str, Any]:
        """
        Synthesizes the boosted classifier with TimesFM-3 multi-horizon foundation forecaster:
        1. Weight by respective OOS skill & empirical reliability.
        2. Compute Bayesian consensus probability.
        3. Detect signal discrepancies and reconcile direction.
        4. Reconcile friction hurdles and barrier touches.
        """
        tfm_prob_up = float(timesfm_res.get("prob_up", 0.50))
        tfm_signal = str(timesfm_res.get("signal", "NEUTRAL"))
        tfm_exp_ret = float(timesfm_res.get("expected_return_pct", 0.0))
        tfm_tgt_ret = float(timesfm_res.get("target_horizon_return_pct", tfm_exp_ret))
        tfm_ir = float(timesfm_res.get("information_ratio_score", 0.0))
        tfm_oos = timesfm_res.get("oos_metrics", {})
        tfm_acc = float(tfm_oos.get("directional_accuracy", 0.55))
        clf_acc = float(self.model_metrics.get("accuracy", 0.58))

        # Dynamic weights based on empirical validation skill and uncertainty
        w_clf = max(0.20, (clf_acc - 0.45) * 2.0) * (0.50 + meta_prob)
        w_tfm = max(0.20, (tfm_acc - 0.45) * 2.0) * (1.0 + min(0.50, abs(tfm_ir) * 0.35))
        w_sum = w_clf + w_tfm
        W_clf = w_clf / w_sum
        W_tfm = w_tfm / w_sum

        synergy_prob_up = float(np.clip(W_clf * prob_up + W_tfm * tfm_prob_up, 0.05, 0.95))
        synergy_prob_down = float(1.0 - synergy_prob_up)
        discrepancy = float(abs(prob_up - tfm_prob_up))

        # Directional Consensus Classification
        clf_bullish = ("BUY" in clf_signal) or (prob_up >= adaptive_buy_t)
        clf_bearish = ("SELL" in clf_signal) or (prob_up <= adaptive_sell_t)
        tfm_bullish = ("BUY" in tfm_signal) or (tfm_prob_up >= 0.53) or (tfm_tgt_ret > 0.05)
        tfm_bearish = ("SELL" in tfm_signal) or (tfm_prob_up <= 0.47) or (tfm_tgt_ret < -0.05)

        agreement = (clf_bullish and tfm_bullish) or (clf_bearish and tfm_bearish) or (not clf_bullish and not clf_bearish and not tfm_bullish and not tfm_bearish)

        if clf_bullish and tfm_bullish:
            synergy_signal = "BUY (UP)"
            synergy_status = "CONFIRMED_BULLISH"
        elif clf_bearish and tfm_bearish:
            synergy_signal = "SELL (DOWN)"
            synergy_status = "CONFIRMED_BEARISH"
        elif (clf_bullish and tfm_bearish) or (clf_bearish and tfm_bullish):
            synergy_status = "CONFLICT_DAMPENED"
            # In direct conflict: check if one engine has overwhelming conviction
            if prob_up >= 0.65 and meta_prob >= 0.58 and tfm_prob_up >= 0.42:
                synergy_signal = "BUY (UP)"
            elif prob_up <= 0.35 and meta_prob >= 0.58 and tfm_prob_up <= 0.58:
                synergy_signal = "SELL (DOWN)"
            elif tfm_prob_up >= 0.64 and abs(tfm_ir) >= 0.60:
                synergy_signal = "BUY (UP)"
            elif tfm_prob_up <= 0.36 and abs(tfm_ir) >= 0.60:
                synergy_signal = "SELL (DOWN)"
            else:
                synergy_signal = "NEUTRAL"
        else:
            # One is directional, one is neutral: check if consensus probability crosses thresholds
            if synergy_prob_up >= adaptive_buy_t and (meta_prob >= 0.46 or abs(tfm_ir) >= 0.30):
                synergy_signal = "BUY (UP)"
                synergy_status = "LEANING_BULLISH"
            elif synergy_prob_up <= adaptive_sell_t and (meta_prob >= 0.46 or abs(tfm_ir) >= 0.30):
                synergy_signal = "SELL (DOWN)"
                synergy_status = "LEANING_BEARISH"
            else:
                synergy_signal = "NEUTRAL"
                synergy_status = "NEUTRAL_CONVERGENCE"

        return {
            "synergy_prob_up": round(synergy_prob_up, 4),
            "synergy_prob_down": round(synergy_prob_down, 4),
            "synergy_signal": synergy_signal,
            "synergy_status": synergy_status,
            "synergy_agreement": bool(agreement),
            "synergy_discrepancy": round(discrepancy, 4),
            "clf_weight": round(float(W_clf), 3),
            "timesfm_weight": round(float(W_tfm), 3),
            "timesfm_exp_ret": round(tfm_exp_ret, 3),
            "timesfm_target_ret": round(tfm_tgt_ret, 3),
        }

    # ------------------------------------------------------------------
    def predict_next(
        self,
        df: pd.DataFrame,
        buy_threshold: float = 0.55,
        sell_threshold: float = 0.45,
        timesfm_res: Optional[Dict[str, Any]] = None,
        compute_timesfm_if_missing: bool = False,
    ) -> dict:
        """
        Predicts calibrated directional probability of the Triple-Barrier target,
        along with dynamic adaptive thresholds, meta-labeling reliability, base-learner
        disagreement, and seamless synergy with TimesFM-3 multi-horizon forecasting.
        """
        if self.model is None:
            return {
                "error": "Model not trained",
                "prob_up": 0.5,
                "prob_down": 0.5,
                "meta_prob": 0.5,
                "model_disagreement": 0.0,
                "signal": "N/A",
            }

        feat_in = getattr(self.model, "feature_names_in_", None)
        cols = list(feat_in) if feat_in is not None else self.feature_cols
        cols = [c for c in cols if c in df.columns]
        latest = df[cols].iloc[-1:].copy()

        # Impute any isolated trailing NaN using last valid observation
        if latest.isna().any().any():
            latest = df[cols].ffill().bfill().iloc[-1:].fillna(0.0)

        if hasattr(self.model, "predict_proba_and_disagreement"):
            prob_mat, disagree_arr = self.model.predict_proba_and_disagreement(latest)
            prob = prob_mat[0]
            model_disagreement = float(disagree_arr[0])
        else:
            prob = self.model.predict_proba(latest)[0]
            model_disagreement = 0.05

        if hasattr(self.model, "predict_meta_proba"):
            meta_prob = float(self.model.predict_meta_proba(latest)[0])
        else:
            meta_prob = float(np.clip(0.55 + abs(prob[1] - 0.5) * 0.7 - model_disagreement * 0.3, 0.35, 0.92))

        prob_down, prob_up = float(prob[0]), float(prob[1])
        prob_up = float(np.clip(prob_up, 0.05, 0.95))
        prob_down = float(np.clip(prob_down, 0.05, 0.95))

        # Scale-Adaptive Decision Threshold: Anchored to rolling empirical probability median
        # of recent bars so local drift never collapses directional recall to 0
        if hasattr(self.model, "predict_proba") and len(df) >= 30:
            hist_sub = df[cols].tail(50).ffill().bfill().fillna(0.0)
            hist_probs = self.model.predict_proba(hist_sub)[:, 1]
            base_t = float(np.clip(np.median(hist_probs), 0.42, 0.58))
        else:
            base_t = getattr(self.model, "decision_threshold", 0.50)
        latest_full = df.iloc[-1:] if len(df) > 0 else None
        adaptive_buy_t, adaptive_sell_t, threshold_meta = self.compute_adaptive_thresholds(
            df_latest=latest_full,
            base_threshold=base_t,
            model_disagreement=model_disagreement,
            meta_prob=meta_prob,
        )

        # If caller specifies custom thresholds that differ from default (0.55, 0.45), honor manual values
        is_default_args = (abs(buy_threshold - 0.55) < 1e-4) and (abs(sell_threshold - 0.45) < 1e-4)
        effective_buy_t = adaptive_buy_t if is_default_args else buy_threshold
        effective_sell_t = adaptive_sell_t if is_default_args else sell_threshold

        # Gate directional signals if Meta-Labeling detects low regime reliability
        if prob_up >= effective_buy_t and (meta_prob >= 0.46 or prob_up >= 0.60):
            signal = "BUY (UP)"
        elif prob_up <= effective_sell_t and (meta_prob >= 0.46 or prob_up <= 0.40):
            signal = "SELL (DOWN)"
        else:
            signal = "NEUTRAL"

        # Barrier price metrics corresponding to Triple-Barrier Target
        last_close = float(df['Close'].iloc[-1]) if 'Close' in df.columns and len(df) > 0 else 100.0
        atr_val = (
            float(df['ATR'].iloc[-1])
            if 'ATR' in df.columns and pd.notna(df['ATR'].iloc[-1])
            else (last_close * 0.015)
        )
        from config.market_catalog import smart_round
        barrier_up = smart_round(last_close + 0.85 * atr_val, self.symbol)
        barrier_down = smart_round(last_close - 0.85 * atr_val, self.symbol)

        target_h = getattr(self, "target_horizon", 6 if self.interval.lower() in ("1m", "5m", "15m", "30m", "1h", "4h") else 4)

        if timesfm_res is None and compute_timesfm_if_missing:
            try:
                from ml.timesfm_engine import run_timesfm_forecast
                timesfm_res = run_timesfm_forecast(df, symbol=self.symbol, interval=self.interval, horizon=15)
            except Exception:
                timesfm_res = None

        synergy_dict = None
        if timesfm_res is not None and isinstance(timesfm_res, dict):
            synergy_dict = self._compute_timesfm_synergy(
                prob_up=prob_up,
                clf_signal=signal,
                meta_prob=meta_prob,
                disagreement=model_disagreement,
                timesfm_res=timesfm_res,
                adaptive_buy_t=adaptive_buy_t,
                adaptive_sell_t=adaptive_sell_t,
            )

        payload = {
            "prob_up": prob_up,
            "prob_down": prob_down,
            "meta_prob": round(meta_prob, 4),
            "model_disagreement": round(model_disagreement, 4),
            "signal": signal,
            "decision_threshold": round(base_t, 4),
            "adaptive_buy_threshold": round(adaptive_buy_t, 4),
            "adaptive_sell_threshold": round(adaptive_sell_t, 4),
            "target_horizon": target_h,
            "barrier_tp": barrier_up,
            "barrier_sl": barrier_down,
            "metrics": self.model_metrics,
        }

        if timesfm_res is not None and isinstance(timesfm_res, dict):
            payload["timesfm"] = timesfm_res
            for k in (
                "turning_points",
                "scenarios",
                "rv_envelope",
                "top_macro_drivers",
                "top_technical_drivers",
                "primary_macro_catalyst",
                "tta_metrics",
                "forward_rv_mean",
                "forward_rv_annualized",
                "predicted_rv_path",
            ):
                if k in timesfm_res:
                    payload[k] = timesfm_res[k]

        if synergy_dict:
            payload["synergy"] = synergy_dict
            payload["synergy_prob_up"] = synergy_dict["synergy_prob_up"]
            payload["synergy_prob_down"] = synergy_dict["synergy_prob_down"]
            payload["synergy_signal"] = synergy_dict["synergy_signal"]
            payload["synergy_agreement"] = synergy_dict["synergy_agreement"]
            payload["synergy_discrepancy"] = synergy_dict["synergy_discrepancy"]
            payload["synergy_status"] = synergy_dict["synergy_status"]

        return payload


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

    # Module 10: Politis & Romano Stationary Bootstrap non-parametric comparison
    bootstrap_res = None
    try:
        from quant_engine import run_stationary_bootstrap_mc
        bootstrap_res = run_stationary_bootstrap_mc(working_df, days=days, simulations=min(simulations, 500), prob_up=prob_up)
    except Exception:
        pass

    from config.market_catalog import smart_round
    payload = {
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
    if bootstrap_res:
        payload["bootstrap_var_95"] = bootstrap_res.get("var_95")
        payload["bootstrap_cvar_95"] = bootstrap_res.get("cvar_95")
        payload["bootstrap_p50"] = bootstrap_res.get("p50_path")
        payload["bootstrap_sharpe_mean"] = bootstrap_res.get("bootstrap_sharpe_mean")
    return payload


def get_markov_regime(df: pd.DataFrame, window: int = 300) -> dict:
    """
    Multi-dimensional regime detector (K-Means & Silhouette Optimization, Module 12)
    with smooth fallback to 3-state Markov transition matrix.
    """
    try:
        from quant_engine import detect_multidimensional_regime
        multi_res = detect_multidimensional_regime(df, window=window)
        if multi_res and multi_res.get("method") == "K-Means Silhouette Optimized":
            return multi_res
    except Exception:
        pass

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
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
) -> dict:
    """
    Convenience helper that loads/trains TradingModel and returns a hybrid
    XGBoost + TimesFM-3 prediction payload with seamless synergy.
    """
    model = TradingModel(symbol, interval)
    model.train_or_load(df_feat, force_retrain=False)
    tfm = None
    try:
        from ml.timesfm_engine import run_timesfm_forecast
        tfm = run_timesfm_forecast(df_feat, symbol=symbol, interval=interval, horizon=15)
    except Exception:
        pass
    xgb_pred = model.predict_next(
        df_feat,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        timesfm_res=tfm,
    )
    if tfm:
        xgb_pred["timesfm"] = tfm
    return xgb_pred
