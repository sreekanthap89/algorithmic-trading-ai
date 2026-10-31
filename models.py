import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import os
import warnings
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

warnings.filterwarnings('ignore')

MODELS_DIR = "models"


class TradingModel:
    def __init__(self, symbol: str, interval: str):
        self.symbol = symbol
        self.interval = interval
        self.safe_symbol = symbol.replace("=", "_").replace("^", "_")
        self.model_path = os.path.join(MODELS_DIR, f"{self.safe_symbol}_{self.interval}_xgb.pkl")
        self.meta_path = os.path.join(MODELS_DIR, f"{self.safe_symbol}_{self.interval}_xgb.meta.json")
        self.model = None
        self.model_metrics: dict = {}
        # Feature set must match what features.add_features produces.
        # Imported lazily to avoid a hard dependency cycle.
        try:
            from features import LIVE_FEATURE_COLS
            self.feature_cols = list(LIVE_FEATURE_COLS)
        except Exception:
            self.feature_cols = [
                'RSI', 'MACD', 'MACD_Diff', 'MACD_Signal',
                'BB_High', 'BB_Low', 'BB_Mid', 'BB_Width', 'BB_pctB',
                'ATR', 'ATR_Pct', 'SMA_20', 'SMA_50', 'SMA_200',
                'EMA_12', 'EMA_26',
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

    def train_or_load(self, df: pd.DataFrame, force_retrain: bool = False,
                      test_frac: float = 0.2) -> dict:
        """
        Trains a new model or loads an existing one.

        When (re)training, the last `test_frac` of the time series is held out
        as an out-of-sample test set so we can measure real predictive skill
        instead of just fitting the whole history (which overfits and tells us
        nothing about forward performance).
        """
        # Use whichever feature columns are actually present in the data.
        cols = [c for c in self.feature_cols if c in df.columns]
        train_df = df.dropna(subset=cols + ['Target'])

        # The ATR-gated target can leave a long neutral band; drop trailing NaNs.
        train_df = train_df.dropna()
        if len(train_df) < 40:
            print(f"[{self.symbol}] Only {len(train_df)} usable rows; keeping existing model.")
            if os.path.exists(self.model_path):
                self._load()
            return self.model_metrics

        X = train_df[cols]
        y = train_df['Target'].astype(float).values

        should_train = force_retrain or not os.path.exists(self.model_path)

        # A model trained on different features is stale/incompatible.
        meta = self._load_json(self.meta_path)
        if os.path.exists(self.model_path) and not force_retrain:
            saved_cols = meta.get('feature_cols', [])
            if saved_cols and saved_cols != cols:
                print(f"[{self.symbol}] Feature set changed → retraining.")
                should_train = True

        if not should_train and os.path.exists(self.model_path):
            print(f"[{self.symbol}] Loading existing model...")
            self._load()
            return self.model_metrics

        print(f"[{self.symbol}] Training new model on {len(X)} rows "
              f"({int((1 - test_frac) * len(X))} train / "
              f"{int(test_frac * len(X))} OOS test)...")

        split = int(len(X) * (1.0 - test_frac))
        n_test = max(10, int(len(X) * test_frac))
        split = len(X) - n_test
        split = max(10, min(split, len(X) - 1))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y[:split], y[split:]

        # Handle class imbalance so the model doesn't just learn the majority class.
        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        scale_pos = (n_neg / n_pos) if n_pos > 0 else 1.0

        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            random_state=42,
            eval_metric='logloss',
            objective='binary:logistic',
        )
        self.model.fit(X_train, y_train)

        # Out-of-sample evaluation (no time-series leakage; test = later data).
        self.model_metrics = self._evaluate(self.model, X_test, y_test)

        # Persist model + metadata (feature set + xgb version for load checks).
        joblib.dump(self.model, self.model_path)
        self._write_meta(cols, split)
        print(f"[{self.symbol}] Saved model. OOS "
              f"acc={self.model_metrics.get('accuracy', 0):.3f} "
              f"auc={self.model_metrics.get('roc_auc', 0):.3f} "
              f"win-rate-class1={self.model_metrics.get('precision', 0):.3f}")
        return self.model_metrics

    # ------------------------------------------------------------------
    def _evaluate(self, model, X_test, y_test) -> dict:
        if len(X_test) == 0 or len(np.unique(y_test)) < 2:
            return {"accuracy": 0.0, "roc_auc": 0.5, "precision": 0.0,
                    "recall": 0.0, "f1": 0.0, "note": "insufficient test data"}
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        # XGBoost column order guarantees column 1 = positive class.
        p_up = proba[:, 1]
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, p_up)),
            "base_rate": float(np.mean(y_test)),
            "test_rows": int(len(X_test)),
        }

    def _write_meta(self, cols, split):
        import json
        meta = {
            "symbol": self.symbol,
            "interval": self.interval,
            "feature_cols": cols,
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
        self.feature_cols = meta.get("feature_cols", self.feature_cols)

    # ------------------------------------------------------------------
    def predict_next(self, df: pd.DataFrame,
                     buy_threshold: float = 0.55,
                     sell_threshold: float = 0.45) -> dict:
        """
        Predicts the probability of the next candle going UP (calibrated, clipped).
        """
        if self.model is None:
            return {"error": "Model not trained", "prob_up": 0.5, "signal": "N/A"}

        # Use the exact feature columns the model was trained on, when known,
        # otherwise fall back to the shared contract.
        cols = list(getattr(self.model, "feature_names_in_", None)) or self.feature_cols
        # Only predict on columns present in both the model and the data.
        cols = [c for c in cols if c in df.columns]
        latest = df[cols].iloc[-1:]

        if latest.isna().any().any():
            return {"prob_up": 0.5, "prob_down": 0.5,
                    "signal": "INSUFFICIENT DATA"}

        prob = self.model.predict_proba(latest)[0]
        prob_down, prob_up = float(prob[0]), float(prob[1])
        # Clip to avoid over-confident extremes (probability calibration guard).
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
    2. Bayesian drift conditioning incorporating AI directional probability (prob_up).
    3. Variance-reduced Antithetic Variates Monte Carlo simulation with deterministic
       bar-timestamp seeding for empirical path distributions and VaR computation.
    """
    if df is None or len(df) < 2:
        return None

    working_df = df
    returns = working_df['Return'].dropna().values if 'Return' in working_df.columns \
        else working_df['Close'].pct_change().dropna().values
    if len(returns) == 0:
        return None

    mu_hist = float(np.mean(returns))
    std = float(np.std(returns))
    std = std if std > 0 else 1e-6

    # Bayesian drift updating: incorporate AI model directional probability
    # If prob_up > 0.5, expected positive directional edge; if < 0.5, negative edge.
    if prob_up is not None and not np.isnan(prob_up):
        kappa = 1.0 / np.sqrt(max(days, 1))
        mu = mu_hist * 0.2 + (float(prob_up) - 0.5) * std * kappa
    else:
        mu = mu_hist

    last_price = float(working_df['Close'].iloc[-1])
    if np.isnan(last_price) or last_price <= 0:
        return None

    # =========================================================================
    # 1. EXACT CLOSED-FORM GEOMETRIC BROWNIAN MOTION (Zero Variance)
    # Under Itô's Lemma: d ln S_t = (mu - 0.5 * std^2) dt + std * dW_t
    # Quantiles Q_p(t) = S_0 * exp( (mu - 0.5 * std^2)*t + z_p * std * sqrt(t) )
    # =========================================================================
    t_steps = np.arange(1, days + 1, dtype=float)
    drift_term = (mu - 0.5 * (std ** 2)) * t_steps
    diffusion_term = std * np.sqrt(t_steps)

    # Standard normal critical values: Phi^-1(0.85), Phi^-1(0.15), Phi^-1(0.05)
    z_85 = 1.0364333894937898
    z_15 = -1.0364333894937898
    z_05 = -1.6448536269514722

    analytical_median = last_price * np.exp(drift_term)
    analytical_upper85 = last_price * np.exp(drift_term + z_85 * diffusion_term)
    analytical_lower15 = last_price * np.exp(drift_term + z_15 * diffusion_term)
    var_95_analytical = float(last_price * np.exp(drift_term[-1] + z_05 * diffusion_term[-1]))

    # =========================================================================
    # 2. VARIANCE-REDUCED ANTITHETIC SIMULATION (Deterministic Reproducibility)
    # =========================================================================
    if seed is None:
        last_ts = str(working_df.index[-1]) if hasattr(working_df.index, '__getitem__') else "0"
        import hashlib
        seed = int(hashlib.sha256(f"{last_ts}_{days}_{len(working_df)}".encode()).hexdigest()[:8], 16)

    rng = np.random.default_rng(seed)
    half_sims = max(1, simulations // 2)
    half_draws = rng.normal(0.0, 1.0, size=(half_sims, days))
    standard_normals = np.vstack([half_draws, -half_draws])  # exactly 0 empirical mean error

    sim_returns = (mu - 0.5 * (std ** 2)) + std * standard_normals
    prices = last_price * np.exp(np.cumsum(sim_returns, axis=1))
    sim_df = pd.DataFrame(prices)

    final_prices = sim_df.iloc[:, -1]

    return {
        "mean_expected": float(np.mean(final_prices)),
        "var_95": var_95_analytical,
        "std": std,
        "mu": mu,
        "paths": sim_df,
        "analytical_median": [float(x) for x in analytical_median],
        "analytical_upper85": [float(x) for x in analytical_upper85],
        "analytical_lower15": [float(x) for x in analytical_lower15],
    }


def get_markov_regime(df: pd.DataFrame, window: int = 300) -> dict:
    """
    3-state Markov regime detector (Bull / Bear / Choppy).

    Uses only the most recent `window` bars (avoids recomputing the entire
    transition matrix over full history on every refresh) and reports conviction
    as the dominant *next-state* transition probability.
    """
    fallback = {
        "regime": "CHOPPY / RANGE",
        "state_index": 2,
        "transition_matrix": [[0.33, 0.33, 0.34], [0.33, 0.33, 0.34], [0.33, 0.33, 0.34]],
        "state_probs": [0.33, 0.33, 0.34],
        "markov_score": 0.5,
    }
    if df is None or len(df) < 15:
        return fallback

    if 'Return' in df.columns:
        returns = df['Return'].dropna().values
    else:
        returns = df['Close'].pct_change().dropna().values
    returns = returns[-window:]
    if len(returns) < 15:
        return fallback

    std_ret = np.std(returns) if np.std(returns) > 0 else 0.01
    states = []
    for r in returns:
        if r > 0.5 * std_ret:
            states.append(0)   # Bull
        elif r < -0.5 * std_ret:
            states.append(1)   # Bear
        else:
            states.append(2)   # Choppy

    trans_counts = np.zeros((3, 3))
    for i in range(len(states) - 1):
        trans_counts[states[i]][states[i + 1]] += 1

    row_sums = trans_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    p_matrix = trans_counts / row_sums

    current_state = states[-1]
    state_probs = p_matrix[current_state].tolist()

    regime_names = {0: "BULL TREND", 1: "BEAR TREND", 2: "CHOPPY / RANGE"}
    state_label = regime_names[current_state]

    # Conviction = how strongly the model expects to stay in the current regime,
    # a genuine probability rather than an arbitrary weighted sum.
    conviction = max(state_probs)

    return {
        "regime": state_label,
        "state_index": current_state,
        "transition_matrix": p_matrix.tolist(),
        "state_probs": [round(p, 4) for p in state_probs],
        "markov_score": round(float(conviction), 4),
    }
