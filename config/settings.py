"""
Configuration settings for the Trading Application.
Centralized place for all hyperparameters and configuration.
"""

import os
from datetime import timedelta

# ==================== PATH CONFIGURATION ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")
BACKTEST_DIR = os.path.join(BASE_DIR, "backtest_results")

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, METRICS_DIR, BACKTEST_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ==================== DATA CONFIGURATION ====================
DATA_SOURCES = {
    "1d": {"period": "2y", "interval": "1d"},
    "1h": {"period": "1mo", "interval": "1h"},
    "5m": {"period": "5d", "interval": "5m"}
}

# ==================== MODEL CONFIGURATION ====================
MODEL_CONFIG = {
    "base_models": {
        "XGBoost": {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.05,
            "random_state": 42,
            "eval_metric": "logloss"
        },
        "LightGBM": {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.05,
            "random_state": 42,
            "verbose": -1
        },
        "RandomForest": {
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42
        },
        "ExtraTrees": {
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42
        },
        "DeepMLP": {
            "hidden_layer_sizes": (64, 32),
            "max_iter": 200,
            "random_state": 42,
            "early_stopping": True
        }
    },
    "meta_stacker": {
        "C": 1.0,
        "random_state": 42
    },
    "cross_validation_splits": 3,
    "min_training_samples": 40
}

# ==================== ENSEMBLE FUSION WEIGHTS ====================
ENSEMBLE_WEIGHTS = {
    "ml_stacker": 0.45,
    "quantum_engine": 0.25,
    "quantile_regression": 0.15,
    "markov_regime": 0.15
}

# ==================== SIGNAL THRESHOLDS ====================
SIGNAL_THRESHOLDS = {
    "buy_threshold": 0.56,
    "sell_threshold": 0.44,
    "neutral_lower": 0.44,
    "neutral_upper": 0.56,
    "prob_min_clip": 0.02,
    "prob_max_clip": 0.98
}

# ==================== FEATURE CONFIGURATION ====================
# Mirrors features.LIVE_FEATURE_COLS so training and inference agree.
FEATURE_COLS = [
    'RSI', 'MACD', 'MACD_Diff', 'MACD_Signal',
    'BB_High', 'BB_Low', 'BB_Mid', 'BB_Width', 'BB_pctB',
    'ATR', 'ATR_Pct',
    'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26',
    'Dist_SMA20', 'Dist_SMA50', 'Dist_EMA12',
    'Return', 'Log_Return',
    'Return_Vol_10', 'Return_Skew_20', 'Return_Kurt_20', 'Z_Score_Return',
    'Streak_Length', 'VWAP_Dev', 'RSI_Divergence',
    'Hawkes_Intensity', 'Pair_RSI_MACD_Lift',
]

# ==================== QUANTUM SIGNALS CONFIGURATION ====================
QUANTUM_CONFIG = {
    "fft_window": 64,
    "hawkes_alpha": 0.4,
    "hawkes_beta": 0.2,
    "hawkes_lookback": 30,
    "hmm_n_states": 3,
    "maxent_n_moments": 4
}

# ==================== QUANTILE REGRESSION ====================
QUANTILE_CONFIG = {
    "quantiles": [0.10, 0.50, 0.90],
    "n_estimators": 40,
    "max_depth": 3,
    "uncertainty_thresholds": {
        "high": 0.08,
        "moderate": 0.04
    }
}

# ==================== MONTE CARLO CONFIGURATION ====================
MONTE_CARLO_CONFIG = {
    "simulations": 1000,
    "days_ahead": 20,
    "confidence_level": 0.95,
    "jump_threshold_sigma": 2.0
}

# ==================== BACKTESTING CONFIGURATION ====================
BACKTEST_CONFIG = {
    "initial_capital": 100000.0,
    "commission": 0.001,  # 0.1% per trade
    "slippage": 0.0005,   # 0.05% slippage
    "risk_per_trade": 0.02,  # 2% risk per trade
    "max_position_size": 0.1  # Max 10% of capital per position
}

# ==================== LOGGING CONFIGURATION ====================
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": os.path.join(LOGS_DIR, "trading_app.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": os.path.join(LOGS_DIR, "errors.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 3
        }
    },
    "loggers": {
        "": {
            "level": "DEBUG",
            "handlers": ["console", "file", "error_file"]
        }
    }
}

# ==================== METRICS CONFIGURATION ====================
METRICS_CONFIG = {
    "track_predictions": True,
    "track_model_performance": True,
    "track_portfolio": True,
    "save_interval": 100  # Save metrics every N predictions
}

# ==================== API CONFIGURATION ====================
API_CONFIG = {
    "yfinance_max_retries": 3,
    "yfinance_timeout": 30,
    "cache_update_interval": 2  # days before cache needs refresh
}

# ==================== MODEL VERSIONING ====================
MODEL_VERSIONING = {
    "enabled": True,
    "auto_archive": True,
    "max_versions_per_symbol_interval": 5
}
