import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import os
import warnings

# Suppress sklearn warnings if any
warnings.filterwarnings('ignore')

MODELS_DIR = "models"

class TradingModel:
    def __init__(self, symbol: str, interval: str):
        self.symbol = symbol
        self.interval = interval
        self.safe_symbol = symbol.replace("=", "_").replace("^", "_")
        self.model_path = os.path.join(MODELS_DIR, f"{self.safe_symbol}_{self.interval}_xgb.pkl")
        self.model = None
        self.feature_cols = ['RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'BB_Mid', 'ATR', 'SMA_20', 'SMA_50', 'Return']
        
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)

    def train_or_load(self, df: pd.DataFrame, force_retrain=False):
        """
        Trains a new model or loads an existing one.
        """
        # Drop rows with NaN (indicators take time to compute)
        train_df = df.dropna(subset=self.feature_cols + ['Target'])
        
        # We don't train on the very last row because 'Target' is shifted and will be NaN for it,
        # or we just dropped it above.
        
        X = train_df[self.feature_cols]
        y = train_df['Target']

        if not force_retrain and os.path.exists(self.model_path):
            print(f"[{self.symbol}] Loading existing model...")
            self.model = joblib.load(self.model_path)
        else:
            print(f"[{self.symbol}] Training new model on {len(X)} data points...")
            # XGBoost is highly effective for tabular financial data
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
            self.model.fit(X, y)
            joblib.dump(self.model, self.model_path)
            print(f"[{self.symbol}] Model saved to {self.model_path}")

    def predict_next(self, df: pd.DataFrame) -> dict:
        """
        Predicts the probability of the next candle going UP.
        """
        if self.model is None:
            return {"error": "Model not trained"}
            
        # Get the very last row to predict the NEXT candle
        latest = df[self.feature_cols].iloc[-1:]
        
        # If indicators are NaN on the last row (e.g., not enough data), return 50/50
        if latest.isna().any().any():
            return {"prob_up": 0.5, "prob_down": 0.5, "signal": "INSUFFICIENT DATA"}
            
        prob = self.model.predict_proba(latest)[0]
        prob_down, prob_up = prob[0], prob[1]
        
        signal = "BUY (UP)" if prob_up > 0.55 else "SELL (DOWN)" if prob_up < 0.45 else "NEUTRAL"
        
        return {
            "prob_up": float(prob_up),
            "prob_down": float(prob_down),
            "signal": signal
        }

def run_monte_carlo(df: pd.DataFrame, days=30, simulations=1000):
    """
    Runs a Monte Carlo simulation based on historical returns
    to calculate Value at Risk (VaR) and projected ranges.
    """
    if len(df) < 2:
        return None
        
    returns = df['Return'].dropna().values
    if len(returns) == 0:
        return None
        
    mu = np.mean(returns)
    std = np.std(returns)
    last_price = df['Close'].iloc[-1]
    
    simulation_results = []
    
    for _ in range(simulations):
        # Random normal distribution of returns
        sim_returns = np.random.normal(mu, std, days)
        # Cumulative product to get price path
        price_path = last_price * np.exp(np.cumsum(sim_returns))
        simulation_results.append(price_path)
        
    sim_df = pd.DataFrame(simulation_results).T
    
    # Calculate VaR (95% confidence worst case) at end of period
    final_prices = sim_df.iloc[-1]
    var_95 = np.percentile(final_prices, 5)
    
    return {
        "mean_expected": float(np.mean(final_prices)),
        "var_95": float(var_95),
        "paths": sim_df
    }

def get_markov_regime(df: pd.DataFrame):
    """
    Simple 2-state Markov chain to detect if market is trending or choppy.
    State 0: Choppy (ATR is high, returns are mean reverting)
    State 1: Trending (Consistent direction)
    For simplicity, we'll use a momentum threshold.
    """
    if len(df) < 5:
        return "Unknown"
        
    recent = df.tail(10)
    up_days = len(recent[recent['Return'] > 0])
    
    if up_days >= 7:
        return "BULL TREND"
    elif up_days <= 3:
        return "BEAR TREND"
    else:
        return "CHOPPY / RANGE"
