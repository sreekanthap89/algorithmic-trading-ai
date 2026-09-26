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

def get_markov_regime(df: pd.DataFrame) -> dict:
    """
    3-state Markov Transition Chain for market regime detection:
    State 0: Bull Trend
    State 1: Bear Trend
    State 2: Choppy / Range
    
    Computes transition probability matrix P[i][j] from rolling return quantiles
    or hmmlearn if available.
    """
    if df is None or len(df) < 15:
        return {
            "regime": "CHOPPY / RANGE",
            "state_index": 2,
            "transition_matrix": [[0.33, 0.33, 0.34], [0.33, 0.33, 0.34], [0.33, 0.33, 0.34]],
            "state_probs": [0.33, 0.33, 0.34],
            "markov_score": 0.5
        }

    returns = df['Return'].dropna().values if 'Return' in df.columns else df['Close'].pct_change().dropna().values
    
    if len(returns) < 15:
        return {
            "regime": "CHOPPY / RANGE",
            "state_index": 2,
            "transition_matrix": [[0.33, 0.33, 0.34], [0.33, 0.33, 0.34], [0.33, 0.33, 0.34]],
            "state_probs": [0.33, 0.33, 0.34],
            "markov_score": 0.5
        }

    # Map returns to states
    std_ret = np.std(returns) if np.std(returns) > 0 else 0.01
    states = []
    for r in returns:
        if r > 0.5 * std_ret:
            states.append(0)  # Bull
        elif r < -0.5 * std_ret:
            states.append(1)  # Bear
        else:
            states.append(2)  # Chop

    # Build 3x3 Empirical Transition Matrix
    trans_counts = np.zeros((3, 3))
    for i in range(len(states) - 1):
        trans_counts[states[i]][states[i + 1]] += 1

    # Normalize rows
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    p_matrix = trans_counts / row_sums

    current_state = states[-1]
    state_probs = p_matrix[current_state].tolist()

    regime_names = {0: "BULL TREND", 1: "BEAR TREND", 2: "CHOPPY / RANGE"}
    state_label = regime_names[current_state]

    # Calculate conviction score (0.0 to 1.0)
    score = float(np.clip(state_probs[0] * 1.0 + state_probs[2] * 0.5 + state_probs[1] * 0.0, 0.0, 1.0))

    return {
        "regime": state_label,
        "state_index": current_state,
        "transition_matrix": p_matrix.tolist(),
        "state_probs": [round(p, 4) for p in state_probs],
        "markov_score": round(score, 4)
    }

