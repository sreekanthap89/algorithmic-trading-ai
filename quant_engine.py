"""
=============================================================
 INSTITUTIONAL QUANT ENGINE (TRADING APP)
 Inspired by M7 Step 11 & BlackRock Aladdin Risk Infrastructure
=============================================================
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor

def quantile_regression_uncertainty(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Fits continuous quantile regressors (tau = 0.10, 0.50, 0.90) via Gradient Boosting
    to isolate Epistemic Uncertainty (confidence interval width q_90 - q_10)
    from Aleatoric market risk.
    """
    valid_df = df.dropna(subset=feature_cols + ['Target']).copy()
    if len(valid_df) < 30:
        last_price = float(df['Close'].iloc[-1]) if len(df) > 0 else 100.0
        return {
            "q10_price": last_price * 0.95,
            "q50_price": last_price,
            "q90_price": last_price * 1.05,
            "epistemic_uncertainty": float(last_price * 0.10),
            "uncertainty_level": "HIGH (INSUFFICIENT DATA)",
            "quantile_score": 0.5
        }

    X = valid_df[feature_cols]
    # Predict percentage change to next candle
    y = valid_df['Close'].pct_change().shift(-1).dropna()
    X = X.iloc[:-1] # Match shifted target length

    if len(X) < 25:
        last_price = float(df['Close'].iloc[-1])
        return {
            "q10_price": last_price * 0.95,
            "q50_price": last_price,
            "q90_price": last_price * 1.05,
            "epistemic_uncertainty": float(last_price * 0.10),
            "uncertainty_level": "MODERATE",
            "quantile_score": 0.5
        }

    # Latest row for prediction
    latest_X = df[feature_cols].iloc[-1:]

    try:
        # Quantile 0.10 Regressor
        q10_model = GradientBoostingRegressor(loss="quantile", alpha=0.10, n_estimators=40, max_depth=3, random_state=42)
        q10_model.fit(X, y)
        ret_10 = float(q10_model.predict(latest_X)[0])

        # Quantile 0.50 Regressor (Median)
        q50_model = GradientBoostingRegressor(loss="quantile", alpha=0.50, n_estimators=40, max_depth=3, random_state=42)
        q50_model.fit(X, y)
        ret_50 = float(q50_model.predict(latest_X)[0])

        # Quantile 0.90 Regressor
        q90_model = GradientBoostingRegressor(loss="quantile", alpha=0.90, n_estimators=40, max_depth=3, random_state=42)
        q90_model.fit(X, y)
        ret_90 = float(q90_model.predict(latest_X)[0])

    except Exception:
        ret_10, ret_50, ret_90 = -0.02, 0.0, 0.02

    last_price = float(df['Close'].iloc[-1])
    q10_price = last_price * (1.0 + ret_10)
    q50_price = last_price * (1.0 + ret_50)
    q90_price = last_price * (1.0 + ret_90)

    epistemic_uncertainty = abs(q90_price - q10_price)
    uncertainty_ratio = epistemic_uncertainty / (last_price + 1e-9)

    if uncertainty_ratio > 0.08:
        level = "HIGH (WIDE SPREAD)"
    elif uncertainty_ratio > 0.04:
        level = "MODERATE"
    else:
        level = "LOW (HIGH CONFIDENCE)"

    # Score based on median return prediction vs uncertainty ratio
    quantile_score = float(np.clip(0.5 + (ret_50 * 10.0), 0.05, 0.95))

    return {
        "q10_price": float(q10_price),
        "q50_price": float(q50_price),
        "q90_price": float(q90_price),
        "ret_10_pct": float(ret_10 * 100.0),
        "ret_50_pct": float(ret_50 * 100.0),
        "ret_90_pct": float(ret_90 * 100.0),
        "epistemic_uncertainty": float(epistemic_uncertainty),
        "uncertainty_level": level,
        "quantile_score": quantile_score
    }

def run_jump_diffusion_mc(df: pd.DataFrame, days: int = 20, simulations: int = 1000) -> dict:
    """
    Runs Merton Stochastic Jump-Diffusion Monte Carlo Simulation:
      dS_t = mu * S_t * dt + sigma * S_t * dW_t + J_t * dN_t
    where J_t ~ Normal(mu_j, sigma_j) and dN_t ~ Poisson(lambda_j * dt).
    Calculates 95% Value-at-Risk (VaR) and 95% Expected Shortfall (CVaR).
    """
    if len(df) < 5:
        return None

    returns = df['Return'].dropna().values
    if len(returns) < 5:
        return None

    last_price = float(df['Close'].iloc[-1])
    mu = float(np.mean(returns))
    sigma = float(np.std(returns))

    # Detect discrete jump parameters (extreme returns exceeding 2 sigma)
    jump_threshold = 2.0 * sigma if sigma > 0 else 0.01
    jumps = returns[np.abs(returns) > jump_threshold]
    
    lambda_j = float(len(jumps) / len(returns)) if len(returns) > 0 else 0.05
    mu_j = float(np.mean(jumps)) if len(jumps) > 0 else 0.0
    sigma_j = float(np.std(jumps)) if len(jumps) > 0 else (sigma * 1.5)

    simulation_results = []
    dt = 1.0

    for _ in range(simulations):
        price_path = [last_price]
        cur_price = last_price

        for _ in range(days):
            # Geometric Brownian Motion component
            dW = np.random.normal(0, 1)
            gbm_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * dW

            # Poisson jump component
            n_jumps = np.random.poisson(lambda_j * dt)
            jump_ret = sum(np.random.normal(mu_j, sigma_j) for _ in range(n_jumps)) if n_jumps > 0 else 0.0

            total_ret = gbm_ret + jump_ret
            cur_price = cur_price * np.exp(total_ret)
            price_path.append(cur_price)

        simulation_results.append(price_path[1:])

    sim_matrix = np.array(simulation_results).T  # Shape: (days, simulations)
    sim_df = pd.DataFrame(sim_matrix)

    final_prices = sim_matrix[-1, :]
    var_95 = float(np.percentile(final_prices, 5))
    
    # Expected Shortfall (CVaR 95%) - mean of prices below 5th percentile
    tail_prices = final_prices[final_prices <= var_95]
    cvar_95 = float(np.mean(tail_prices)) if len(tail_prices) > 0 else var_95

    return {
        "mean_expected": float(np.mean(final_prices)),
        "var_95": float(var_95),
        "cvar_95": float(cvar_95),
        "lambda_jump": float(lambda_j),
        "paths": sim_df
    }

def calculate_rolling_ic(predictions_hist: list, actuals_hist: list) -> float:
    """
    Evaluates Information Coefficient (IC) using Spearman rank correlation
    between model predictions and actual realized returns (Grinold & Kahn).
    """
    if len(predictions_hist) < 5 or len(actuals_hist) < 5:
        return 0.10  # Default initial baseline IC

    try:
        ic, _ = spearmanr(predictions_hist, actuals_hist)
        if np.isnan(ic):
            ic = 0.05
    except Exception:
        ic = 0.05

    return float(np.clip(ic, -1.0, 1.0))

def fuse_signals(component_scores: dict) -> float:
    """
    Step 11 Master Signal Fusion across Steps 1 to 10:
    Component weights (sum to 1.0):
      data_quality:       0.05
      volume_profile:     0.10
      var_risk:           0.05
      monte_carlo:        0.10
      markov_regime:      0.10
      multi_alpha:        0.15
      microstructure:     0.05
      deep_learning:      0.10
      stacking_ensemble:  0.20
      quantum_signals:    0.10
    """
    weights = {
        "data_quality": 0.05,
        "volume_profile": 0.10,
        "var_risk": 0.05,
        "monte_carlo": 0.10,
        "markov_regime": 0.10,
        "multi_alpha": 0.15,
        "microstructure": 0.05,
        "deep_learning": 0.10,
        "stacking_ensemble": 0.20,
        "quantum_signals": 0.10
    }

    fused_conviction = sum(weights[k] * component_scores.get(k, 0.5) for k in weights)
    return float(np.clip(fused_conviction, 0.0, 1.0))

def kelly_position_sizing(win_rate: float, reward_risk_ratio: float, total_capital: float = 10000.0) -> dict:
    """
    Calculates position sizing via Half-Kelly Criterion:
      f* = (p * b - q) / b
    where p = win_rate, q = 1 - p, b = reward_risk_ratio.
    Rules:
      - Half-Kelly: position_size = 0.5 * f* * total_capital
      - Hard cap: max 25% of capital
      - Min threshold: only trade if f* > 0.02
    """
    p = float(np.clip(win_rate, 0.01, 0.99))
    q = 1.0 - p
    b = float(max(0.1, reward_risk_ratio))

    full_kelly = (p * b - q) / b
    
    if full_kelly <= 0.02:
        return {"fraction": 0.0, "position_usd": 0.0, "kelly_status": "BELOW THRESHOLD (NO EDGE)"}

    half_kelly = 0.5 * full_kelly
    capped_fraction = float(np.clip(half_kelly, 0.0, 0.25))
    position_usd = round(capped_fraction * total_capital, 2)

    return {
        "full_kelly": round(full_kelly, 4),
        "half_kelly": round(half_kelly, 4),
        "capped_fraction": round(capped_fraction, 4),
        "position_usd": position_usd,
        "kelly_status": "OPTIMAL POSITION ALLOCATED"
    }

def generate_master_trade_signal(
    df: pd.DataFrame, 
    component_scores: dict, 
    conviction_threshold: float = 0.75, 
    total_capital: float = 10000.0
) -> dict:
    """
    Generates a master trade decision combining signal fusion, dynamic stops, and Kelly sizing.
    """
    if df is None or len(df) == 0:
        return {"action": "HOLD", "conviction": 0.5, "reason": "No data"}

    last_price = float(df['Close'].iloc[-1])
    atr = float(df['ATR'].iloc[-1]) if 'ATR' in df.columns else (last_price * 0.02)

    conviction = fuse_signals(component_scores)
    
    # Calculate entry, SL, TP
    reward_risk_ratio = 2.0
    stop_dist = 2.0 * atr
    tp_dist = reward_risk_ratio * stop_dist

    if conviction >= conviction_threshold:
        action = "BUY"
        stop_loss = round(last_price - stop_dist, 2)
        take_profit = round(last_price + tp_dist, 2)
    elif conviction <= (1.0 - conviction_threshold):
        action = "SELL"
        stop_loss = round(last_price + stop_dist, 2)
        take_profit = round(last_price - tp_dist, 2)
    else:
        action = "HOLD"
        stop_loss = round(last_price - stop_dist, 2)
        take_profit = round(last_price + tp_dist, 2)

    sizing = kelly_position_sizing(conviction, reward_risk_ratio, total_capital)

    return {
        "action": action,
        "conviction": round(conviction, 4),
        "entry": round(last_price, 2),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "reward_risk": reward_risk_ratio,
        "position_size_usd": sizing["position_usd"],
        "kelly_details": sizing
    }

