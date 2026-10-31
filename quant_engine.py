import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor

def quantile_regression_uncertainty(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Fits continuous quantile regressors (tau = 0.10, 0.50, 0.90) via Gradient Boosting
    to isolate epistemic uncertainty (the width of the 90%‑10% prediction interval) from
    aleatoric market risk.  The function now:
      • Uses a strictly lagged target to avoid any forward‑looking leakage.
      • Returns a minimal, well‑documented set of fields.
      • Removes the unused ``quantile_score`` (dead code) and ensures all returned
        numeric values are plain ``float`` types.
    """
    # -------------------------------------------------------------------------
    # 1️⃣  Prepare a clean training set – target is the *next* candle’s return.
    # -------------------------------------------------------------------------
    valid_df = df.dropna(subset=feature_cols + ["Target"], how="any").copy()
    if len(valid_df) < 30:
        last_price = float(df["Close"].iloc[-1]) if len(df) > 0 else 100.0
        return {
            "q10_price": float(last_price * 0.95),
            "q50_price": float(last_price),
            "q90_price": float(last_price * 1.05),
            "epistemic_uncertainty": float(last_price * 0.10),
            "uncertainty_level": "HIGH (INSUFFICIENT DATA)",
        }

    # Features matrix – strictly lagged so that no future information leaks.
    X = valid_df[feature_cols]
    # Target = % change of Close on the following bar (shifted by -1).
    y = valid_df["Close"].pct_change().shift(-1).dropna()
    # Align X and y – drop the last X row that no longer has a matching target.
    X = X.iloc[:-1]
    y = y.iloc[:-1]

    if len(X) < 25:
        last_price = float(df["Close"].iloc[-1])
        return {
            "q10_price": float(last_price * 0.95),
            "q50_price": float(last_price),
            "q90_price": float(last_price * 1.05),
            "epistemic_uncertainty": float(last_price * 0.10),
            "uncertainty_level": "MODERATE",
        }

    # -------------------------------------------------------------------------
    # 2️⃣  Fit three quantile regressors (10th, 50th, 90th percentiles).
    # -------------------------------------------------------------------------
    latest_X = df[feature_cols].iloc[-1:]

    try:
        # 10 % quantile regressor
        q10_model = GradientBoostingRegressor(loss="quantile", alpha=0.10,
                                              n_estimators=40, max_depth=3, random_state=42)
        q10_model.fit(X, y)
        ret_10 = float(q10_model.predict(latest_X)[0])

        # 50 % (median) quantile regressor
        q50_model = GradientBoostingRegressor(loss="quantile", alpha=0.50,
                                              n_estimators=40, max_depth=3, random_state=42)
        q50_model.fit(X, y)
        ret_50 = float(q50_model.predict(latest_X)[0])

        # 90 % quantile regressor
        q90_model = GradientBoostingRegressor(loss="quantile", alpha=0.90,
                                              n_estimators=40, max_depth=3, random_state=42)
        q90_model.fit(X, y)
        ret_90 = float(q90_model.predict(latest_X)[0])

    except Exception:
        # Fallback when the model fails to fit – use mild defaults.
        ret_10, ret_50, ret_90 = -0.02, 0.0, 0.02

    last_price = float(df["Close"].iloc[-1])
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

    # Return a clean, type‑stable payload.
    return {
        "q10_price": float(q10_price),
        "q50_price": float(q50_price),
        "q90_price": float(q90_price),
        "ret_10_pct": float(ret_10 * 100.0),
        "ret_50_pct": float(ret_50 * 100.0),
        "ret_90_pct": float(ret_90 * 100.0),
        "epistemic_uncertainty": float(epistemic_uncertainty),
        "uncertainty_level": level,
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
    Computes the Information Coefficient (IC) as the Spearman rank correlation
    between model predictions and realized returns (Grinold & Kahn).
    Returns ``np.nan`` when insufficient data prevents a reliable calculation.
    """
    if len(predictions_hist) < 5 or len(actuals_hist) < 5:
        return np.nan  # Not enough observations for a stable IC

    try:
        ic, _ = spearmanr(predictions_hist, actuals_hist)
        if np.isnan(ic):
            ic = 0.0
    except Exception:
        ic = 0.0

    return float(np.clip(ic, -1.0, 1.0))

def fuse_signals(component_scores: dict, weights: Optional[Dict[str, float]] = None) -> float:
    """
    Step 11 Master Signal Fusion across Steps 1‑10.
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
    # Default weight set – kept in a separate constant for discoverability
    default_weights: Dict[str, float] = {
        "data_quality": 0.05,
        "volume_profile": 0.10,
        "var_risk": 0.05,
        "monte_carlo": 0.10,
        "markov_regime": 0.10,
        "multi_alpha": 0.15,
        "microstructure": 0.05,
        "deep_learning": 0.10,
        "stacking_ensemble": 0.20,
        "quantum_signals": 0.10,
    }
    # Allow caller‑provided override
    weights = weights or default_weights

    # Normalise weights to guarantee they sum to 1.0 in case of manual edits
    total = sum(weights.values())
    norm_weights = {k: v / total for k, v in weights.items()}

    fused_conviction = sum(norm_weights[k] * component_scores.get(k, 0.5) for k in norm_weights)
    return float(np.clip(fused_conviction, 0.0, 1.0))

def kelly_position_sizing(win_rate: float, reward_risk_ratio: float, total_capital: float = 10000.0) -> dict:
    """
    Calculates position sizing via Half‑Kelly Criterion:
      f* = (p * b - q) / b
    where p = win_rate, q = 1 - p, b = reward_risk_ratio.
    Rules:
      - Half‑Kelly: position_size = 0.5 * f* * total_capital
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

def get_markov_regime(df: pd.DataFrame, window: int = 300) -> dict:
    """
    3‑state Markov regime detector (Bull / Bear / Choppy).
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

    conviction = max(state_probs)

    return {
        "regime": state_label,
        "state_index": current_state,
        "transition_matrix": p_matrix.tolist(),
        "state_probs": [round(p, 4) for p in state_probs],
        "markov_score": round(float(conviction), 4),
    }