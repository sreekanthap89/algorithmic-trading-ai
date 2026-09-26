"""
=============================================================
 HESTON STOCHASTIC VOLATILITY & MONTE CARLO SIMULATION
 Step 4 Enhancement (Trading App)
=============================================================
"""

import numpy as np
import pandas as pd

def run_heston_monte_carlo(
    df: pd.DataFrame, 
    days: int = 30, 
    simulations: int = 10000, 
    kappa: float = 2.0, 
    theta: float = 0.04, 
    xi: float = 0.3, 
    rho: float = -0.7
) -> dict:
    """
    Runs a Heston Stochastic Volatility Monte Carlo simulation:
      dS_t = mu * S_t * dt + sqrt(V_t) * S_t * dW_S
      dV_t = kappa * (theta - V_t) * dt + xi * sqrt(V_t) * dW_V
    
    Parameters:
      kappa : Mean reversion speed of variance
      theta : Long-run variance
      xi    : Volatility of volatility
      rho   : Correlation between asset price asset return and variance shocks
    """
    if len(df) < 2:
        return None

    returns = df['Return'].dropna().values if 'Return' in df.columns else df['Close'].pct_change().dropna().values
    if len(returns) < 2:
        return None

    last_price = float(df['Close'].iloc[-1])
    mu = float(np.mean(returns))
    v0 = float(np.var(returns)) if np.var(returns) > 0 else theta

    dt = 1.0 / 252.0  # Daily step scaled annually
    
    # Pre-generate correlated random normal variables for all paths and time steps
    cov_matrix = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(cov_matrix)

    paths = np.zeros((simulations, days + 1))
    paths[:, 0] = last_price
    
    v_t = np.full(simulations, max(v0, 1e-4))

    for t in range(1, days + 1):
        z_uncorrelated = np.random.normal(0, 1, size=(simulations, 2))
        z_correlated = np.dot(z_uncorrelated, L.T)
        
        dW_S = z_correlated[:, 0]
        dW_V = z_correlated[:, 1]
        
        v_t_pos = np.maximum(v_t, 1e-6)
        
        # Heston Variance Update: dV_t
        dv = kappa * (theta - v_t_pos) * dt + xi * np.sqrt(v_t_pos) * np.sqrt(dt) * dW_V
        v_t = v_t_pos + dv
        v_t = np.maximum(v_t, 1e-6)
        
        # Heston Stock Price Update: dS_t
        ret = (mu - 0.5 * v_t) * dt + np.sqrt(v_t) * np.sqrt(dt) * dW_S
        paths[:, t] = paths[:, t - 1] * np.exp(ret)

    # Convert to DataFrame shape: (days + 1, simulations)
    sim_df = pd.DataFrame(paths.T)
    final_prices = paths[:, -1]
    
    var_95 = float(np.percentile(final_prices, 5))
    cvar_95 = float(np.mean(final_prices[final_prices <= var_95])) if len(final_prices[final_prices <= var_95]) > 0 else var_95

    return {
        "mean_expected": float(np.mean(final_prices)),
        "median_expected": float(np.median(final_prices)),
        "var_95": float(var_95),
        "cvar_95": float(cvar_95),
        "paths": sim_df
    }

def compute_tp_sl_probability(paths: pd.DataFrame, entry_price: float, take_profit: float, stop_loss: float) -> dict:
    """
    Evaluates the probability of paths hitting take_profit before stop_loss.
    """
    if paths is None or paths.empty:
        return {"prob_tp_first": 0.5, "prob_sl_first": 0.5, "prob_neither": 0.0}

    # Matrix shape: (simulations, days)
    paths_matrix = paths.values.T
    
    n_paths = paths_matrix.shape[0]
    tp_hits = 0
    sl_hits = 0
    neither = 0

    for path in paths_matrix:
        tp_idx = np.where(path >= take_profit)[0]
        sl_idx = np.where(path <= stop_loss)[0]

        first_tp = tp_idx[0] if len(tp_idx) > 0 else 999999
        first_sl = sl_idx[0] if len(sl_idx) > 0 else 999999

        if first_tp < first_sl:
            tp_hits += 1
        elif first_sl < first_tp:
            sl_hits += 1
        else:
            neither += 1

    return {
        "prob_tp_first": round(tp_hits / n_paths, 4),
        "prob_sl_first": round(sl_hits / n_paths, 4),
        "prob_neither": round(neither / n_paths, 4),
        "tp_ratio": round(tp_hits / (tp_hits + sl_hits + 1e-6), 4)
    }
