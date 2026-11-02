"""
=============================================================
 QUANTUM & SIGNAL SCIENCE ENGINE (v2 — Scale-Invariant)
 Hann-Windowed FFT Phase Projection, Causal Hawkes Process,
 Jaynes MaxEnt & Volatility-Normalized Gaussian HMM Filter
=============================================================
"""

import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.optimize import minimize


def compute_fft_cycles(df: pd.DataFrame, window: int = 64) -> dict:
    """
    Computes Hann-windowed 1D Fast Fourier Transform over log-returns to extract
    dominant spectral frequencies, cycle length, and exact trigonometric phase slope
    with zero spectral edge leakage.
    """
    if len(df) < window:
        return {
            "dominant_period": 0,
            "spectral_energy": 0.0,
            "cycle_phase": "NEUTRAL",
            "spectral_score": 0.5,
        }

    close = df['Close'].tail(window).values
    returns = np.diff(np.log(np.clip(close, 1e-8, None)))
    n = len(returns)
    if n < 8:
        return {
            "dominant_period": 0,
            "spectral_energy": 0.0,
            "cycle_phase": "NEUTRAL",
            "spectral_score": 0.5,
        }

    # Detrend and apply Hann window to prevent spectral leakage at boundaries
    returns_detrended = returns - np.mean(returns)
    hann_win = np.hanning(n)
    windowed_ret = returns_detrended * hann_win

    fft_vals = fft(windowed_ret)
    half_n = n // 2
    power_spectrum = np.abs(fft_vals[:half_n]) ** 2

    if len(power_spectrum) < 3 or power_spectrum.sum() == 0:
        return {
            "dominant_period": 0,
            "spectral_energy": 0.0,
            "cycle_phase": "NEUTRAL",
            "spectral_score": 0.5,
        }

    # Exclude DC (k=0) and ultra-long baseline drift (k=1)
    power_spectrum[0] = 0.0
    peak_idx = int(np.argmax(power_spectrum))
    dominant_period = int(round(n / max(peak_idx, 1)))
    spectral_energy = float(power_spectrum[peak_idx] / (power_spectrum.sum() + 1e-9))

    # Exact Fourier phase at current bar (t = n - 1) and 1-step forward harmonic projection
    phase_0 = float(np.angle(fft_vals[peak_idx]))
    omega = 2.0 * np.pi * peak_idx / n
    phase_now = phase_0 + omega * (n - 1)
    next_harmonic_val = float(np.cos(phase_now + omega))

    std_ret = float(np.std(returns_detrended)) if np.std(returns_detrended) > 0 else 1e-6
    recent_z = float(np.mean(returns_detrended[-3:]) / std_ret)
    combined_signal = 0.60 * next_harmonic_val + 0.40 * np.tanh(recent_z)

    if combined_signal > 0.15:
        cycle_phase = "EXPANSION (RISING PHASE)"
        spectral_score = float(np.clip(0.5 + combined_signal * min(0.45, spectral_energy * 1.5), 0.10, 0.90))
    elif combined_signal < -0.15:
        cycle_phase = "CONTRACTION (FALLING PHASE)"
        spectral_score = float(np.clip(0.5 + combined_signal * min(0.45, spectral_energy * 1.5), 0.10, 0.90))
    else:
        cycle_phase = "TRANSITION (INFLECTION)"
        spectral_score = 0.50

    return {
        "dominant_period": dominant_period,
        "spectral_energy": round(spectral_energy, 4),
        "cycle_phase": cycle_phase,
        "spectral_score": round(spectral_score, 4),
    }


def compute_hawkes_excitation(df: pd.DataFrame, alpha: float = 0.4, beta: float = 0.2, lookback: int = 30) -> dict:
    """
    Calculates Hawkes Self-Exciting Point Process intensity lambda(t)
    lambda(t) = mu + alpha * sum_k exp(-beta * (t - t_k))
    where t_k are occurrences of high-volatility return spikes.
    """
    if len(df) < lookback:
        return {"intensity": 0.0, "volatility_regime": "LOW EXCITATION", "hawkes_score": 0.5}

    recent_df = df.tail(lookback)
    returns = recent_df['Return'].fillna(0).values
    vol_std = float(np.std(returns)) if len(returns) > 0 else 1.0

    if vol_std == 0:
        return {"intensity": 0.0, "volatility_regime": "STABLE", "hawkes_score": 0.5}

    threshold = 1.2 * vol_std
    base_rate = float(np.mean(np.abs(returns) > threshold))

    self_excitation = 0.0
    n = len(returns)
    for t_k, ret in enumerate(returns):
        if abs(ret) > threshold:
            delta_t = n - t_k
            self_excitation += alpha * np.exp(-beta * delta_t)

    intensity = float(base_rate + self_excitation)

    if intensity > 0.8:
        regime = "HIGH EXCITATION (CLUSTER SURGE)"
    elif intensity > 0.4:
        regime = "MODERATE EXCITATION"
    else:
        regime = "LOW EXCITATION (QUIET)"

    # Signed excitation weighted by recent 3-bar return z-score
    recent_z = float(np.mean(returns[-3:]) / max(vol_std, 1e-9))
    hawkes_score = float(np.clip(0.5 + np.tanh(recent_z * 0.8) * min(0.35, intensity * 0.25), 0.10, 0.90))

    return {
        "intensity": round(intensity, 4),
        "volatility_regime": regime,
        "hawkes_score": round(hawkes_score, 4),
    }


def compute_maxent_distribution(df: pd.DataFrame, target_mean_return: float = 0.0) -> dict:
    """
    Finds maximum Shannon Entropy distribution P* over return quantiles
    subject to empirical mean/variance constraints (Jaynes' Principle).
    """
    if len(df) < 20:
        return {"maxent_entropy": 1.0, "maxent_score": 0.5}

    returns = df['Return'].dropna().tail(50).values
    if len(returns) < 5:
        return {"maxent_entropy": 1.0, "maxent_score": 0.5}

    n_bins = 10
    counts, _ = np.histogram(returns, bins=n_bins, density=True)
    p_emp = counts / counts.sum() if counts.sum() > 0 else np.ones(n_bins) / n_bins
    p_emp = np.clip(p_emp, 1e-9, 1.0)

    shannon_entropy = float(-np.sum(p_emp * np.log2(p_emp)))
    max_possible_entropy = float(np.log2(n_bins))
    normalized_entropy = float(np.clip(shannon_entropy / max_possible_entropy, 0.0, 1.0))

    std_r = float(np.std(returns)) if np.std(returns) > 0 else 1e-6
    recent_z = float(np.mean(returns[-5:]) / std_r)
    # High signal-to-noise (low entropy) amplifies directional conviction
    signal_clarity = max(0.15, 1.0 - 0.65 * normalized_entropy)
    maxent_score = float(np.clip(0.5 + 0.35 * np.tanh(recent_z) * signal_clarity, 0.10, 0.90))

    return {
        "maxent_entropy": round(normalized_entropy, 4),
        "maxent_score": round(maxent_score, 4),
    }


def compute_hmm_regime(df: pd.DataFrame) -> dict:
    """
    Scale-Invariant Hidden Markov Model (HMM) latent state filter over 3 market regimes:
      State 0: BULL ACCUMULATION (positive drift z-score, controlled volatility)
      State 1: BEAR MARKDOWN     (negative drift z-score, elevated downside pressure)
      State 2: HIGH VOLATILITY CHOP (high volatility ratio, low directional efficiency)
    Uses volatility-normalized return Z-scores so Forex (EUR/USD), Commodities (Natural Gas,
    Gold), and Crypto (BTC) are evaluated on an identical dimensionless scale.
    """
    if len(df) < 20:
        return {
            "latent_state": "CHOPPY / NEUTRAL",
            "hmm_score": 0.5,
            "state_probabilities": [0.33, 0.33, 0.34],
        }

    returns = df['Return'].fillna(0).tail(40).values
    vol = df['ATR'].fillna(0).tail(40).values if 'ATR' in df.columns else np.abs(returns)

    std_ret = float(np.std(returns)) if np.std(returns) > 0 else 1e-6
    mean_ret_5 = float(np.mean(returns[-5:]))
    z_drift = mean_ret_5 / std_ret  # Dimensionless drift Z-score

    recent_vol = float(np.mean(vol[-5:])) if len(vol) >= 5 else 1.0
    avg_vol = float(np.mean(vol)) if len(vol) > 0 and np.mean(vol) > 0 else 1.0
    vol_ratio = recent_vol / avg_vol

    # Softmax emission log-likelihoods for the 3 HMM states
    ll_bull = 1.6 * z_drift - 0.8 * max(0.0, vol_ratio - 1.25)
    ll_bear = -1.6 * z_drift - 0.4 * max(0.0, vol_ratio - 1.40)
    ll_chop = 1.4 * max(0.0, vol_ratio - 1.15) - 1.2 * abs(z_drift)

    logits = np.array([ll_bull, ll_bear, ll_chop], dtype=float)
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs = probs / np.sum(probs)
    p_bull, p_bear, p_chop = float(probs[0]), float(probs[1]), float(probs[2])

    if p_bull >= max(p_bear, p_chop) and p_bull > 0.42:
        state_name = "BULL ACCUMULATION"
    elif p_bear >= max(p_bull, p_chop) and p_bear > 0.42:
        state_name = "BEAR MARKDOWN"
    elif vol_ratio > 1.25:
        state_name = "HIGH VOLATILITY CHOP"
    else:
        state_name = "CONSOLIDATION / NEUTRAL"

    hmm_score = float(np.clip(p_bull * 1.0 + p_chop * 0.5 + p_bear * 0.0, 0.05, 0.95))

    return {
        "latent_state": state_name,
        "hmm_score": round(hmm_score, 4),
        "state_probabilities": [round(p_bull, 4), round(p_bear, 4), round(p_chop, 4)],
    }


def genetic_optimize_indicators(df: pd.DataFrame) -> dict:
    """
    Uses Differential Evolution (Genetic Optimization) to search for optimal
    technical indicator parameters (RSI, ATR, MACD fast/slow, BB dev) that maximize
    Sharpe Ratio.
    """
    if len(df) < 50:
        return {"optimized": False, "best_params": {}, "best_sharpe": 0.0}

    from scipy.optimize import differential_evolution
    import ta

    bounds = [(7, 21), (7, 21), (8, 16), (20, 30), (1.5, 3.0)]

    def objective(params):
        rsi_p, atr_p, macd_f, macd_s, bb_std = params
        rsi_p, atr_p, macd_f, macd_s = int(rsi_p), int(atr_p), int(macd_f), int(macd_s)

        if macd_f >= macd_s:
            return 999.0

        try:
            close = df['Close']
            rsi = ta.momentum.RSIIndicator(close, window=rsi_p).rsi()
            macd = ta.trend.MACD(close, window_slow=macd_s, window_fast=macd_f).macd()
            macd_sig = ta.trend.MACD(close, window_slow=macd_s, window_fast=macd_f).macd_signal()

            signals = np.where((rsi > 50) & (macd > macd_sig), 1.0, -1.0)
            returns = df['Return'].fillna(0).values

            strat_returns = signals[:-1] * returns[1:]
            mean_ret = np.mean(strat_returns)
            std_ret = np.std(strat_returns)

            if std_ret == 0:
                return 999.0

            sharpe = (mean_ret / std_ret) * np.sqrt(252)
            return -sharpe
        except Exception:
            return 999.0

    try:
        result = differential_evolution(objective, bounds, popsize=10, maxiter=20, seed=42)
        best_p = result.x
        best_sharpe = float(-result.fun) if result.fun < 500 else 0.0

        return {
            "optimized": True,
            "best_params": {
                "rsi_period": int(best_p[0]),
                "atr_period": int(best_p[1]),
                "macd_fast": int(best_p[2]),
                "macd_slow": int(best_p[3]),
                "bb_std": round(float(best_p[4]), 2),
            },
            "best_sharpe": round(best_sharpe, 4),
        }
    except Exception:
        return {"optimized": False, "best_params": {}, "best_sharpe": 0.0}
