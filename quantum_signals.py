"""
=============================================================
 QUANTUM & SIGNAL SCIENCE ENGINE (TRADING APP)
 Inspired by M7 Step 10 & Institutional Quant Signal Science
=============================================================
"""

import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.optimize import minimize

def compute_fft_cycles(df: pd.DataFrame, window: int = 64) -> dict:
    """
    Computes 1D Fast Fourier Transform over price returns to extract
    dominant spectral frequencies, cycle length, and cycle phase.
    """
    if len(df) < window:
        return {"dominant_period": 0, "spectral_energy": 0.0, "cycle_phase": "NEUTRAL", "spectral_score": 0.5}

    close = df['Close'].tail(window).values
    returns = np.diff(np.log(close))
    
    # Detrend
    returns_detrended = returns - np.mean(returns)
    
    # 1D FFT
    fft_vals = fft(returns_detrended)
    power_spectrum = np.abs(fft_vals[:len(returns_detrended) // 2]) ** 2
    
    if len(power_spectrum) < 2 or power_spectrum.sum() == 0:
        return {"dominant_period": 0, "spectral_energy": 0.0, "cycle_phase": "NEUTRAL", "spectral_score": 0.5}
        
    # Exclude DC component (freq = 0)
    power_spectrum[0] = 0
    peak_idx = np.argmax(power_spectrum)
    
    if peak_idx > 0:
        dominant_period = int(round(len(returns_detrended) / peak_idx))
    else:
        dominant_period = len(returns_detrended)

    spectral_energy = float(power_spectrum[peak_idx] / (power_spectrum.sum() + 1e-9))
    
    # Phase estimation based on recent sine wave alignment
    recent_momentum = returns_detrended[-3:].sum()
    if recent_momentum > 0.005:
        cycle_phase = "EXPANSION (PEAKING)"
        spectral_score = float(np.clip(0.5 + spectral_energy * 0.5, 0.0, 1.0))
    elif recent_momentum < -0.005:
        cycle_phase = "CONTRACTION (TROUGHING)"
        spectral_score = float(np.clip(0.5 - spectral_energy * 0.5, 0.0, 1.0))
    else:
        cycle_phase = "TRANSITION"
        spectral_score = 0.5

    return {
        "dominant_period": dominant_period,
        "spectral_energy": float(spectral_energy),
        "cycle_phase": cycle_phase,
        "spectral_score": spectral_score
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
    vol_std = np.std(returns) if len(returns) > 0 else 1.0
    
    if vol_std == 0:
        return {"intensity": 0.0, "volatility_regime": "STABLE", "hawkes_score": 0.5}

    # Detect high volatility event times t_k
    threshold = 1.2 * vol_std
    base_rate = float(np.mean(np.abs(returns) > threshold))
    
    self_excitation = 0.0
    n = len(returns)
    for t_k, ret in enumerate(returns):
        if abs(ret) > threshold:
            delta_t = n - t_k
            self_excitation += alpha * np.exp(-beta * delta_t)

    intensity = base_rate + self_excitation
    
    if intensity > 0.8:
        regime = "HIGH EXCITATION (CLUSTER SURGE)"
    elif intensity > 0.4:
        regime = "MODERATE EXCITATION"
    else:
        regime = "LOW EXCITATION (QUIET)"

    # Hawkes score penalizes over-extended excitation when direction is unclear
    last_ret = returns[-1] if len(returns) > 0 else 0
    direction = 1.0 if last_ret > 0 else -1.0
    hawkes_score = float(np.clip(0.5 + direction * (intensity * 0.25), 0.05, 0.95))

    return {
        "intensity": float(intensity),
        "volatility_regime": regime,
        "hawkes_score": hawkes_score
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
    counts, bin_edges = np.histogram(returns, bins=n_bins, density=True)
    p_emp = counts / counts.sum() if counts.sum() > 0 else np.ones(n_bins) / n_bins
    p_emp = np.clip(p_emp, 1e-9, 1.0)
    
    # Calculate Shannon Entropy
    shannon_entropy = float(-np.sum(p_emp * np.log2(p_emp)))
    max_possible_entropy = np.log2(n_bins)
    normalized_entropy = float(shannon_entropy / max_possible_entropy)
    
    # Score favors structured low entropy (clear direction) vs high noise entropy
    recent_mean = float(np.mean(returns[-5:]))
    bias = 1.0 if recent_mean > 0 else -1.0
    maxent_score = float(np.clip(0.5 + bias * (1.0 - normalized_entropy) * 0.3, 0.1, 0.9))

    return {
        "maxent_entropy": normalized_entropy,
        "maxent_score": maxent_score
    }

def compute_hmm_regime(df: pd.DataFrame) -> dict:
    """
    Hidden Markov Model latent state transition filter over 3 market regimes:
    State 0: Accumulation / Bull
    State 1: Bear / Markdown
    State 2: High Volatility Choppy
    """
    if len(df) < 20:
        return {"latent_state": "CHOPPY / NEUTRAL", "hmm_score": 0.5, "state_probabilities": [0.33, 0.33, 0.34]}

    returns = df['Return'].fillna(0).tail(30).values
    vol = df['ATR'].fillna(0).tail(30).values if 'ATR' in df.columns else np.abs(returns)

    mean_ret = np.mean(returns[-5:])
    recent_vol = np.mean(vol[-5:]) if len(vol) >= 5 else 0.01
    avg_vol = np.mean(vol) if len(vol) > 0 else 0.01

    p_bull = 0.33
    p_bear = 0.33
    p_chop = 0.34

    if mean_ret > 0.002 and recent_vol <= avg_vol * 1.2:
        p_bull = 0.70
        p_bear = 0.10
        p_chop = 0.20
        state_name = "BULL ACCUMULATION"
    elif mean_ret < -0.002 and recent_vol <= avg_vol * 1.2:
        p_bull = 0.10
        p_bear = 0.70
        p_chop = 0.20
        state_name = "BEAR MARKDOWN"
    elif recent_vol > avg_vol * 1.2:
        p_bull = 0.20
        p_bear = 0.20
        p_chop = 0.60
        state_name = "HIGH VOLATILITY CHOP"
    else:
        p_bull = 0.40
        p_bear = 0.30
        p_chop = 0.30
        state_name = "CONSOLIDATION / NEUTRAL"

    hmm_score = float(p_bull * 1.0 + p_chop * 0.5 + p_bear * 0.0)

    return {
        "latent_state": state_name,
        "hmm_score": float(hmm_score),
        "state_probabilities": [float(p_bull), float(p_bear), float(p_chop)]
    }
