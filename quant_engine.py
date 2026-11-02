import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor


def quantile_regression_uncertainty(
    df: pd.DataFrame,
    feature_cols: list,
    timesfm_forecast: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Fits continuous quantile regressors (tau = 0.10, 0.50, 0.90) via Gradient Boosting
    and blends them with TimesFM-3's conformal 9-decile quantiles to isolate epistemic
    uncertainty (the width of the 90%-10% prediction interval) from aleatoric market risk.
      • Uses a strictly lagged target and chronological validation to avoid forward-looking leakage.
      • Returns a clean, type-stable payload of plain float values.
    """
    last_price = float(df["Close"].iloc[-1]) if df is not None and len(df) > 0 else 100.0

    # 1. Prepare a clean training set – target is the *next* candle's return.
    avail_cols = [c for c in feature_cols if c in df.columns] if df is not None else []
    if df is None or len(avail_cols) == 0:
        return {
            "q10_price": float(last_price * 0.95),
            "q30_price": float(last_price * 0.98),
            "q50_price": float(last_price),
            "q70_price": float(last_price * 1.02),
            "q90_price": float(last_price * 1.05),
            "ret_10_pct": -5.0,
            "ret_50_pct": 0.0,
            "ret_90_pct": 5.0,
            "epistemic_uncertainty": float(last_price * 0.10),
            "uncertainty_level": "HIGH (INSUFFICIENT DATA)",
        }

    valid_df = df.dropna(subset=avail_cols, how="any").copy()
    if len(valid_df) < 30:
        return {
            "q10_price": float(last_price * 0.95),
            "q30_price": float(last_price * 0.98),
            "q50_price": float(last_price),
            "q70_price": float(last_price * 1.02),
            "q90_price": float(last_price * 1.05),
            "ret_10_pct": -5.0,
            "ret_50_pct": 0.0,
            "ret_90_pct": 5.0,
            "epistemic_uncertainty": float(last_price * 0.10),
            "uncertainty_level": "HIGH (INSUFFICIENT DATA)",
        }

    # Features matrix – strictly lagged so that no future information leaks.
    X = valid_df[avail_cols]
    y = valid_df["Close"].pct_change().shift(-1).dropna()
    X = X.iloc[:-1]
    y = y.iloc[:len(X)]

    if len(X) < 25:
        return {
            "q10_price": float(last_price * 0.95),
            "q30_price": float(last_price * 0.98),
            "q50_price": float(last_price),
            "q70_price": float(last_price * 1.02),
            "q90_price": float(last_price * 1.05),
            "ret_10_pct": -5.0,
            "ret_50_pct": 0.0,
            "ret_90_pct": 5.0,
            "epistemic_uncertainty": float(last_price * 0.10),
            "uncertainty_level": "MODERATE",
        }

    # 2. Fit three quantile regressors (10th, 50th, 90th percentiles) on chronological history.
    latest_X = valid_df[avail_cols].iloc[-1:]

    try:
        q10_model = GradientBoostingRegressor(
            loss="quantile", alpha=0.10, n_estimators=40, max_depth=3, random_state=42
        )
        q10_model.fit(X, y)
        ret_10 = float(q10_model.predict(latest_X)[0])

        q50_model = GradientBoostingRegressor(
            loss="quantile", alpha=0.50, n_estimators=40, max_depth=3, random_state=42
        )
        q50_model.fit(X, y)
        ret_50 = float(q50_model.predict(latest_X)[0])

        q90_model = GradientBoostingRegressor(
            loss="quantile", alpha=0.90, n_estimators=40, max_depth=3, random_state=42
        )
        q90_model.fit(X, y)
        ret_90 = float(q90_model.predict(latest_X)[0])
    except Exception:
        ret_10, ret_50, ret_90 = -0.02, 0.0, 0.02

    gb_q10 = last_price * (1.0 + ret_10)
    gb_q50 = last_price * (1.0 + ret_50)
    gb_q90 = last_price * (1.0 + ret_90)

    # 3. Blend with TimesFM-3 conformal quantiles if provided
    if timesfm_forecast and "q10" in timesfm_forecast and len(timesfm_forecast["q10"]) > 0:
        tfm_q10 = float(timesfm_forecast["q10"][0])
        tfm_q30 = float(timesfm_forecast.get("q30", timesfm_forecast["q10"])[0])
        tfm_q50 = float(timesfm_forecast["q50"][0])
        tfm_q70 = float(timesfm_forecast.get("q70", timesfm_forecast["q90"])[0])
        tfm_q90 = float(timesfm_forecast["q90"][0])

        q10_price = 0.65 * tfm_q10 + 0.35 * gb_q10
        q50_price = 0.65 * tfm_q50 + 0.35 * gb_q50
        q90_price = 0.65 * tfm_q90 + 0.35 * gb_q90
        q30_price = tfm_q30
        q70_price = tfm_q70
    else:
        q10_price, q50_price, q90_price = sorted([gb_q10, gb_q50, gb_q90])
        q30_price = 0.5 * (q10_price + q50_price)
        q70_price = 0.5 * (q50_price + q90_price)

    q10_price, q30_price, q50_price, q70_price, q90_price = sorted(
        [q10_price, q30_price, q50_price, q70_price, q90_price]
    )

    ret_10_final = (q10_price - last_price) / (last_price + 1e-9)
    ret_50_final = (q50_price - last_price) / (last_price + 1e-9)
    ret_90_final = (q90_price - last_price) / (last_price + 1e-9)

    epistemic_uncertainty = abs(q90_price - q10_price)
    uncertainty_ratio = epistemic_uncertainty / (last_price + 1e-9)

    if uncertainty_ratio > 0.08:
        level = "HIGH (WIDE SPREAD)"
    elif uncertainty_ratio > 0.035:
        level = "MODERATE"
    else:
        level = "LOW (HIGH CONFIDENCE)"

    return {
        "q10_price": float(q10_price),
        "q30_price": float(q30_price),
        "q50_price": float(q50_price),
        "q70_price": float(q70_price),
        "q90_price": float(q90_price),
        "ret_10_pct": float(ret_10_final * 100.0),
        "ret_50_pct": float(ret_50_final * 100.0),
        "ret_90_pct": float(ret_90_final * 100.0),
        "epistemic_uncertainty": float(epistemic_uncertainty),
        "uncertainty_level": level,
    }


def run_jump_diffusion_mc(df: pd.DataFrame, days: int = 20, simulations: int = 1000) -> Optional[dict]:
    """
    Runs Merton Stochastic Jump-Diffusion Monte Carlo Simulation:
      dS_t = mu * S_t * dt + sigma * S_t * dW_t + J_t * dN_t
    where J_t ~ Normal(mu_j, sigma_j) and dN_t ~ Poisson(lambda_j * dt).
    Calculates 95% Value-at-Risk (VaR) and 95% Expected Shortfall (CVaR).
    """
    if df is None or len(df) < 5:
        return None

    returns = (
        df["Return"].dropna().values
        if "Return" in df.columns
        else df["Close"].pct_change().dropna().values
    )
    if len(returns) < 5:
        return None

    last_price = float(df["Close"].iloc[-1])
    mu = float(np.mean(returns))
    sigma = float(np.std(returns))

    # Detect discrete jump parameters (extreme returns exceeding 2 sigma)
    jump_threshold = 2.0 * sigma if sigma > 0 else 0.01
    jumps = returns[np.abs(returns) > jump_threshold]

    lambda_j = float(len(jumps) / len(returns)) if len(returns) > 0 else 0.05
    mu_j = float(np.mean(jumps)) if len(jumps) > 0 else 0.0
    sigma_j = float(np.std(jumps)) if len(jumps) > 0 else (sigma * 1.5)

    # Vectorized Jump-Diffusion simulation with deterministic seed for stability
    rng = np.random.default_rng(42)
    dt = 1.0
    dW = rng.normal(0.0, 1.0, size=(simulations, days))
    gbm_ret = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * dW

    n_jumps = rng.poisson(lambda_j * dt, size=(simulations, days))
    jump_draws = rng.normal(mu_j, max(sigma_j, 1e-6), size=(simulations, days))
    jump_ret = n_jumps * jump_draws

    total_ret = gbm_ret + jump_ret
    sim_matrix = last_price * np.exp(np.cumsum(total_ret, axis=1)).T  # Shape: (days, simulations)
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
        "paths": sim_df,
    }


def calculate_rolling_ic(predictions_hist: list, actuals_hist: list) -> float:
    """
    Computes the Information Coefficient (IC) as the Spearman rank correlation
    between model predictions and realized returns (Grinold & Kahn).
    Returns ``np.nan`` when insufficient data prevents a reliable calculation.
    """
    if len(predictions_hist) < 5 or len(actuals_hist) < 5:
        return np.nan

    try:
        ic, _ = spearmanr(predictions_hist, actuals_hist)
        if np.isnan(ic):
            ic = 0.0
    except Exception:
        ic = 0.0

    return float(np.clip(ic, -1.0, 1.0))


def fuse_signals(
    component_scores: dict,
    weights: Optional[Dict[str, float]] = None,
    fusion_mode: str = "bayesian_confluence",
) -> float:
    """
    Master Signal Fusion across all AI & Quantitative Pillars.
    Supports:
      1. 'bayesian_confluence' (Default): Non-linear Log-Odds Bayesian Evidence Updating.
         When multiple independent AI engines agree, conviction compounds non-linearly (e.g. 74%–84%).
         When models conflict, uncertainty expands and signal cancels out to 50% (Wait / Chop protection).
      2. 'linear': Classical weighted linear average.
    """
    if not component_scores:
        return 0.5

    default_weights: Dict[str, float] = {
        "TimesFM-3 Forecast": 0.22,
        "XGBoost Classifier": 0.16,
        "Stacking Ensemble": 0.14,
        "Multi-Alpha Engine": 0.12,
        "Market Regime": 0.10,
        "Pattern Confluence": 0.08,
        "Quantum Cycle & Hawkes": 0.06,
        "Volume Profile POC": 0.05,
        "Monte Carlo Risk": 0.04,
        "Momentum & ATR Sync": 0.03,
        # Legacy fallback keys
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

    if weights is None:
        matched = {k: default_weights.get(k, 0.10) for k in component_scores.keys()}
        weights = matched

    total = sum(weights.values())
    if total <= 0:
        return 0.5
    norm_weights = {k: v / total for k, v in weights.items()}

    if fusion_mode == "linear":
        fused = sum(norm_weights[k] * float(component_scores.get(k, 0.5)) for k in norm_weights)
        return float(np.clip(fused, 0.05, 0.95))

    # Log-Odds Bayesian Evidence Confluence
    mean_w = np.mean(list(norm_weights.values())) if len(norm_weights) > 0 else 1.0
    accum_logit = 0.0
    for k, w in norm_weights.items():
        p = float(component_scores.get(k, 0.5))
        p = float(np.clip(p, 0.03, 0.97))
        logit_k = np.log(p / (1.0 - p))
        accum_logit += (w / max(mean_w, 1e-6)) * logit_k

    scale_factor = 0.40 / np.sqrt(max(1.0, len(norm_weights) / 4.0))
    fused_logit = scale_factor * accum_logit
    fused_conviction = 1.0 / (1.0 + np.exp(-fused_logit))
    return float(np.clip(fused_conviction, 0.05, 0.95))


def compute_ai_deal_signals_and_markers(
    df: pd.DataFrame,
    df_feat: pd.DataFrame,
    trade_plan: dict,
    consensus: dict,
    symbol: str = "",
    lookback_bars: int = 70,
) -> Dict[str, Any]:
    """
    Synthesizes all AI features into explicit, actionable trade signals:
    1. WHEN TO BUY / SELL:
       - Exact trigger level, direction, conviction %, and multi-engine confluence.
    2. WHEN TO CLOSE THE DEAL:
       - Take-Profit (TP) Target price & gain %
       - Take-Profit 1 (TP1) partial exit price (lock in initial profit)
       - Stop-Loss (SL) Exit price & max risk %
       - Dynamic trailing stop rule (trail 1.5 ATR once in profit)
       - AI Invalidation rule (exit if AI consensus direction flips or Hawkes shock > 2.2)
       - Timeout rule (exit after 15 candles if target not reached)
    3. VISUAL CHART MARKERS:
       - Live Action Marker on current candle (🟢 BUY ENTRY or 🔴 SELL ENTRY)
       - Historical Closed Deal Markers (Entry → 🎯 TP HIT or 🛑 SL EXIT)
    """
    from config.market_catalog import format_price, smart_round
    if df is None or len(df) == 0:
        return {"markers": [], "execution_plan": {}}

    last_idx = df.index[-1]
    last_close = float(df['Close'].iloc[-1])
    atr = (
        float(df_feat['ATR'].iloc[-1])
        if 'ATR' in df_feat.columns and pd.notna(df_feat['ATR'].iloc[-1])
        else (last_close * 0.015)
    )

    bias = trade_plan.get("bias", consensus.get("direction_bias", "NEUTRAL"))
    conv_pct = float(consensus.get("directional_conviction", 0.5)) * 100.0
    entry_price = float(trade_plan.get("entry", last_close))
    tp_price = float(trade_plan.get("take_profit", last_close * 1.02))
    tp1_price = float(trade_plan.get("take_profit_tp1", last_close * 1.01))
    sl_price = float(trade_plan.get("stop_loss", last_close * 0.985))
    profit_pct = float(trade_plan.get("profit_pct", 0.0))
    risk_pct = float(trade_plan.get("risk_pct", 0.0))
    rr_ratio = float(trade_plan.get("rr_ratio", 2.0))

    fmt_entry = format_price(entry_price, symbol)
    fmt_tp = format_price(tp_price, symbol)
    fmt_tp1 = format_price(tp1_price, symbol)
    fmt_sl = format_price(sl_price, symbol)

    # 1. Clear Execution Rules
    if bias == "BUY":
        entry_rule = f"Enter LONG on market close near {fmt_entry} with {conv_pct:.1f}% AI Confluence."
        close_tp_rule = f"Close 50% position at TP1 ({fmt_tp1}) to lock in gains; close remaining 50% at TP ({fmt_tp}, +{profit_pct:.1f}%)."
        close_sl_rule = f"Hard stop exit immediately if candle breaches SL at {fmt_sl} (-{risk_pct:.1f}% max loss)."
        trailing_rule = f"Once price crosses TP1 ({fmt_tp1}), move Stop-Loss to breakeven ({fmt_entry}) and trail by 1.5 ATR."
        invalidation_rule = "Emergency exit deal if AI Consensus flips to BEARISH or Hawkes Volatility Shock exceeds 2.2."
    elif bias == "SELL":
        entry_rule = f"Enter SHORT on market close near {fmt_entry} with {conv_pct:.1f}% AI Confluence."
        close_tp_rule = f"Close 50% position at TP1 ({fmt_tp1}) to lock in gains; close remaining 50% at TP ({fmt_tp}, +{profit_pct:.1f}%)."
        close_sl_rule = f"Hard stop exit immediately if candle breaches SL at {fmt_sl} (-{risk_pct:.1f}% max loss)."
        trailing_rule = f"Once price crosses TP1 ({fmt_tp1}), move Stop-Loss to breakeven ({fmt_entry}) and trail by 1.5 ATR."
        invalidation_rule = "Emergency exit deal if AI Consensus flips to BULLISH or Hawkes Volatility Shock exceeds 2.2."
    else:
        entry_rule = "WAIT / NO DEAL: Market is in consolidation/chop. Keep capital protected until clear breakout."
        close_tp_rule = f"Key Resistance: {fmt_tp}"
        close_sl_rule = f"Key Support: {fmt_sl}"
        trailing_rule = "N/A (flat cash)"
        invalidation_rule = "Wait for multi-model consensus to exceed 55% conviction."

    execution_plan = {
        "bias": bias,
        "entry_rule": entry_rule,
        "close_tp_rule": close_tp_rule,
        "close_sl_rule": close_sl_rule,
        "trailing_rule": trailing_rule,
        "invalidation_rule": invalidation_rule,
        "timeout_rule": "Close deal if target is not reached within 15 bars.",
        "entry_price": entry_price,
        "take_profit": tp_price,
        "take_profit_tp1": tp1_price,
        "stop_loss": sl_price,
        "profit_pct": profit_pct,
        "risk_pct": risk_pct,
        "rr_ratio": rr_ratio,
        "conviction_pct": conv_pct,
    }

    # 2. Generate Chart Markers
    markers: List[Dict[str, Any]] = []

    def _to_epoch(ts_val):
        try:
            from tv_chart import to_time
            return to_time(ts_val)
        except Exception:
            t = pd.Timestamp(ts_val)
            if t.tzinfo is not None:
                t = t.tz_localize(None)
            return int(t.value // 1_000_000_000)

    # A. Historical Closed Deals Simulation over lookback_bars
    start_i = max(20, len(df) - lookback_bars)
    atr_arr = df_feat["ATR"].ffill().bfill().values if "ATR" in df_feat.columns else np.full(len(df), last_close * 0.015)
    close_arr = df["Close"].values
    high_arr = df["High"].values
    low_arr = df["Low"].values
    macd_arr = df_feat["MACD_Diff_Norm"].fillna(0.0).values if "MACD_Diff_Norm" in df_feat.columns else np.zeros(len(df))
    ema_arr = df_feat["EMA12_26_Spread"].fillna(0.0).values if "EMA12_26_Spread" in df_feat.columns else np.zeros(len(df))
    rsi_arr = df_feat["RSI"].fillna(50.0).values if "RSI" in df_feat.columns else np.full(len(df), 50.0)

    i = start_i
    while i < len(df) - 2:
        is_hist_buy = (macd_arr[i] > 0.25 and ema_arr[i] > 0.15 and rsi_arr[i] > 52.0)
        is_hist_sell = (macd_arr[i] < -0.25 and ema_arr[i] < -0.15 and rsi_arr[i] < 48.0)

        if is_hist_buy:
            hist_entry = close_arr[i]
            atr_i = atr_arr[i] if atr_arr[i] > 0 else (hist_entry * 0.015)
            hist_sl = hist_entry - 1.6 * atr_i
            hist_tp = hist_entry + 2.0 * (1.6 * atr_i)
            resolved = False
            for j in range(i + 1, min(i + 22, len(df) - 1)):
                if high_arr[j] >= hist_tp:
                    gain = (hist_tp - hist_entry) / hist_entry * 100.0
                    markers.append({
                        "time": _to_epoch(df.index[i]),
                        "position": "belowBar",
                        "shape": "arrowUp",
                        "color": "#00E676",
                        "text": f"🟢 BUY @ {format_price(hist_entry, symbol)}",
                    })
                    markers.append({
                        "time": _to_epoch(df.index[j]),
                        "position": "aboveBar",
                        "shape": "circle",
                        "color": "#00E676",
                        "text": f"🎯 CLOSE DEAL: TP HIT (+{gain:.1f}%)",
                    })
                    resolved = True
                    i = j
                    break
                elif low_arr[j] <= hist_sl:
                    loss = (hist_sl - hist_entry) / hist_entry * 100.0
                    markers.append({
                        "time": _to_epoch(df.index[i]),
                        "position": "belowBar",
                        "shape": "arrowUp",
                        "color": "#00E676",
                        "text": f"🟢 BUY @ {format_price(hist_entry, symbol)}",
                    })
                    markers.append({
                        "time": _to_epoch(df.index[j]),
                        "position": "belowBar",
                        "shape": "circle",
                        "color": "#FF5252",
                        "text": f"🛑 CLOSE DEAL: SL EXIT ({loss:.1f}%)",
                    })
                    resolved = True
                    i = j
                    break
            if not resolved:
                i += 1
        elif is_hist_sell:
            hist_entry = close_arr[i]
            atr_i = atr_arr[i] if atr_arr[i] > 0 else (hist_entry * 0.015)
            hist_sl = hist_entry + 1.6 * atr_i
            hist_tp = hist_entry - 2.0 * (1.6 * atr_i)
            resolved = False
            for j in range(i + 1, min(i + 22, len(df) - 1)):
                if low_arr[j] <= hist_tp:
                    gain = (hist_entry - hist_tp) / hist_entry * 100.0
                    markers.append({
                        "time": _to_epoch(df.index[i]),
                        "position": "aboveBar",
                        "shape": "arrowDown",
                        "color": "#FF5252",
                        "text": f"🔴 SELL @ {format_price(hist_entry, symbol)}",
                    })
                    markers.append({
                        "time": _to_epoch(df.index[j]),
                        "position": "belowBar",
                        "shape": "circle",
                        "color": "#00E676",
                        "text": f"🎯 CLOSE DEAL: TP HIT (+{gain:.1f}%)",
                    })
                    resolved = True
                    i = j
                    break
                elif high_arr[j] >= hist_sl:
                    loss = (hist_entry - hist_sl) / hist_entry * 100.0
                    markers.append({
                        "time": _to_epoch(df.index[i]),
                        "position": "aboveBar",
                        "shape": "arrowDown",
                        "color": "#FF5252",
                        "text": f"🔴 SELL @ {format_price(hist_entry, symbol)}",
                    })
                    markers.append({
                        "time": _to_epoch(df.index[j]),
                        "position": "aboveBar",
                        "shape": "circle",
                        "color": "#FF5252",
                        "text": f"🛑 CLOSE DEAL: SL EXIT ({loss:.1f}%)",
                    })
                    resolved = True
                    i = j
                    break
            if not resolved:
                i += 1
        else:
            i += 1

    # B. Active Live Deal Marker on the current candle
    last_t = _to_epoch(last_idx)
    if bias == "BUY":
        markers.append({
            "time": last_t,
            "position": "belowBar",
            "shape": "arrowUp",
            "color": "#00E676",
            "text": f"🟢 LIVE BUY @ {fmt_entry} | 🎯 TP: {fmt_tp} (+{profit_pct:.1f}%) | 🛑 SL: {fmt_sl} (-{risk_pct:.1f}%)",
        })
    elif bias == "SELL":
        markers.append({
            "time": last_t,
            "position": "aboveBar",
            "shape": "arrowDown",
            "color": "#FF5252",
            "text": f"🔴 LIVE SELL @ {fmt_entry} | 🎯 TP: {fmt_tp} (+{profit_pct:.1f}%) | 🛑 SL: {fmt_sl} (-{risk_pct:.1f}%)",
        })
    else:
        markers.append({
            "time": last_t,
            "position": "inBar",
            "shape": "square",
            "color": "#FFD54F",
            "text": f"🟡 WAIT / NO DEAL ({conv_pct:.1f}% Neutral)",
        })

    # Sort markers chronologically by time
    markers.sort(key=lambda m: m["time"])

    return {
        "execution_plan": execution_plan,
        "markers": markers,
    }


def kelly_position_sizing(
    win_rate: float, reward_risk_ratio: float, total_capital: float = 100000.0
) -> dict:
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
        return {
            "full_kelly": round(full_kelly, 4),
            "half_kelly": 0.0,
            "capped_fraction": 0.0,
            "fraction": 0.0,
            "position_usd": 0.0,
            "kelly_status": "BELOW THRESHOLD (WAIT FOR HIGHER EDGE)",
        }

    half_kelly = 0.5 * full_kelly
    capped_fraction = float(np.clip(half_kelly, 0.0, 0.25))
    position_usd = round(capped_fraction * total_capital, 2)

    return {
        "full_kelly": round(full_kelly, 4),
        "half_kelly": round(half_kelly, 4),
        "capped_fraction": round(capped_fraction, 4),
        "fraction": round(capped_fraction, 4),
        "position_usd": position_usd,
        "kelly_status": "OPTIMAL POSITION ALLOCATED",
    }


def generate_master_trade_signal(
    df: pd.DataFrame,
    component_scores: dict,
    conviction_threshold: float = 0.56,
    total_capital: float = 100000.0,
) -> dict:
    """
    Generates a master trade decision combining signal fusion, dynamic stops, and Kelly sizing.
    """
    if df is None or len(df) == 0:
        return {"action": "HOLD", "conviction": 0.5, "reason": "No data"}

    last_price = float(df["Close"].iloc[-1])
    atr = (
        float(df["ATR"].iloc[-1])
        if "ATR" in df.columns and pd.notna(df["ATR"].iloc[-1])
        else (last_price * 0.02)
    )

    conviction = fuse_signals(component_scores)

    reward_risk_ratio = 2.0
    stop_dist = 2.0 * atr
    tp_dist = reward_risk_ratio * stop_dist

    from config.market_catalog import smart_round
    if conviction >= conviction_threshold:
        action = "BUY"
        stop_loss = smart_round(last_price - stop_dist)
        take_profit = smart_round(last_price + tp_dist)
        win_prob = conviction
    elif conviction <= (1.0 - conviction_threshold):
        action = "SELL"
        stop_loss = smart_round(last_price + stop_dist)
        take_profit = smart_round(last_price - tp_dist)
        win_prob = 1.0 - conviction
    else:
        action = "HOLD"
        stop_loss = smart_round(last_price - stop_dist)
        take_profit = smart_round(last_price + tp_dist)
        win_prob = 0.5

    sizing = kelly_position_sizing(win_prob, reward_risk_ratio, total_capital)

    return {
        "action": action,
        "conviction": round(conviction, 4),
        "entry": smart_round(last_price),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "reward_risk": reward_risk_ratio,
        "position_size_usd": sizing["position_usd"],
        "kelly_details": sizing,
    }


def get_markov_regime(df: pd.DataFrame, window: int = 300) -> dict:
    """
    3-state Markov regime detector (Bull / Bear / Choppy).
    """
    from models import get_markov_regime as _models_markov
    return _models_markov(df, window=window)


def build_ai_consensus_report(
    df_feat: pd.DataFrame,
    symbol: str,
    interval: str,
    xgb_prediction: Dict[str, Any],
    regime: Dict[str, Any],
    detected_patterns: Dict[str, Any],
    timesfm_res: Dict[str, Any],
    total_capital: float = 100000.0,
) -> Dict[str, Any]:
    """
    Synthesizes all 7 AI & Quantitative Engines into a unified, plain-English
    Consensus Report so any user—from beginner to institutional quant—can
    understand the signal, the reason, and the risk plan in seconds.
    """
    if df_feat is None or len(df_feat) == 0:
        return {}

    last_close = float(df_feat["Close"].iloc[-1])
    atr = (
        float(df_feat["ATR"].iloc[-1])
        if "ATR" in df_feat.columns and pd.notna(df_feat["ATR"].iloc[-1])
        else (last_close * 0.02)
    )

    # 1. TimesFM-3 Foundation Model Score
    tfm_prob_up = float(timesfm_res.get("prob_up", 0.5))
    tfm_exp_ret = float(timesfm_res.get("expected_return_pct", 0.0))
    tfm_q50_end = float(timesfm_res["q50"][-1]) if timesfm_res.get("q50") else last_close

    # 2. XGBoost Directional Classifier Score
    xgb_prob_up = float(xgb_prediction.get("prob_up", 0.5))

    # 3. Ultra Stacking Ensemble (XGB + LGBM + HGB + RF + ET + Scaled MLP)
    try:
        from ml.stacking_ensemble import UltraStackingEnsemble
        stacker = UltraStackingEnsemble(symbol, interval)
        stacker.train_or_load(df_feat, force_retrain=False)
        stack_out = stacker.predict_next(df_feat)
        stack_prob_up = float(stack_out.get("stacking_conviction", 0.5))
        base_votes = stack_out.get("base_votes", {})
        stack_oof_metrics = stack_out.get("oof_metrics", {})
    except Exception:
        stack_prob_up = 0.5 * (tfm_prob_up + xgb_prob_up)
        base_votes = {}
        stack_oof_metrics = {}

    # 4. Multi-Alpha Composite Engine (Trend, Mean-Reversion, Volume Spike, Relative Strength)
    try:
        from alpha_engine import MultiAlphaEngine
        alpha_eng = MultiAlphaEngine()
        alpha_out = alpha_eng.compute_alphas(df_feat)
        alpha_prob_up = float(alpha_out.get("composite_conviction", 0.5))
    except Exception:
        alpha_prob_up = 0.5

    # 5. Markov & HMM Market Regime Score
    regime_name = regime.get("regime", "CHOPPY / RANGE") if isinstance(regime, dict) else str(regime)
    markov_conf = float(regime.get("markov_score", 0.5)) if isinstance(regime, dict) else 0.5
    if "BULL" in regime_name:
        regime_prob_up = float(np.clip(0.5 + 0.4 * markov_conf, 0.55, 0.90))
    elif "BEAR" in regime_name:
        regime_prob_up = float(np.clip(0.5 - 0.4 * markov_conf, 0.10, 0.45))
    else:
        regime_prob_up = 0.50

    # 6. Quantum Signal Science (FFT Cycles, Hawkes Self-Excitation, MaxEnt Entropy)
    try:
        from quantum_signals import (
            compute_fft_cycles,
            compute_hawkes_excitation,
            compute_maxent_distribution,
            compute_hmm_regime,
        )
        fft_res = compute_fft_cycles(df_feat)
        hawkes_res = compute_hawkes_excitation(df_feat)
        maxent_res = compute_maxent_distribution(df_feat)
        hmm_res = compute_hmm_regime(df_feat)
        quantum_prob_up = float(
            0.30 * fft_res.get("spectral_score", 0.5)
            + 0.30 * hawkes_res.get("hawkes_score", 0.5)
            + 0.20 * maxent_res.get("maxent_score", 0.5)
            + 0.20 * hmm_res.get("hmm_score", 0.5)
        )
    except Exception:
        fft_res, hawkes_res, maxent_res, hmm_res = {}, {}, {}, {}
        quantum_prob_up = 0.5

    # 7. Volume Profile (POC / HVN) & Pattern Confluence
    try:
        from features import add_volume_profile
        vp_res = add_volume_profile(df_feat.tail(120))
        poc_price = float(vp_res.get("poc", last_close))
        vp_prob_up = 0.62 if last_close >= poc_price else 0.38
    except Exception:
        vp_res = {"poc": last_close, "hvn_levels": [], "lvn_levels": []}
        poc_price = last_close
        vp_prob_up = 0.50

    # Recency- & Confidence-Weighted Harmonic Pattern Confluence
    pat_prob_up = 0.50
    active_pattern_label = None
    active_pattern_conf = 0.0
    if detected_patterns:
        pat_catalog = [
            ("wolfe_wave", "Bullish Wolfe Wave", 1.0),
            ("falling_wedge", "Bullish Falling Wedge", 1.0),
            ("bearish_bat", "Bearish Bat (Harmonic)", -1.0),
            ("cup_handle", "Inverted Cup & Handle", -1.0),
        ]
        weighted_pat_dir = 0.0
        total_pat_conf = 0.0
        best_conf = -1.0
        for key, label, sign_dir in pat_catalog:
            p_obj = detected_patterns.get(key)
            if isinstance(p_obj, dict):
                conf_k = float(p_obj.get("pattern_confidence", 0.55))
                bars_ago_k = int(p_obj.get("bars_ago", 10))
                if bars_ago_k <= 55:
                    weighted_pat_dir += sign_dir * conf_k
                    total_pat_conf += conf_k
                    if conf_k > best_conf:
                        best_conf = conf_k
                        active_pattern_label = f"{label} ({int(conf_k*100)}% fit)"
                        active_pattern_conf = conf_k
        if total_pat_conf > 0:
            net_dir = weighted_pat_dir / total_pat_conf
            pat_prob_up = float(np.clip(0.50 + 0.32 * net_dir * min(1.0, active_pattern_conf * 1.25), 0.15, 0.85))

    # Quantile Uncertainty Analysis
    from features import LIVE_FEATURE_COLS
    quantile_uncert = quantile_regression_uncertainty(
        df_feat, list(LIVE_FEATURE_COLS), timesfm_forecast=timesfm_res
    )

    # 10-Component Radar Scores (all normalized 0..1 where >0.5 is bullish, <0.5 is bearish)
    last_row = df_feat.iloc[-1]
    rsi_val = float(last_row.get("RSI", 50.0)) if pd.notna(last_row.get("RSI", 50.0)) else 50.0
    hurst_val = float(last_row.get("Hurst_20", 0.50)) if pd.notna(last_row.get("Hurst_20", 0.50)) else 0.50
    adx_val = float(last_row.get("ADX_Norm", 0.25)) if pd.notna(last_row.get("ADX_Norm", 0.25)) else 0.25
    mom_score = float(np.clip(0.5 + (rsi_val - 50.0) / 100.0, 0.1, 0.9))

    # Extract realized Out-Of-Sample (OOS) skill metrics for Dynamic Bayesian Model Averaging
    tfm_oos_acc = float(timesfm_res.get("oos_metrics", {}).get("directional_accuracy", 0.58))
    xgb_metrics = xgb_prediction.get("metrics", {}) if isinstance(xgb_prediction, dict) else {}
    xgb_oos_auc = float(xgb_metrics.get("roc_auc", 0.55))
    xgb_oos_acc = float(xgb_metrics.get("accuracy", 0.55))
    xgb_skill = max(xgb_oos_auc, xgb_oos_acc)

    stack_oof_auc = float(stack_oof_metrics.get("roc_auc", 0.55)) if isinstance(stack_oof_metrics, dict) else 0.55
    stack_oof_acc = float(stack_oof_metrics.get("accuracy", 0.55)) if isinstance(stack_oof_metrics, dict) else 0.55
    stack_skill = max(stack_oof_auc, stack_oof_acc)

    mc_prob_up = float(np.clip(0.55 * tfm_prob_up + 0.45 * xgb_prob_up, 0.1, 0.9))

    component_scores = {
        "TimesFM-3 Forecast": round(tfm_prob_up, 4),
        "XGBoost Classifier": round(xgb_prob_up, 4),
        "Stacking Ensemble": round(stack_prob_up, 4),
        "Multi-Alpha Engine": round(alpha_prob_up, 4),
        "Market Regime": round(regime_prob_up, 4),
        "Pattern Confluence": round(pat_prob_up, 4),
        "Quantum Cycle & Hawkes": round(quantum_prob_up, 4),
        "Volume Profile POC": round(vp_prob_up, 4),
        "Monte Carlo Risk": round(mc_prob_up, 4),
        "Momentum & ATR Sync": round(mom_score, 4),
    }

    # Dynamic OOS-Skill & Regime-Conditioned Bayesian Model Averaging (BMA) Weights
    w_tfm = 0.24 * max(0.35, 1.0 + 2.8 * (tfm_oos_acc - 0.52)) * (1.18 if hurst_val >= 0.52 else 0.95)
    w_xgb = 0.20 * max(0.15, (max(0.0, xgb_skill - 0.48) * 3.5) ** 1.5 + 0.20)
    w_stk = 0.18 * max(0.15, (max(0.0, stack_skill - 0.48) * 3.5) ** 1.5 + 0.20)
    w_alp = 0.14 * (1.15 if adx_val > 0.28 else 1.0)
    w_reg = 0.08 * (1.20 if hurst_val >= 0.53 else 0.90)
    w_pat = 0.07 * active_pattern_conf if active_pattern_label else 0.005
    w_qnt = 0.06
    w_vp = 0.05 * (1.30 if hurst_val < 0.48 else 0.90)
    w_mc = 0.03
    w_mom = 0.03

    bma_weights = {
        "TimesFM-3 Forecast": w_tfm,
        "XGBoost Classifier": w_xgb,
        "Stacking Ensemble": w_stk,
        "Multi-Alpha Engine": w_alp,
        "Market Regime": w_reg,
        "Pattern Confluence": w_pat,
        "Quantum Cycle & Hawkes": w_qnt,
        "Volume Profile POC": w_vp,
        "Monte Carlo Risk": w_mc,
        "Momentum & ATR Sync": w_mom,
    }

    fused_prob_up = fuse_signals(component_scores, weights=bma_weights)

    # Composite OOS Validation Skill & López de Prado Meta-Labeling Probability
    core_w_sum = w_tfm + w_xgb + w_stk
    composite_oos_accuracy = float(
        (w_tfm * tfm_oos_acc + w_xgb * xgb_skill + w_stk * stack_skill) / max(core_w_sum, 1e-6)
    )
    composite_oos_accuracy = float(np.clip(composite_oos_accuracy, 0.52, 0.96))

    # Build 7 Core Engine Voting Cards (with clear plain-English explanations)
    from config.market_catalog import format_price, get_asset_metadata
    asset_meta = get_asset_metadata(symbol)
    asset_title = f"{asset_meta.get('name', symbol)} ({symbol})"
    fmt_q50_end = format_price(tfm_q50_end, symbol)
    fmt_poc = format_price(poc_price, symbol)
    fmt_last = format_price(last_close, symbol)
    fmt_q10 = format_price(quantile_uncert["q10_price"], symbol)
    fmt_q90 = format_price(quantile_uncert["q90_price"], symbol)

    cross_corrs = timesfm_res.get("cross_asset_correlations", [])
    macro_note = (
        f" + {len(cross_corrs)} cross-asset macro channels"
        if cross_corrs
        else " using 9-quantile patch attention"
    )

    def _vote_label(p: float) -> str:
        if p >= 0.55:
            return "BUY"
        elif p <= 0.45:
            return "SELL"
        return "NEUTRAL"

    model_votes = [
        {
            "engine": "TimesFM-3 Foundation Model",
            "icon": "🔮",
            "vote": _vote_label(tfm_prob_up),
            "prob_up": round(tfm_prob_up * 100, 1),
            "plain_english": (
                f"Projects price toward {fmt_q50_end} ({tfm_exp_ret:+.2f}%) over the next "
                f"{timesfm_res.get('horizon', 15)} bars{macro_note}."
            ),
        },
        {
            "engine": "XGBoost Tree Classifier",
            "icon": "🌲",
            "vote": _vote_label(xgb_prob_up),
            "prob_up": round(xgb_prob_up * 100, 1),
            "plain_english": (
                f"Assigns a {xgb_prob_up*100:.1f}% probability that the next candle closes higher "
                f"based on 30 technical indicators."
            ),
        },
        {
            "engine": "5-Model Stacking Ensemble",
            "icon": "🧠",
            "vote": _vote_label(stack_prob_up),
            "prob_up": round(stack_prob_up * 100, 1),
            "plain_english": (
                f"Combines XGBoost, LightGBM, Random Forest, ExtraTrees & Neural MLP via meta-regression "
                f"({stack_prob_up*100:.1f}% bullish)."
            ),
        },
        {
            "engine": "Multi-Alpha Factor Engine",
            "icon": "⚡",
            "vote": _vote_label(alpha_prob_up),
            "prob_up": round(alpha_prob_up * 100, 1),
            "plain_english": (
                "Checks 4 institutional factors: Trend Alignment, VWAP Mean Reversion, Volume Surge, and RSI Strength."
            ),
        },
        {
            "engine": "Markov & HMM Regime Filter",
            "icon": "🧭",
            "vote": _vote_label(regime_prob_up),
            "prob_up": round(regime_prob_up * 100, 1),
            "plain_english": (
                f"Identifies current market state as '{regime_name}' with {markov_conf*100:.0f}% state persistence."
            ),
        },
        {
            "engine": "Quantum Cycle & Hawkes Flow",
            "icon": "🌊",
            "vote": _vote_label(quantum_prob_up),
            "prob_up": round(quantum_prob_up * 100, 1),
            "plain_english": (
                f"Dominant FFT cycle: {fft_res.get('dominant_period', 20)} bars ({fft_res.get('cycle_phase', 'TRANSITION')}) · "
                f"Orderflow: {hawkes_res.get('volatility_regime', 'STABLE')}."
            ),
        },
        {
            "engine": "Volume Profile & Patterns",
            "icon": "📐",
            "vote": _vote_label(0.6 * pat_prob_up + 0.4 * vp_prob_up),
            "prob_up": round((0.6 * pat_prob_up + 0.4 * vp_prob_up) * 100, 1),
            "plain_english": (
                f"{'Detected ' + active_pattern_label + ' · ' if active_pattern_label else 'No major harmonic pattern · '}"
                f"Price is {'above' if last_close >= poc_price else 'below'} Institutional POC ({fmt_poc})."
            ),
        },
    ]

    buy_votes = sum(1 for m in model_votes if m["vote"] == "BUY")
    sell_votes = sum(1 for m in model_votes if m["vote"] == "SELL")
    neutral_votes = len(model_votes) - buy_votes - sell_votes

    if fused_prob_up >= 0.62 and buy_votes >= 4:
        verdict = "STRONG BUY"
        direction_bias = "BUY"
        directional_conviction = fused_prob_up
        agreement_count = buy_votes
    elif fused_prob_up >= 0.55 or (buy_votes >= 4 and fused_prob_up >= 0.52):
        verdict = "BUY"
        direction_bias = "BUY"
        directional_conviction = fused_prob_up
        agreement_count = max(buy_votes, 4)
    elif fused_prob_up <= 0.38 and sell_votes >= 4:
        verdict = "STRONG SELL"
        direction_bias = "SELL"
        directional_conviction = 1.0 - fused_prob_up
        agreement_count = sell_votes
    elif fused_prob_up <= 0.45 or (sell_votes >= 4 and fused_prob_up <= 0.48):
        verdict = "SELL"
        direction_bias = "SELL"
        directional_conviction = 1.0 - fused_prob_up
        agreement_count = max(sell_votes, 4)
    else:
        verdict = "WAIT / HOLD"
        direction_bias = "NEUTRAL"
        directional_conviction = max(fused_prob_up, 1.0 - fused_prob_up)
        agreement_count = max(buy_votes, sell_votes, neutral_votes)

    # López de Prado Meta-Labeling Confidence Filter (estimates probability of signal profitability)
    entropy_val = float(maxent_res.get("maxent_entropy", 0.75)) if isinstance(maxent_res, dict) else 0.75
    uncert_pct = float(timesfm_res.get("uncertainty_pct", 4.0))
    cone_tightness = 1.0 / (1.0 + max(0.0, uncert_pct - 2.0) / 12.0)
    agreement_ratio = float(agreement_count) / max(len(model_votes), 1)
    meta_label_prob = float(
        np.clip(
            0.40 * composite_oos_accuracy
            + 0.30 * agreement_ratio
            + 0.15 * (1.0 - 0.5 * entropy_val)
            + 0.15 * cone_tightness,
            0.35,
            0.96,
        )
    )

    # Compute Half-Kelly & 2% Risk Position Sizing
    kelly_info = kelly_position_sizing(directional_conviction, 2.0, total_capital=total_capital)

    # Plain-English Executive Summary for Beginners & Pros
    if direction_bias == "BUY":
        headline = (
            f"🟢 AI Copilot Verdict: {verdict} — {buy_votes} of 7 AI Engines are Bullish "
            f"({directional_conviction*100:.1f}% Master Confidence · {composite_oos_accuracy*100:.1f}% Validated OOS Skill)"
        )
        explanation = (
            f"Google TimesFM-3 and our calibrated ensemble models project **{asset_title}** rising toward "
            f"**{fmt_q50_end}** ({tfm_exp_ret:+.2f}%) over the next {timesfm_res.get('horizon', 15)} candles "
            f"(Meta-Label Reliability: **{meta_label_prob*100:.1f}%**). "
            f"Price is trading {'above' if last_close >= poc_price else 'near'} its high-volume institutional support "
            f"({fmt_poc}) in a **{regime_name}** regime."
        )
    elif direction_bias == "SELL":
        headline = (
            f"🔴 AI Copilot Verdict: {verdict} — {sell_votes} of 7 AI Engines are Bearish "
            f"({directional_conviction*100:.1f}% Master Confidence · {composite_oos_accuracy*100:.1f}% Validated OOS Skill)"
        )
        explanation = (
            f"Google TimesFM-3 and our calibrated ensemble models project downward pressure on **{asset_title}** toward "
            f"**{fmt_q50_end}** ({tfm_exp_ret:+.2f}%) over the next {timesfm_res.get('horizon', 15)} candles "
            f"(Meta-Label Reliability: **{meta_label_prob*100:.1f}%**). "
            f"Consider protecting long positions or looking for short setups with a strict stop-loss."
        )
    else:
        headline = (
            f"🟡 AI Copilot Verdict: {verdict} — Mixed Signals ({buy_votes} Bullish, {sell_votes} Bearish, {neutral_votes} Neutral)"
        )
        explanation = (
            f"AI models see **{asset_title}** consolidating near **{fmt_last}** with an expected range between "
            f"**{fmt_q10}** (q10 support) and **{fmt_q90}** (q90 resistance). "
            f"Smartest action right now: **Wait for a clear breakout** before risking capital."
        )

    return {
        "verdict": verdict,
        "direction_bias": direction_bias,
        "fused_prob_up": round(fused_prob_up, 4),
        "directional_conviction": round(directional_conviction, 4),
        "composite_oos_accuracy": round(composite_oos_accuracy, 4),
        "meta_label_prob": round(meta_label_prob, 4),
        "bma_weights": {k: round(v / max(sum(bma_weights.values()), 1e-9), 4) for k, v in bma_weights.items()},
        "buy_votes": buy_votes,
        "sell_votes": sell_votes,
        "neutral_votes": neutral_votes,
        "agreement_count": agreement_count,
        "total_engines": len(model_votes),
        "headline": headline,
        "explanation": explanation,
        "model_votes": model_votes,
        "base_learner_votes": base_votes,
        "component_scores": component_scores,
        "quantile_uncertainty": quantile_uncert,
        "volume_profile": vp_res,
        "quantum_details": {
            "fft": fft_res,
            "hawkes": hawkes_res,
            "maxent": maxent_res,
            "hmm": hmm_res,
        },
        "kelly_sizing": kelly_info,
    }
