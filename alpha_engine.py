"""
=============================================================
 MULTI-ALPHA COMPOSITE ENGINE & DYNAMIC RISK CONTROL (v2)
 6-Factor Continuous Regime-Conditioned Institutional Alphas
=============================================================
"""

import numpy as np
import pandas as pd


class MultiAlphaEngine:
    def __init__(self, requires_n_confirms: int = 2):
        self.requires_n_confirms = requires_n_confirms
        self.weights = {
            "alpha_1_trend": 0.24,
            "alpha_2_mean_reversion": 0.16,
            "alpha_3_volume_spike": 0.18,
            "alpha_4_relative_strength": 0.18,
            "alpha_5_order_flow": 0.12,
            "alpha_6_hurst_regime": 0.12,
        }

    def compute_alphas(self, df: pd.DataFrame) -> dict:
        """
        Computes 6 continuous, scale-invariant institutional alpha factors in [-1, +1]
        and aggregates them with regime-adaptive weights (boosting trend & momentum in
        high-Hurst/high-ADX regimes; boosting mean-reversion in low-Hurst range regimes).
        """
        if df is None or len(df) < 30:
            return {
                "composite_conviction": 0.5,
                "n_confirms": 0,
                "pos_confirms": 0,
                "neg_confirms": 0,
                "signal_fired": False,
                "action": "HOLD",
                "alphas": {},
            }

        latest = df.iloc[-1]
        close = float(latest.get('Close', 0.0))
        atr = float(latest.get('ATR', max(close * 0.01, 1e-6)))
        if atr <= 0 or np.isnan(atr):
            atr = max(close * 0.01, 1e-6)

        hurst = float(latest.get('Hurst_20', 0.50))
        if np.isnan(hurst):
            hurst = 0.50
        adx_norm = float(latest.get('ADX_Norm', 0.25))
        if np.isnan(adx_norm):
            adx_norm = 0.25
        eff_ratio = float(latest.get('Efficiency_Ratio', 0.50))
        if np.isnan(eff_ratio):
            eff_ratio = 0.50

        # --- Alpha 1: Continuous Trend & Directional Movement ---
        macd_diff_n = float(latest.get('MACD_Diff_Norm', (latest.get('MACD_Diff', 0.0) / atr)))
        ema_spread = float(latest.get('EMA12_26_Spread', (latest.get('EMA_12', close) - latest.get('EMA_26', close)) / atr))
        sma_spread = float(latest.get('SMA20_50_Spread', (latest.get('SMA_20', close) - latest.get('SMA_50', close)) / atr))
        di_spread = float(latest.get('DI_Spread', 0.0))
        slope_n = float(latest.get('EMA20_Slope_Norm', 0.0))
        a1_raw = (
            0.30 * np.tanh(macd_diff_n * 1.2)
            + 0.25 * np.tanh(ema_spread * 0.6)
            + 0.15 * np.tanh(sma_spread * 0.4)
            + 0.15 * np.tanh(di_spread * 3.0)
            + 0.15 * np.tanh(slope_n * 1.2)
        )
        a1_score = float(np.clip(a1_raw, -1.0, 1.0))

        # --- Alpha 2: Regime-Gated Mean Reversion (VWAP + Bollinger %B + StochRSI) ---
        vwap_dev = float(latest.get('VWAP_Dev', 0.0))
        bb_pctb = float(latest.get('BB_pctB', 0.5))
        stoch_rsi = float(latest.get('Stoch_RSI', 0.5))
        # Dampen mean-reversion when a strong trend (ADX > 0.30 or Hurst > 0.55) is active
        trend_dampener = float(np.clip(1.0 - max(0.0, (adx_norm - 0.25) * 1.8) - max(0.0, (hurst - 0.52) * 2.0), 0.20, 1.0))
        mr_raw = -0.45 * np.tanh(vwap_dev * 0.8) - 0.35 * np.tanh((bb_pctb - 0.5) * 2.5) - 0.20 * np.tanh((stoch_rsi - 0.5) * 2.5)
        a2_score = float(np.clip(mr_raw * trend_dampener, -1.0, 1.0))

        # --- Alpha 3: Volume Surge & Institutional Participation ---
        vol_z = float(latest.get('Volume_Z', 0.0))
        if np.isnan(vol_z):
            vol = float(latest.get('Volume', 0.0))
            vol_ma = float(df['Volume'].rolling(20, min_periods=5).mean().iloc[-1]) if 'Volume' in df.columns else 1.0
            vol_z = (vol - vol_ma) / max(vol_ma, 1e-6)
        ret_z = float(latest.get('Z_Score_Return', 0.0))
        if np.isnan(ret_z):
            ret_z = float(latest.get('Return', 0.0)) / max(atr / max(close, 1e-6), 1e-6)
        vol_amp = float(np.clip(1.0 + 0.5 * np.tanh(vol_z), 0.5, 1.8))
        a3_score = float(np.clip(np.tanh(ret_z * 0.75) * vol_amp, -1.0, 1.0))

        # --- Alpha 4: Multi-Horizon Momentum Term Structure ---
        mom3 = float(latest.get('Mom_3_ATR', 0.0))
        mom5 = float(latest.get('Mom_5_ATR', 0.0))
        mom10 = float(latest.get('Mom_10_ATR', 0.0))
        rsi_val = float(latest.get('RSI', 50.0))
        a4_raw = (
            0.35 * np.tanh(mom3 * 0.5)
            + 0.30 * np.tanh(mom5 * 0.4)
            + 0.20 * np.tanh(mom10 * 0.3)
            + 0.15 * np.tanh((rsi_val - 50.0) / 18.0)
        )
        a4_score = float(np.clip(a4_raw, -1.0, 1.0))

        # --- Alpha 5: Microstructure Order Flow & Close Location Value (CLV) ---
        clv = float(latest.get('CLV', 0.0))
        of_z = float(latest.get('Order_Flow_Z', 0.0))
        wick_imb = float(latest.get('Wick_Imbalance', 0.0))
        clv_3bar = float(df['CLV'].tail(3).mean()) if 'CLV' in df.columns else clv
        a5_score = float(np.clip(0.35 * clv + 0.25 * clv_3bar + 0.20 * np.tanh(of_z * 0.7) + 0.20 * np.tanh(wick_imb * 0.8), -1.0, 1.0))

        # --- Alpha 6: Fractional Differentiation & Hurst Persistence ---
        frac_diff = float(latest.get('FracDiff_Close', 0.0))
        if hurst >= 0.50:
            # Persistent regime: follow FracDiff memory direction
            a6_score = float(np.clip(np.tanh(frac_diff * 0.8) * (0.6 + 0.8 * eff_ratio), -1.0, 1.0))
        else:
            # Anti-persistent regime: fade extreme FracDiff extensions
            a6_score = float(np.clip(-0.5 * np.tanh(frac_diff * 0.8), -1.0, 1.0))

        alphas = {
            "alpha_1_trend": round(a1_score, 4),
            "alpha_2_mean_reversion": round(a2_score, 4),
            "alpha_3_volume_spike": round(a3_score, 4),
            "alpha_4_relative_strength": round(a4_score, 4),
            "alpha_5_order_flow": round(a5_score, 4),
            "alpha_6_hurst_regime": round(a6_score, 4),
        }

        # Dynamic Regime Weighting: boost Trend & Momentum when Hurst > 0.52 / ADX high;
        # boost Mean-Reversion & Order Flow when Hurst < 0.48
        dyn_w = dict(self.weights)
        if hurst > 0.52 or adx_norm > 0.28:
            dyn_w["alpha_1_trend"] *= 1.35
            dyn_w["alpha_4_relative_strength"] *= 1.25
            dyn_w["alpha_2_mean_reversion"] *= 0.55
        elif hurst < 0.48 and adx_norm < 0.22:
            dyn_w["alpha_2_mean_reversion"] *= 1.45
            dyn_w["alpha_5_order_flow"] *= 1.25
            dyn_w["alpha_1_trend"] *= 0.70

        w_sum = sum(dyn_w.values())
        norm_w = {k: v / w_sum for k, v in dyn_w.items()}

        pos_confirms = sum(1 for val in alphas.values() if val > 0.12)
        neg_confirms = sum(1 for val in alphas.values() if val < -0.12)

        raw_weighted = sum(norm_w[k] * alphas[k] for k in alphas)
        composite_conviction = float(np.clip(0.5 + 0.48 * raw_weighted, 0.05, 0.95))

        if pos_confirms >= self.requires_n_confirms and composite_conviction >= 0.56:
            action = "BUY"
            signal_fired = True
        elif neg_confirms >= self.requires_n_confirms and composite_conviction <= 0.44:
            action = "SELL"
            signal_fired = True
        else:
            action = "HOLD"
            signal_fired = False

        return {
            "composite_conviction": round(composite_conviction, 4),
            "n_confirms": max(pos_confirms, neg_confirms),
            "pos_confirms": pos_confirms,
            "neg_confirms": neg_confirms,
            "signal_fired": signal_fired,
            "action": action,
            "alphas": alphas,
        }


def compute_bid_ask_imbalance(df: pd.DataFrame) -> float:
    """
    Computes synthetic order book pressure: (High - Close) / ATR.
    High value indicates selling pressure at ask, low value indicates buying pressure at bid.
    """
    if df is None or len(df) == 0:
        return 0.5

    latest = df.iloc[-1]
    high = float(latest.get('High', 0.0))
    close = float(latest.get('Close', 0.0))
    atr = float(latest.get('ATR', 1.0))

    imbalance = (high - close) / (atr if atr > 0 else 1.0)
    return float(np.clip(imbalance, 0.0, 2.0))


def dynamic_stop_loss(entry_price: float, atr: float, regime: str, symbol: str = "") -> dict:
    """
    Calculates dynamic stop-loss based on market volatility regime:
    - Trending: 1.5 * ATR (tight)
    - Choppy:   2.5 * ATR (wide)
    - High Vol: 3.0 * ATR (very wide)
    """
    from config.market_catalog import smart_round
    regime_str = str(regime).upper()
    if "BULL" in regime_str or "BEAR" in regime_str:
        multiplier = 1.5
    elif "HIGH" in regime_str or "EXCITATION" in regime_str:
        multiplier = 3.0
    else:
        multiplier = 2.5

    stop_dist = multiplier * (atr if atr > 0 else entry_price * 0.02)
    buy_stop = max(1e-5, entry_price - stop_dist)
    sell_stop = entry_price + stop_dist

    return {
        "multiplier": multiplier,
        "stop_distance": smart_round(stop_dist, symbol),
        "buy_stop_loss": smart_round(buy_stop, symbol),
        "sell_stop_loss": smart_round(sell_stop, symbol),
    }


class CooldownTracker:
    def __init__(self, cooldown_bars: int = 3):
        self.cooldown_bars = cooldown_bars
        self.current_cooldown = 0

    def trigger_cooldown(self):
        self.current_cooldown = self.cooldown_bars

    def update_bar(self):
        if self.current_cooldown > 0:
            self.current_cooldown -= 1

    def is_active(self) -> bool:
        return self.current_cooldown > 0
