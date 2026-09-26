"""
=============================================================
 MULTI-ALPHA COMPOSITE ENGINE & DYNAMIC RISK CONTROL
 Step 6 & Step 7 Implementation (Trading App)
=============================================================
"""

import numpy as np
import pandas as pd

class MultiAlphaEngine:
    def __init__(self, requires_n_confirms: int = 2):
        self.requires_n_confirms = requires_n_confirms
        self.weights = {
            "alpha_1_trend": 0.30,
            "alpha_2_mean_reversion": 0.20,
            "alpha_3_volume_spike": 0.30,
            "alpha_4_relative_strength": 0.20
        }

    def compute_alphas(self, df: pd.DataFrame) -> dict:
        """
        Computes 4 independent alpha signals and aggregates weighted conviction.
        """
        if df is None or len(df) < 30:
            return {
                "composite_conviction": 0.5,
                "n_confirms": 0,
                "signal_fired": False,
                "action": "HOLD",
                "alphas": {}
            }

        latest = df.iloc[-1]
        
        # Alpha 1: Trend Momentum
        # MACD > MACD_Signal and Close > SMA_20 > SMA_50
        macd = latest.get('MACD', 0.0)
        macd_sig = latest.get('MACD_Signal', 0.0)
        close = latest.get('Close', 0.0)
        sma20 = latest.get('SMA_20', close)
        sma50 = latest.get('SMA_50', close)
        
        a1_score = 1.0 if (macd > macd_sig and close > sma20 > sma50) else (-1.0 if (macd < macd_sig and close < sma20 < sma50) else 0.0)

        # Alpha 2: Mean Reversion
        # Z-score of (Close - VWAP) / ATR
        vwap = latest.get('VWAP', close)
        atr = latest.get('ATR', 1.0)
        z_score = (close - vwap) / (atr if atr > 0 else 1.0)
        a2_score = 1.0 if z_score < -1.5 else (-1.0 if z_score > 1.5 else 0.0)

        # Alpha 3: Volume Spike
        # Volume > 2x 20-bar rolling average AND price up
        vol = latest.get('Volume', 0.0)
        vol_ma = df['Volume'].rolling(20).mean().iloc[-1] if 'Volume' in df.columns else 1.0
        ret = latest.get('Return', 0.0)
        
        a3_score = 1.0 if (vol > 2.0 * vol_ma and ret > 0) else (-1.0 if (vol > 2.0 * vol_ma and ret < 0) else 0.0)

        # Alpha 4: Relative Strength vs Benchmark / RSI
        rsi = latest.get('RSI', 50.0)
        a4_score = 1.0 if rsi > 55.0 else (-1.0 if rsi < 45.0 else 0.0)

        alphas = {
            "alpha_1_trend": a1_score,
            "alpha_2_mean_reversion": a2_score,
            "alpha_3_volume_spike": a3_score,
            "alpha_4_relative_strength": a4_score
        }

        # Count positive and negative confirmations
        pos_confirms = sum(1 for val in alphas.values() if val > 0)
        neg_confirms = sum(1 for val in alphas.values() if val < 0)

        # Weighted conviction score mapped to [0.0, 1.0]
        raw_weighted = sum(self.weights[k] * alphas[k] for k in alphas)
        composite_conviction = float(np.clip(0.5 + 0.5 * raw_weighted, 0.0, 1.0))

        if pos_confirms >= self.requires_n_confirms and composite_conviction > 0.60:
            action = "BUY"
            signal_fired = True
        elif neg_confirms >= self.requires_n_confirms and composite_conviction < 0.40:
            action = "SELL"
            signal_fired = True
        else:
            action = "HOLD"
            signal_fired = False

        return {
            "composite_conviction": round(composite_conviction, 4),
            "pos_confirms": pos_confirms,
            "neg_confirms": neg_confirms,
            "signal_fired": signal_fired,
            "action": action,
            "alphas": alphas
        }

def compute_bid_ask_imbalance(df: pd.DataFrame) -> float:
    """
    Computes synthetic order book pressure: (High - Close) / ATR.
    High value indicates selling pressure at ask, low value indicates buying pressure at bid.
    """
    if df is None or len(df) == 0:
        return 0.5

    latest = df.iloc[-1]
    high = latest.get('High', 0.0)
    close = latest.get('Close', 0.0)
    atr = latest.get('ATR', 1.0)

    imbalance = (high - close) / (atr if atr > 0 else 1.0)
    return float(np.clip(imbalance, 0.0, 2.0))

def dynamic_stop_loss(entry_price: float, atr: float, regime: str) -> dict:
    """
    Calculates dynamic stop-loss based on market volatility regime:
    - Trending: 1.5 * ATR (tight)
    - Choppy:   2.5 * ATR (wide)
    - High Vol: 3.0 * ATR (very wide)
    """
    regime_str = str(regime).upper()
    if "BULL" in regime_str or "BEAR" in regime_str:
        multiplier = 1.5
    elif "HIGH" in regime_str or "EXCITATION" in regime_str:
        multiplier = 3.0
    else:
        multiplier = 2.5

    stop_dist = multiplier * (atr if atr > 0 else entry_price * 0.02)
    buy_stop = max(0.01, entry_price - stop_dist)
    sell_stop = entry_price + stop_dist

    return {
        "multiplier": multiplier,
        "stop_distance": round(stop_dist, 2),
        "buy_stop_loss": round(buy_stop, 2),
        "sell_stop_loss": round(sell_stop, 2)
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
