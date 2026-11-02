import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from config.market_catalog import (
    ASSET_CLASS_CATALOG,
    resolve_symbol,
    get_asset_metadata,
    get_price_decimals,
    smart_round,
    format_price,
    compute_asset_position_Breakdown,
)
from data_engine import fetch_data, fetch_cross_asset_covariates
from features import add_features
from models import TradingModel, run_monte_carlo, get_markov_regime
from patterns import detect_and_draw_patterns
from ml.timesfm_engine import run_timesfm_forecast, check_timesfm_system
from quant_engine import (
    build_ai_consensus_report,
    run_jump_diffusion_mc,
    compute_ai_deal_signals_and_markers,
)
import importlib
import tv_chart
importlib.reload(tv_chart)
from tv_chart import (
    ChartSpec, add_patterns, add_deal_markers, render_tv_chart, series_points, to_time,
    LightweightChartsNotBuilt,
    SOLID as LS_SOLID, DOTTED as LS_DOTTED, DASHED as LS_DASHED,
)
import portfolio as pf
from pages.execution_dashboard import render_execution_dashboard
from utils.logging_setup import get_logger

logger = get_logger("trading_app")

# Centralised signal thresholds + prediction logging.
try:
    from config.settings import SIGNAL_THRESHOLDS
    _BUY_T = float(SIGNAL_THRESHOLDS.get("buy_threshold", 0.56))
    _SELL_T = float(SIGNAL_THRESHOLDS.get("sell_threshold", 0.44))
except Exception:
    _BUY_T, _SELL_T = 0.56, 0.44

DEFAULT_COMPONENT_WEIGHTS = {
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
}

try:
    from metrics.validation import PredictionTracker
    _HAS_TRACKER = True
except Exception:
    _HAS_TRACKER = False

st.set_page_config(
    page_title="Universal AI Trading Copilot — Natural Gas, Gold, Forex, Crypto & Stocks",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sleek modern dark UI styling designed for instant visual clarity
st.markdown(
    """
<style>
    .copilot-hero {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.96));
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35);
    }
    .verdict-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        font-weight: 800;
        font-size: 0.95rem;
        letter-spacing: 0.4px;
        margin-bottom: 8px;
    }
    .pill-buy { background: rgba(0, 230, 118, 0.18); color: #00E676; border: 1px solid #00E676; }
    .pill-sell { background: rgba(255, 82, 82, 0.18); color: #FF5252; border: 1px solid #FF5252; }
    .pill-wait { background: rgba(255, 213, 79, 0.18); color: #FFD54F; border: 1px solid #FFD54F; }
    .asset-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        font-size: 0.78rem;
        font-weight: 700;
        margin-left: 8px;
    }
    .step-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 10px;
        padding: 12px 14px;
        height: 100%;
    }
    .step-label {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #94A3B8;
        font-weight: 600;
    }
    .step-value {
        font-size: 1.22rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 2px;
    }
    .step-sub {
        font-size: 0.77rem;
        color: #CBD5E1;
        margin-top: 2px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS: FUTURE PROJECTIONS & TRADE EXECUTIONS
# ============================================================

def get_future_dates(last_date, interval: str, n_bars: int = 15):
    """Generates future timestamps extending past the last historical candle."""
    step_map = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
        "1wk": timedelta(weeks=1),
    }
    step = step_map.get(interval, timedelta(days=1))
    return [last_date + step * (i + 1) for i in range(n_bars)]


def calculate_position_size(cash: float, entry: float, stop_loss: float, risk_pct: float = 0.02) -> float:
    """
    Calculates the total USD position size based on a fixed percentage of cash risk (default 2%).
    Capped at 25% of available cash for capital safety.
    """
    risk_amount = cash * risk_pct
    price_diff = abs(entry - stop_loss)
    if price_diff <= 0 or entry <= 0:
        return 0.0

    units = risk_amount / price_diff
    raw_usd = units * entry
    return float(min(raw_usd, cash * 0.25))


def log_prediction(symbol: str, interval: str, prediction: dict, df_feat: pd.DataFrame):
    """
    Logs the current forward-looking prediction to the on-disk tracker so drift
    detection can run later, and resolves the previous pending prediction's
    actual outcome once a new bar has arrived.
    """
    if not _HAS_TRACKER:
        return
    try:
        last_ts = df_feat.index[-1]
        if last_ts == st.session_state.get("last_logged_ts"):
            return
        tracker = PredictionTracker(symbol, interval)
        pending = st.session_state.get("pending_prediction")
        if pending and (pending.get("symbol") != symbol or pending.get("interval") != interval):
            st.session_state.pop("pending_prediction", None)
            st.session_state.pop("last_logged_ts", None)
        if pending and pending.get("symbol") == symbol and pending.get("interval") == interval:
            if len(df_feat) >= 2:
                actual = float(df_feat["Close"].iloc[-1]) > float(df_feat["Close"].iloc[-2])
                tracker.log_prediction(pending["signal"], pending["prob_up"], actual_result=actual)
        tracker.log_prediction(
            prediction.get("signal", "NEUTRAL"),
            float(prediction.get("prob_up", 0.5)),
            actual_result=None,
        )
        st.session_state["pending_prediction"] = {
            "symbol": symbol,
            "interval": interval,
            "ts": last_ts,
            "signal": prediction.get("signal", "NEUTRAL"),
            "prob_up": float(prediction.get("prob_up", 0.5)),
        }
        st.session_state["last_logged_ts"] = last_ts
    except Exception as e:
        st.session_state["prediction_log_error"] = str(e)


def compute_trade_plan(
    last_close: float,
    atr: float,
    prediction: dict,
    regime: dict,
    detected_patterns: dict,
    timesfm_res: dict = None,
    consensus_bias: str = None,
    symbol: str = "",
    poc_val: float = None,
):
    """
    Builds a quantile-aware trade plan combining harmonic patterns, TimesFM-3
    9-decile prediction intervals, dynamic ATR stops, Volume POC boundaries,
    and asset-adaptive precision.
    """
    sig = prediction.get("signal", "NEUTRAL")
    prob_up = float(prediction.get("prob_up", 0.5))

    pattern_name = None
    pat_tp, pat_sl = None, None

    # Only treat patterns as actionable entry signals if they triggered recently (<= 8 bars)
    max_active_pat_bars = 8

    # 1. Establish Master Direction Bias: Consensus first (fused 7 AI models), then classifier signal, then patterns
    if consensus_bias in ("BUY", "SELL"):
        bias = consensus_bias
    elif "BUY" in sig or prob_up > _BUY_T:
        bias = "BUY"
    elif "SELL" in sig or prob_up < _SELL_T:
        bias = "SELL"
    elif "bearish_bat" in detected_patterns and detected_patterns["bearish_bat"].get("bars_ago", 999) <= max_active_pat_bars:
        bias = "SELL"
    elif "cup_handle" in detected_patterns and detected_patterns["cup_handle"].get("bars_ago", 999) <= max_active_pat_bars:
        bias = "SELL"
    elif "wolfe_wave" in detected_patterns and detected_patterns["wolfe_wave"].get("bars_ago", 999) <= max_active_pat_bars:
        bias = "BUY"
    elif "falling_wedge" in detected_patterns and detected_patterns["falling_wedge"].get("bars_ago", 999) <= max_active_pat_bars:
        bias = "BUY"
    else:
        bias = "NEUTRAL"

    # 2. Extract Confluent Pattern Targets only if they align with the master direction
    if bias == "SELL":
        if "bearish_bat" in detected_patterns and detected_patterns["bearish_bat"].get("bars_ago", 999) <= max_active_pat_bars:
            pattern_name = "Bearish Bat (Harmonic XABCD)"
            pat = detected_patterns["bearish_bat"]
            pat_tp = pat.get("target")
            pat_sl = pat.get("stop")
        elif "cup_handle" in detected_patterns and detected_patterns["cup_handle"].get("bars_ago", 999) <= max_active_pat_bars:
            pattern_name = "Inverted Cup & Handle"
            pat = detected_patterns["cup_handle"]
            pat_tp = pat.get("target")
            pat_sl = pat.get("support")
    elif bias == "BUY":
        if "wolfe_wave" in detected_patterns and detected_patterns["wolfe_wave"].get("bars_ago", 999) <= max_active_pat_bars:
            pattern_name = "Bullish Wolfe Wave"
            pat = detected_patterns["wolfe_wave"]
            pat_tp = pat.get("p6", [None, None])[1] if "p6" in pat else None
            pat_sl = pat.get("p5", [None, None])[1] if "p5" in pat else None
        elif "falling_wedge" in detected_patterns and detected_patterns["falling_wedge"].get("bars_ago", 999) <= max_active_pat_bars:
            pattern_name = "Bullish Falling Wedge"
            pat = detected_patterns["falling_wedge"]
            pat_tp = pat.get("target")
            pat_sl = pat.get("stop")

    regime_name = regime.get("regime", "CHOPPY") if isinstance(regime, dict) else str(regime)
    if "BULL" in regime_name or "BEAR" in regime_name:
        sl_multiplier = 1.5
    else:
        sl_multiplier = 2.2

    risk_dist = sl_multiplier * (atr if atr > 0 else last_close * 0.015)
    # Guard against ultra-tiny ATR on quiet bars: enforce realistic minimum risk distance
    min_dist = max(1.5 * (atr if atr > 0 else 0), last_close * 0.0075)
    risk_dist = max(risk_dist, min_dist)
    rr_target_ratio = 2.0

    tfm_q70, tfm_q30 = None, None
    if timesfm_res and timesfm_res.get("q70") and timesfm_res.get("q30"):
        tfm_q70 = float(timesfm_res["q70"][-1])
        tfm_q30 = float(timesfm_res["q30"][-1])

    if bias == "SELL":
        entry = last_close
        raw_sl = pat_sl if pat_sl and pat_sl > entry else (entry + risk_dist)
        # Protect Stop-Loss from being swallowed inside high-churn Volume POC zone
        if poc_val and float(poc_val) > entry and raw_sl <= float(poc_val) + 0.3 * risk_dist:
            raw_sl = float(poc_val) + 0.5 * (atr if atr > 0 else risk_dist)

        default_tp = entry - (rr_target_ratio * abs(raw_sl - entry))
        if pat_tp and pat_tp < entry:
            raw_tp = pat_tp
        elif tfm_q30 and tfm_q30 < entry - 0.5 * risk_dist:
            raw_tp = 0.5 * default_tp + 0.5 * tfm_q30
        else:
            raw_tp = default_tp
        raw_tp1 = entry - (1.0 * abs(raw_sl - entry))
    elif bias == "BUY":
        entry = last_close
        raw_sl = pat_sl if pat_sl and pat_sl < entry else (entry - risk_dist)
        # Protect Stop-Loss from being swallowed inside high-churn Volume POC zone
        if poc_val and float(poc_val) < entry and raw_sl >= float(poc_val) - 0.3 * risk_dist:
            raw_sl = float(poc_val) - 0.5 * (atr if atr > 0 else risk_dist)

        default_tp = entry + (rr_target_ratio * abs(entry - raw_sl))
        if pat_tp and pat_tp > entry:
            raw_tp = pat_tp
        elif tfm_q70 and tfm_q70 > entry + 0.5 * risk_dist:
            raw_tp = 0.5 * default_tp + 0.5 * tfm_q70
        else:
            raw_tp = default_tp
        raw_tp1 = entry + (1.0 * abs(entry - raw_sl))
    else:
        entry = last_close
        raw_sl = entry - risk_dist
        raw_tp = entry + risk_dist
        raw_tp1 = raw_tp
        raw_tp1 = raw_tp

    stop_loss = smart_round(raw_sl, symbol)
    take_profit = smart_round(raw_tp, symbol)
    take_profit_tp1 = smart_round(raw_tp1, symbol)
    entry_rounded = smart_round(entry, symbol)

    profit_pct = abs(take_profit - entry) / entry * 100 if entry > 0 else 0.0
    risk_pct = abs(entry - stop_loss) / entry * 100 if entry > 0 else 0.0
    rr_actual = profit_pct / risk_pct if risk_pct > 0 else (2.0 if bias != "NEUTRAL" else 1.0)

    return {
        "bias": bias,
        "pattern_name": pattern_name,
        "entry": entry_rounded,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "take_profit_tp1": take_profit_tp1,
        "profit_pct": profit_pct,
        "risk_pct": risk_pct,
        "rr_ratio": rr_actual,
        "atr": atr,
    }


# ============================================================
# SIDEBAR CONFIGURATION: UNIVERSAL ASSET CLASS SELECTOR
# ============================================================

st.sidebar.markdown("## 🌍 Universal AI Trading Copilot")
st.sidebar.caption("Trade **Natural Gas, Gold, Oil, Forex, Bitcoin & Global Stocks** with **Google TimesFM-3**")

asset_categories = list(ASSET_CLASS_CATALOG.keys()) + ["🔍 Search Any Global Market / Ticker"]
selected_category = st.sidebar.selectbox(
    "1. Select Asset Class",
    options=asset_categories,
    index=0,
    key="asset_category_select",
)

if selected_category == "🔍 Search Any Global Market / Ticker":
    raw_query = st.sidebar.text_input(
        "2. Type Asset Name or Ticker",
        value="natural gas",
        help="Type 'natural gas', 'gold', 'silver', 'crude oil', 'eur/usd', 'gbp/usd', 'usd/inr', 'bitcoin', or any Yahoo Finance symbol.",
        key="custom_symbol_query",
    )
    symbol = resolve_symbol(raw_query)
    resolved_meta = get_asset_metadata(symbol)
    st.sidebar.caption(f"✅ Resolved to: **{resolved_meta['name']} (`{symbol}`)**")
else:
    cat_instruments = ASSET_CLASS_CATALOG[selected_category]
    instrument_label = st.sidebar.selectbox(
        "2. Choose Market Instrument",
        options=list(cat_instruments.keys()),
        index=0,
        key=f"inst_select_{selected_category}",
    )
    symbol = cat_instruments[instrument_label]["symbol"]

asset_meta = get_asset_metadata(symbol)

timeframe_options = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk"]
saved_tf = st.session_state.get("active_interval", "1d")
tf_idx = timeframe_options.index(saved_tf) if saved_tf in timeframe_options else 6

interval = st.sidebar.selectbox("3. Candle Timeframe", options=timeframe_options, index=tf_idx, key="interval_select")

period_map = {
    "1m": "7d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "1h": "730d",
    "4h": "730d",
    "1d": "max",
    "1wk": "max",
}
period = period_map.get(interval, "max")

forecast_horizon = st.sidebar.slider(
    "🔮 TimesFM-3 Forecast Horizon (Bars)",
    min_value=5,
    max_value=30,
    value=15,
    step=5,
    help="Number of future candles TimesFM-3 predicts in a single pass with 9 quantiles (q10..q90).",
    key="tfm_horizon_slider",
)

enable_cross_asset = st.sidebar.checkbox(
    "🌐 Cross-Asset Macro Drivers (DXY, Oil, Gold, Yields)",
    value=True,
    help="Feeds live correlated macro assets (e.g., Crude Oil & DXY for Natural Gas; DXY & Yields for Gold/Forex; ETH & Nasdaq for BTC) into TimesFM-3's Cross-Variate Attention.",
    key="enable_cross_asset_chk",
)

fetch_clicked = st.sidebar.button("🚀 Analyze Market Now", type="primary", use_container_width=True, key="btn_fetch_analyze")
retrain_ai = st.sidebar.checkbox("🔄 Force Retrain ML Models", value=False, key="retrain_ai_chk")

st.sidebar.divider()

# Chart & Forecast Overlays
st.sidebar.subheader("📈 Chart & AI Overlays", divider=False)
show_future_forecast = st.sidebar.checkbox(
    "🔮 TimesFM-3 Forecast Cone (q10–q90)",
    value=True,
    help="Draws TimesFM-3's median forecast trajectory, 80% confidence cone, and Take-Profit / Stop-Loss zones on the chart.",
    key="show_future_forecast",
)
show_poc_line = st.sidebar.checkbox(
    "🏛️ Institutional Volume POC Line",
    value=True,
    help="Highlights the Point of Control (price level with highest institutional traded volume / tick activity).",
    key="show_poc_line",
)
show_deal_markers = st.sidebar.checkbox(
    "📍 AI Buy/Sell & Deal Close Markers",
    value=True,
    help="Displays visual arrows and target badges on candles showing exact entry points (When to Buy/Sell) and deal exit points (When to Close the Deal).",
    key="show_deal_markers",
)

ind_col1, ind_col2 = st.sidebar.columns(2)
with ind_col1:
    show_sma20 = st.checkbox("SMA-20", value=True, key="chk_sma20")
    show_bb = st.checkbox("Bollinger", value=True, key="chk_bb")
with ind_col2:
    show_sma50 = st.checkbox("SMA-50", value=True, key="chk_sma50")
    show_sma200 = st.checkbox("SMA-200", value=False, key="chk_sma200")

subplot_choice = st.sidebar.selectbox(
    "Lower Indicator Pane",
    options=["Volume", "MACD (12,26,9)", "RSI (14)", "None"],
    index=0,
    key="subplot_select",
)

# Progressive Disclosure: Advanced Chart, Pattern & Auto-Refresh Settings in clean expanders
with st.sidebar.expander("📐 Harmonic & Chart Patterns", expanded=False):
    pat_col1, pat_col2 = st.columns(2)
    with pat_col1:
        show_wolfe_wave = st.checkbox("🌊 Wolfe Wave", value=True, key="pat_wolfe")
        show_bat_pattern = st.checkbox("🦇 Bearish Bat", value=True, key="pat_bat")
    with pat_col2:
        show_cup_handle = st.checkbox("☕ Inv. Cup", value=True, key="pat_cup")
        show_falling_wedge = st.checkbox("📐 Wedge", value=True, key="pat_fw")

with st.sidebar.expander("⚙️ View, Zoom & Auto-Refresh", expanded=False):
    auto_refresh = st.checkbox("⚡ Live Auto-Refresh", value=False, key="auto_refresh_chk")
    if auto_refresh:
        refresh_interval = st.selectbox(
            "Refresh Interval", options=["10s", "30s", "1m", "5m"], index=1, key="refresh_interval_select"
        )
        run_every_val = refresh_interval
    else:
        run_every_val = None

    zoom_options = [
        "All Data (Full History)",
        "Last 2000 Bars",
        "Last 1000 Bars",
        "Last 500 Bars",
        "Last 200 Bars",
        "Last 100 Bars",
    ]
    selected_zoom = st.selectbox("Visible Candle Window", options=zoom_options, index=0, key="zoom_selector")

    chart_height = st.selectbox(
        "Chart Height", options=["Standard (680px)", "Large (880px)", "Full (1080px)"], index=0, key="chart_height_select"
    )
    height_map = {"Standard (680px)": 680, "Large (880px)": 880, "Full (1080px)": 1080}
    chart_h = height_map.get(chart_height, 680)

    chart_timezone = st.selectbox(
        "Timezone Display",
        options=[
            "IST (India Standard Time - UTC+5:30)",
            "Local Time (Browser / Device)",
            "UTC (Market Time)",
            "EST (New York)",
        ],
        index=0,
        key="chart_timezone_select",
    )
    tz_mode_map = {
        "IST (India Standard Time - UTC+5:30)": "IST",
        "Local Time (Browser / Device)": "Local",
        "UTC (Market Time)": "UTC",
        "EST (New York)": "EST",
    }
    chart_tz_mode = tz_mode_map.get(chart_timezone, "IST")

    zoom_mode = st.radio(
        "Zoom Behaviour",
        options=["Both axes (wheel)", "Horizontal only (lock Y)", "Vertical only (lock X)"],
        index=0,
    )
    st.session_state.setdefault("zoom_gen", 1)
    if st.button("⟲ Reset Chart Zoom", use_container_width=True, key="reset_zoom_btn"):
        st.session_state["zoom_gen"] += 1

_zoom_controls = (selected_zoom, zoom_mode, chart_h, chart_tz_mode, symbol, str(interval))
if st.session_state.get("zoom_controls_sig") != _zoom_controls:
    st.session_state["zoom_gen"] += 1
    st.session_state["zoom_controls_sig"] = _zoom_controls
zoom_gen = st.session_state.get("zoom_gen", 1)


# ============================================================
# DATA FETCHING & AI PIPELINE EXECUTION
# ============================================================

def _run_full_analysis(
    sym: str,
    pd_str: str,
    intv: str,
    horizon: int,
    force_retrain: bool = False,
    use_cross_asset: bool = True,
):
    df = fetch_data(sym, period=pd_str, interval=intv)
    if df is None or len(df) == 0:
        return False

    df_feat = add_features(df)
    model = TradingModel(sym, intv)
    model.train_or_load(df_feat, force_retrain=force_retrain)
    prediction = model.predict_next(df_feat, buy_threshold=_BUY_T, sell_threshold=_SELL_T)
    regime = get_markov_regime(df_feat)

    # Fetch aligned Cross-Asset Multivariate Macro Drivers (Oil, Gold, DXY, Yields, ETH)
    cross_covs = {}
    if use_cross_asset:
        cross_covs = fetch_cross_asset_covariates(sym, df_feat, period=pd_str, interval=intv)

    future_dates = get_future_dates(df.index[-1], intv, n_bars=horizon)
    timesfm_res = run_timesfm_forecast(
        df_feat,
        symbol=sym,
        interval=intv,
        horizon=horizon,
        future_timestamps=future_dates,
        cross_asset_covariates=cross_covs,
    )

    log_prediction(sym, intv, prediction, df_feat)

    st.session_state["df"] = df
    st.session_state["df_feat"] = df_feat
    st.session_state["prediction"] = prediction
    st.session_state["regime"] = regime
    st.session_state["timesfm_res"] = timesfm_res
    st.session_state["active_symbol"] = sym
    st.session_state["active_interval"] = intv
    st.session_state["active_horizon"] = horizon
    st.session_state["active_cross_asset"] = use_cross_asset
    return True


symbol_changed = st.session_state.get("active_symbol") != symbol
interval_changed = st.session_state.get("active_interval") != interval
horizon_changed = st.session_state.get("active_horizon") != forecast_horizon
cross_changed = st.session_state.get("active_cross_asset") != enable_cross_asset

if fetch_clicked or ("df" not in st.session_state) or symbol_changed or interval_changed or horizon_changed or cross_changed:
    with st.spinner(
        f"🧠 Running TimesFM-3 Multivariate & 7-Model AI Pipeline on {asset_meta['name']} ({symbol} · {interval})..."
    ):
        ok = _run_full_analysis(
            symbol,
            period,
            interval,
            forecast_horizon,
            force_retrain=retrain_ai,
            use_cross_asset=enable_cross_asset,
        )
    if not ok and (fetch_clicked or symbol_changed or interval_changed):
        st.error(f"Failed to fetch data for {symbol} on {interval}. Check symbol or Yahoo Finance connection.")
        st.stop()


# ============================================================
# AUTO-REFRESH DASHBOARD FRAGMENT
# ============================================================

@st.fragment(run_every=run_every_val)
def render_dashboard():
    """Auto-refresh dashboard data smoothly while preserving chart zoom state."""
    if auto_refresh and "df" in st.session_state and not fetch_clicked:
        _run_full_analysis(
            symbol,
            period,
            interval,
            forecast_horizon,
            force_retrain=False,
            use_cross_asset=enable_cross_asset,
        )

    if "df" not in st.session_state:
        st.info("Click **🚀 Analyze Market Now** in the sidebar to load data.")
        return

    df = st.session_state["df"]
    df_feat = st.session_state["df_feat"]
    prediction = st.session_state.get("prediction", {"signal": "N/A", "prob_up": 0.5})
    regime = st.session_state.get("regime", {"regime": "CHOPPY", "markov_score": 0.5})
    timesfm_res = st.session_state.get("timesfm_res") or run_timesfm_forecast(
        df_feat, symbol=symbol, interval=interval, horizon=forecast_horizon
    )

    last_close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
    pct_change = ((last_close - prev_close) / prev_close * 100) if prev_close != 0 else 0.0
    atr = (
        float(df_feat["ATR"].iloc[-1])
        if "ATR" in df_feat.columns and not np.isnan(df_feat["ATR"].iloc[-1])
        else (last_close * 0.015)
    )

    # Timezone Formatting
    last_ts = df.index[-1]
    if getattr(last_ts, "tzinfo", None) is not None:
        last_ts_utc = last_ts.tz_convert("UTC")
    else:
        last_ts_utc = pd.Timestamp(last_ts).tz_localize("UTC")
    utc_str = last_ts_utc.strftime("%H:%M UTC")
    try:
        ist_str = last_ts_utc.tz_convert("Asia/Kolkata").strftime("%H:%M IST")
    except Exception:
        ist_str = utc_str

    # Zoom Windowing & Database Span Metrics
    data_len = len(df)
    zoom_map = {
        "All Data (Full History)": data_len,
        "Last 2000 Bars": 2000,
        "Last 1000 Bars": 1000,
        "Last 500 Bars": 500,
        "Last 200 Bars": 200,
        "Last 100 Bars": 100,
    }
    zoom_bars = zoom_map.get(selected_zoom, data_len)
    df_zoom = df.iloc[-zoom_bars:].copy() if (zoom_bars < data_len and selected_zoom != "All Data (Full History)") else df.copy()
    df_feat_zoom = df_feat.iloc[-zoom_bars:].copy() if (zoom_bars < data_len and selected_zoom != "All Data (Full History)") else df_feat.copy()

    start_dt = df.index[0]
    end_dt = df.index[-1]
    days_span = (end_dt - start_dt).days
    if days_span >= 365:
        span_label = f"{days_span / 365.25:.1f}Y ({start_dt.strftime('%Y')}–{end_dt.strftime('%Y')})"
    else:
        span_label = f"{days_span}d ({start_dt.strftime('%b %d')}–{end_dt.strftime('%b %d')})"

    # Harmonic & Geometric Pattern Detection (cached per bar)
    _pat_cache_key = (
        str(df.index[-1]),
        len(df),
        show_wolfe_wave,
        show_bat_pattern,
        show_cup_handle,
        show_falling_wedge,
    )
    if st.session_state.get("_pat_cache_key") == _pat_cache_key and "_cached_patterns" in st.session_state:
        detected_patterns = st.session_state["_cached_patterns"]
    else:
        detected_patterns = detect_and_draw_patterns(
            None,
            df_feat_zoom,
            has_sub=False,
            show_wolfe=show_wolfe_wave,
            show_cup=show_cup_handle,
            show_bat=show_bat_pattern,
            show_wedge=show_falling_wedge,
        )
        st.session_state["_cached_patterns"] = detected_patterns
        st.session_state["_pat_cache_key"] = _pat_cache_key

    # Portfolio cash for position sizing
    pf_summary = pf.get_portfolio_summary({symbol: last_close})
    cash = float(pf_summary.get("cash", 100000.0))

    # Build Unified 7-Engine AI Consensus Report
    consensus = build_ai_consensus_report(
        df_feat=df_feat,
        symbol=symbol,
        interval=interval,
        xgb_prediction=prediction,
        regime=regime,
        detected_patterns=detected_patterns,
        timesfm_res=timesfm_res,
        total_capital=cash,
    )

    # Compute Quantile-Aware Trade Plan with Asset-Adaptive Precision
    trade_plan = compute_trade_plan(
        last_close,
        atr,
        prediction,
        regime,
        detected_patterns,
        timesfm_res=timesfm_res,
        consensus_bias=consensus.get("direction_bias"),
        symbol=symbol,
        poc_val=consensus.get("volume_profile", {}).get("poc"),
    )
    deal_signals = compute_ai_deal_signals_and_markers(
        df=df,
        df_feat=df_feat,
        trade_plan=trade_plan,
        consensus=consensus,
        symbol=symbol,
        lookback_bars=min(120, len(df)),
    )
    consensus["deal_signals"] = deal_signals
    bias = trade_plan["bias"]
    tp_price = trade_plan["take_profit"]
    tp1_price = trade_plan["take_profit_tp1"]
    sl_price = trade_plan["stop_loss"]
    profit_pct = trade_plan["profit_pct"]
    risk_pct = trade_plan["risk_pct"]
    rr_ratio = trade_plan["rr_ratio"]
    pos_size_usd = calculate_position_size(cash, last_close, sl_price, risk_pct=0.02)
    pos_breakdown = compute_asset_position_Breakdown(symbol, last_close, sl_price, pos_size_usd)

    # ============================================================
    # HERO SECTION: 3-SECOND PLAIN-ENGLISH AI COPILOT VERDICT
    # ============================================================
    verdict = consensus.get("verdict", "WAIT / HOLD")
    conv_pct = consensus.get("directional_conviction", 0.5) * 100.0
    pill_class = "pill-buy" if "BUY" in verdict else ("pill-sell" if "SELL" in verdict else "pill-wait")
    pill_icon = "🟢" if "BUY" in verdict else ("🔴" if "SELL" in verdict else "🟡")

    tfm_end_price = timesfm_res["q50"][-1] if timesfm_res.get("q50") else last_close
    tfm_exp_pct = timesfm_res.get("expected_return_pct", 0.0)

    entry_price = trade_plan.get("entry", last_close)
    fmt_entry = format_price(entry_price, symbol)
    fmt_last = format_price(last_close, symbol)
    fmt_tfm_end = format_price(tfm_end_price, symbol)
    fmt_tp = format_price(tp_price, symbol)
    fmt_tp1 = format_price(tp1_price, symbol)
    fmt_sl = format_price(sl_price, symbol)

    st.markdown(
        f"""
        <div class="copilot-hero">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                <div>
                    <span class="verdict-pill {pill_class}">{pill_icon} AI VERDICT: {verdict} ({conv_pct:.1f}% CONFIDENCE)</span>
                    <span class="asset-badge">{asset_meta['asset_class']}</span>
                    <span class="asset-badge" style="background:rgba(56, 189, 248, 0.15); color:#38BDF8; border-color:rgba(56, 189, 248, 0.35);">
                        🗄️ DB: {len(df):,} Bars ({span_label})
                    </span>
                    <span style="color:#94A3B8; font-size:0.85rem; margin-left:10px;">
                        <b>{asset_meta['name']} ({symbol})</b> · {interval} · Latest Bar: {ist_str} ({utc_str})
                    </span>
                </div>
                <div style="font-size:1.38rem; font-weight:800; color:#F8FAFC;">
                    {fmt_last}
                    <span style="font-size:0.95rem; color:{'#00E676' if pct_change >= 0 else '#FF5252'};">
                        ({pct_change:+.2f}%)
                    </span>
                </div>
            </div>
            <div style="margin-top:8px; font-size:0.98rem; color:#E2E8F0; line-height:1.45;">
                {consensus.get('explanation', '')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 5-Step Simple Action Strip (Works for Natural Gas, Gold, Forex, Crypto & Stocks)
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        agree_str = f"{consensus.get('buy_votes', 0)} Bull · {consensus.get('sell_votes', 0)} Bear"
        oos_skill_pct = float(consensus.get("composite_oos_accuracy", 0.65)) * 100.0
        st.markdown(
            f"""
            <div class="step-card">
                <div class="step-label">1. AI Consensus</div>
                <div class="step-value" style="color:{'#00E676' if bias=='BUY' else ('#FF5252' if bias=='SELL' else '#FFD54F')};">
                    {verdict}
                </div>
                <div class="step-sub">7 Engines ({agree_str}) · OOS Skill: {oos_skill_pct:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            f"""
            <div class="step-card">
                <div class="step-label">2. TimesFM-3 (+{forecast_horizon} Bars)</div>
                <div class="step-value" style="color:{'#00E676' if tfm_exp_pct>=0 else '#FF5252'};">
                    {fmt_tfm_end}
                </div>
                <div class="step-sub">Projected Move: {tfm_exp_pct:+.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            f"""
            <div class="step-card">
                <div class="step-label">3. Take-Profit Target</div>
                <div class="step-value" style="color:#00E676;">{fmt_tp}</div>
                <div class="step-sub">+{profit_pct:.2f}% (TP1: {fmt_tp1})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s4:
        st.markdown(
            f"""
            <div class="step-card">
                <div class="step-label">4. Safety Stop-Loss</div>
                <div class="step-value" style="color:#FF5252;">{fmt_sl}</div>
                <div class="step-sub">-{risk_pct:.2f}% Max Price Risk</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s5:
        st.markdown(
            f"""
            <div class="step-card">
                <div class="step-label">5. Safe Position Size</div>
                <div class="step-value" style="color:#38BDF8;">{pos_breakdown['primary_size_str']}</div>
                <div class="step-sub">{pos_breakdown['secondary_info_str']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ============================================================
    # AI DEAL EXECUTION & CLOSING BLUEPRINT (WHEN TO BUY/SELL & CLOSE)
    # ============================================================
    exec_plan = deal_signals.get("execution_plan", {})
    bp_border = "rgba(0, 230, 118, 0.40)" if bias == "BUY" else ("rgba(255, 82, 82, 0.40)" if bias == "SELL" else "rgba(255, 213, 79, 0.35)")
    bp_header_color = "#00E676" if bias == "BUY" else ("#FF5252" if bias == "SELL" else "#FFD54F")
    bp_badge_text = "🟢 ACTIONABLE BUY" if bias == "BUY" else ("🔴 ACTIONABLE SELL / SHORT" if bias == "SELL" else "🟡 PATIENT WAIT / NO DEAL")

    st.markdown(
        f"""
        <div style="background:linear-gradient(145deg, rgba(15,23,42,0.95), rgba(30,41,59,0.92)); border:1px solid {bp_border}; border-radius:14px; padding:16px 20px; margin-top:14px; margin-bottom:14px; box-shadow:0 8px 24px rgba(0,0,0,0.32);">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; border-bottom:1px solid rgba(148,163,184,0.15); padding-bottom:10px; margin-bottom:12px;">
                <div style="display:flex; align-items:center; gap:10px;">
                    <span style="font-size:1.15rem; font-weight:900; color:#F8FAFC;">🎯 AI Deal Execution & Closing Blueprint</span>
                    <span style="background:{'rgba(0,230,118,0.18)' if bias=='BUY' else ('rgba(255,82,82,0.18)' if bias=='SELL' else 'rgba(255,213,79,0.18)')}; color:{bp_header_color}; border:1px solid {bp_header_color}; padding:3px 10px; border-radius:999px; font-weight:800; font-size:0.8rem;">
                        {bp_badge_text}
                    </span>
                </div>
                <div style="font-size:0.84rem; color:#94A3B8;">
                    ⚡ <b>Log-Odds Bayesian Confluence</b> · Risk/Reward: <b style="color:#38BDF8;">1:{rr_ratio:.2f}</b> · Conviction: <b style="color:{bp_header_color};">{conv_pct:.1f}%</b>
                </div>
            </div>
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap:12px;">
                <div style="background:rgba(15,23,42,0.65); border:1px solid rgba(148,163,184,0.15); border-radius:10px; padding:12px 14px;">
                    <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.5px; color:#94A3B8; font-weight:700;">
                        {'🟢 When to Buy' if bias=='BUY' else ('🔴 When to Sell' if bias=='SELL' else '🟡 Entry Filter')}
                    </div>
                    <div style="font-size:1.1rem; font-weight:800; color:{'#00E676' if bias=='BUY' else ('#FF5252' if bias=='SELL' else '#FFD54F')}; margin-top:2px;">
                        {fmt_entry}
                    </div>
                    <div style="font-size:0.80rem; color:#E2E8F0; margin-top:4px; line-height:1.35;">
                        {exec_plan.get('entry_rule', '')}
                    </div>
                </div>
                <div style="background:rgba(15,23,42,0.65); border:1px solid rgba(0,230,118,0.25); border-radius:10px; padding:12px 14px;">
                    <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.5px; color:#00E676; font-weight:700;">
                        🎯 When to Close Deal: 100% Target
                    </div>
                    <div style="font-size:1.1rem; font-weight:800; color:#00E676; margin-top:2px;">
                        {fmt_tp} <span style="font-size:0.85rem; font-weight:600;">(+{profit_pct:.1f}%)</span>
                    </div>
                    <div style="font-size:0.80rem; color:#E2E8F0; margin-top:4px; line-height:1.35;">
                        {exec_plan.get('close_tp_rule', '')}
                    </div>
                </div>
                <div style="background:rgba(15,23,42,0.65); border:1px solid rgba(56,189,248,0.25); border-radius:10px; padding:12px 14px;">
                    <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.5px; color:#38BDF8; font-weight:700;">
                        🛡️ 50% Profit Lock & Breakeven Stop
                    </div>
                    <div style="font-size:1.1rem; font-weight:800; color:#38BDF8; margin-top:2px;">
                        {fmt_tp1}
                    </div>
                    <div style="font-size:0.80rem; color:#E2E8F0; margin-top:4px; line-height:1.35;">
                        {exec_plan.get('trailing_rule', '')}
                    </div>
                </div>
                <div style="background:rgba(15,23,42,0.65); border:1px solid rgba(255,82,82,0.25); border-radius:10px; padding:12px 14px;">
                    <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.5px; color:#FF5252; font-weight:700;">
                        🛑 When to Close Deal: Stop-Loss Exit
                    </div>
                    <div style="font-size:1.1rem; font-weight:800; color:#FF5252; margin-top:2px;">
                        {fmt_sl} <span style="font-size:0.85rem; font-weight:600;">(-{risk_pct:.1f}%)</span>
                    </div>
                    <div style="font-size:0.80rem; color:#E2E8F0; margin-top:4px; line-height:1.35;">
                        {exec_plan.get('close_sl_rule', '')}
                    </div>
                </div>
            </div>
            <div style="margin-top:10px; padding-top:8px; border-top:1px solid rgba(148,163,184,0.12); display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; font-size:0.80rem; color:#94A3B8;">
                <div>⚠️ <b>Safety Invalidation:</b> {exec_plan.get('invalidation_rule', '')}</div>
                <div>⏱️ <b>Time Horizon:</b> {exec_plan.get('timeout_rule', '')}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # ============================================================
    # 4-TAB INTUITIVE WORKSPACE
    # ============================================================
    tab_chart, tab_tfm, tab_consensus, tab_portfolio = st.tabs([
        "📈 1. Interactive Chart & AI Trajectory",
        "🔮 2. TimesFM-3 Multivariate Forecast Lab",
        "🤝 3. AI Consensus & Signal Radar (7 Models)",
        "💼 4. Paper Trading & Risk Simulator",
    ])

    # ------------------------------------------------------------
    # TAB 1: TRADINGVIEW CHART + TIMESFM-3 QUANTILE CONE OVERLAY
    # ------------------------------------------------------------
    with tab_chart:
        spec = ChartSpec()
        spec.set_candles(df_zoom)
        last_date = df_zoom.index[-1]

        def _overlay(col, color, width, title, style=LS_SOLID):
            if col in df_feat_zoom.columns:
                spec.line(
                    series_points(df_feat_zoom.index, df_feat_zoom[col]),
                    color,
                    width,
                    style,
                    title,
                    legend=True,
                )

        if show_sma20:
            _overlay("SMA_20", "#FFD54F", 1.2, "SMA 20")
        if show_sma50:
            _overlay("SMA_50", "#FF9800", 1.2, "SMA 50")
        if show_sma200:
            _overlay("SMA_200", "#7E57C2", 1.5, "SMA 200")
        if show_bb and "BB_High" in df_feat_zoom.columns and "BB_Low" in df_feat_zoom.columns:
            _overlay("BB_High", "rgba(148, 163, 184, 0.55)", 1, "BB Upper", LS_DASHED)
            _overlay("BB_Low", "rgba(148, 163, 184, 0.55)", 1, "BB Lower", LS_DASHED)

        # Institutional Volume Profile POC Price Line
        poc_val = consensus.get("volume_profile", {}).get("poc")
        if show_poc_line and poc_val:
            spec.price_line(
                float(poc_val),
                "#E040FB",
                f"🏛️ Vol POC {format_price(poc_val, symbol)}",
                LS_DOTTED,
                1,
            )

        # Harmonic & Geometric Patterns
        if detected_patterns:
            add_patterns(spec, detected_patterns, last_date)

        # AI Buy/Sell & Deal Close Markers (Entry arrows, TP targets, SL protective exits)
        if show_deal_markers and consensus.get("deal_signals"):
            add_deal_markers(spec, consensus["deal_signals"])

        # TimesFM-3 Multi-Horizon Forecast Cone + TP/SL Zones
        if show_future_forecast:
            n_proj = int(timesfm_res.get("horizon", forecast_horizon))
            future_dates = get_future_dates(df.index[-1], interval, n_bars=n_proj)
            forecast_span = [df.index[-1]] + future_dates

            spec.zone(forecast_span, last_close, tp_price, "rgba(0, 230, 118, 0.50)", "rgba(0, 230, 118, 0.10)")
            spec.zone(forecast_span, last_close, sl_price, "rgba(255, 82, 82, 0.50)", "rgba(255, 82, 82, 0.10)")

            spec.price_line(last_close, "#29B6F6", f"ENTRY {fmt_last}", LS_DOTTED, 2)
            spec.price_line(
                tp_price, "#00E676", f"🎯 TP {fmt_tp} (+{profit_pct:.1f}%)", LS_DASHED, 2
            )
            spec.price_line(sl_price, "#FF5252", f"🛑 SL {fmt_sl} (-{risk_pct:.1f}%)", LS_DASHED, 2)

            if timesfm_res and timesfm_res.get("q50"):
                q50_path = [last_close] + list(timesfm_res["q50"])
                q90_path = [last_close] + list(timesfm_res["q90"])
                q10_path = [last_close] + list(timesfm_res["q10"])
                q70_path = [last_close] + list(timesfm_res["q70"])
                q30_path = [last_close] + list(timesfm_res["q30"])

                spec.line(
                    series_points(forecast_span, q90_path),
                    "rgba(56, 189, 248, 0.45)",
                    1.0,
                    LS_DOTTED,
                    "TimesFM-3 q90 (80% Upper)",
                )
                spec.line(
                    series_points(forecast_span, q10_path),
                    "rgba(56, 189, 248, 0.45)",
                    1.0,
                    LS_DOTTED,
                    "TimesFM-3 q10 (80% Lower)",
                )
                spec.line(
                    series_points(forecast_span, q70_path),
                    "rgba(129, 140, 248, 0.65)",
                    1.1,
                    LS_DASHED,
                    "TimesFM-3 q70",
                )
                spec.line(
                    series_points(forecast_span, q30_path),
                    "rgba(129, 140, 248, 0.65)",
                    1.1,
                    LS_DASHED,
                    "TimesFM-3 q30",
                )
                spec.line(
                    series_points(forecast_span, q50_path),
                    "#38BDF8",
                    2.2,
                    LS_SOLID,
                    "🔮 TimesFM-3 Forecast (q50)",
                    legend=True,
                )

        # Secondary Subplot (Volume / MACD / RSI)
        if subplot_choice != "None":
            if subplot_choice == "Volume" and "Volume" in df_zoom.columns:
                spec.histogram(
                    [
                        {
                            "time": to_time(t),
                            "value": float(v),
                            "color": "rgba(8, 153, 129, 0.5)" if c >= o else "rgba(242, 54, 69, 0.5)",
                        }
                        for t, v, c, o in zip(
                            df_zoom.index, df_zoom["Volume"], df_zoom["Close"], df_zoom["Open"]
                        )
                        if pd.notna(v)
                    ],
                    title="Volume / Activity",
                    pane=1,
                    volume=True,
                )
            elif subplot_choice == "MACD (12,26,9)" and "MACD" in df_feat_zoom.columns:
                if "MACD_Signal" in df_feat_zoom.columns:
                    hist = df_feat_zoom["MACD"] - df_feat_zoom["MACD_Signal"]
                    spec.histogram(
                        [
                            {
                                "time": to_time(t),
                                "value": float(v),
                                "color": "rgba(8, 153, 129, 0.5)" if v >= 0 else "rgba(242, 54, 69, 0.5)",
                            }
                            for t, v in zip(hist.index, hist)
                            if pd.notna(v)
                        ],
                        title="MACD Hist",
                        pane=1,
                    )
                spec.line(
                    series_points(df_feat_zoom.index, df_feat_zoom["MACD"]),
                    "#29B6F6",
                    1.3,
                    title="MACD",
                    pane=1,
                    last_value=True,
                )
                if "MACD_Signal" in df_feat_zoom.columns:
                    spec.line(
                        series_points(df_feat_zoom.index, df_feat_zoom["MACD_Signal"]),
                        "#FF5252",
                        1.3,
                        title="Signal",
                        pane=1,
                        last_value=True,
                    )
            elif subplot_choice == "RSI (14)" and "RSI" in df_feat_zoom.columns:
                spec.line(
                    series_points(df_feat_zoom.index, df_feat_zoom["RSI"]),
                    "#AB47BC",
                    1.5,
                    title="RSI 14",
                    pane=1,
                    last_value=True,
                    price_lines=[
                        {
                            "price": 70,
                            "color": "rgba(242, 54, 69, 0.6)",
                            "lineStyle": LS_DASHED,
                            "lineWidth": 1,
                            "axisLabelVisible": True,
                            "title": "70",
                        },
                        {
                            "price": 30,
                            "color": "rgba(8, 153, 129, 0.6)",
                            "lineStyle": LS_DASHED,
                            "lineWidth": 1,
                            "axisLabelVisible": True,
                            "title": "30",
                        },
                    ],
                )

        _zoom_mode_js = {
            "Horizontal only (lock Y)": "lockY",
            "Vertical only (lock X)": "lockX",
        }.get(zoom_mode, "both")
        try:
            render_tv_chart(
                spec,
                title=f"{asset_meta['name']} ({symbol}) · {interval} [{len(df_zoom):,} Bars]",
                height=chart_h,
                intraday=interval not in ("1d", "1wk"),
                state_key=f"{symbol}|{interval}|{selected_zoom}|g{zoom_gen}|{chart_tz_mode}",
                zoom_mode=_zoom_mode_js,
                filename=f"{symbol}_{interval}",
                tz_mode=chart_tz_mode,
            )
        except LightweightChartsNotBuilt as e:
            st.error(str(e))

        # 1-Click Quick Paper Trade Bar right under the chart
        st.markdown("#### ⚡ 1-Click Paper Trade Execution (Risk-Managed)")
        q_col1, q_col2, q_col3, q_col4 = st.columns([1.5, 1, 1, 1.2])
        rec_units = pos_breakdown["recommended_qty"]
        unit_lbl = asset_meta.get("unit_label", symbol)
        with q_col1:
            st.caption(
                f"Suggested AI Size: **{pos_breakdown['primary_size_str']}** (`${pos_size_usd:,.2f}` Notional) · "
                f"Cash: **${cash:,.2f}**"
            )
        with q_col2:
            if st.button(f"🟢 Paper BUY {rec_units} {unit_lbl}", use_container_width=True, key="quick_buy_btn"):
                ok_b, msg_b = pf.buy_asset(symbol, max(rec_units, 0.0001), last_close)
                if ok_b:
                    st.toast(f"✅ {msg_b}")
                    st.rerun()
                else:
                    st.warning(msg_b)
        with q_col3:
            if st.button(f"🔴 Paper SHORT {rec_units} {unit_lbl}", use_container_width=True, key="quick_sell_btn"):
                ok_s, msg_s = pf.sell_asset(symbol, max(rec_units, 0.0001), last_close)
                if ok_s:
                    st.toast(f"✅ {msg_s}")
                    st.rerun()
                else:
                    st.warning(msg_s)
        with q_col4:
            if trade_plan["pattern_name"]:
                st.info(f"📐 Active Pattern: **{trade_plan['pattern_name']}**")
            else:
                st.caption(
                    f"🧭 Regime: **{regime.get('regime', 'N/A')}** · ATR: **{format_price(atr, symbol)}**"
                )

    # ------------------------------------------------------------
    # TAB 2: TIMESFM-3 MULTIVARIATE FORECAST LAB
    # ------------------------------------------------------------
    with tab_tfm:
        st.markdown(
            f"### 🔮 Google TimesFM-3 Multivariate Zero-Shot Forecast Lab — `{asset_meta['name']} ({symbol})`"
        )
        tfm_sys = check_timesfm_system()
        st.caption(
            f"⚡ **Active Backend:** `{tfm_sys.get('active_backend', 'TimesFM-3 Native Engine')}` (Patch Size: {tfm_sys.get('patch_size', 32)} · Quantiles: {tfm_sys.get('num_quantiles', 9)}) — "
            "TimesFM-3 tokenizes historical price patches with Reversible Instance Normalization (RevIN), "
            "applies 2D Causal Temporal + Cross-Variate Attention across technical indicators AND live **Cross-Asset Macro Drivers**, "
            "and decodes all **9 probabilistic deciles ($q_{10} \\dots q_{90}$)** in a single non-autoregressive pass."
        )

        oos = timesfm_res.get("oos_metrics", {})
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(
            f"+{forecast_horizon} Bar Median Target (q50)",
            fmt_tfm_end,
            f"{tfm_exp_pct:+.2f}%",
        )
        m2.metric(
            "80% Prediction Cone (q10–q90)",
            f"±{timesfm_res.get('uncertainty_pct', 0)*0.5:.2f}%",
            timesfm_res.get("uncertainty_level", "MODERATE"),
            delta_color="off",
        )
        m3.metric(
            "Walk-Forward OOS Hit Rate",
            f"{oos.get('directional_accuracy', 0.55)*100:.1f}%",
            f"IC: {oos.get('information_coefficient', 0.10):.2f}",
        )
        m4.metric(
            "80% Conformal Coverage",
            f"{oos.get('quantile_coverage_80', 0.80)*100:.1f}%",
            f"MAPE: {oos.get('mape_pct', 1.2):.2f}%",
        )
        m5.metric(
            "Anomaly Status",
            "⚠️ Regime Spike" if timesfm_res.get("is_anomaly") else "✅ Normal Flow",
            f"Z-Score: {timesfm_res.get('anomaly_zscore', 0.0):.2f}σ",
            delta_color="off",
        )

        # Interactive Plotly 9-Quantile Fan Chart + Covariate Attribution
        col_fan, col_attr = st.columns([1.6, 1.05])

        with col_fan:
            st.markdown("#### 📈 9-Decile Probabilistic Forecast Fan ($q_{10}$ to $q_{90}$)")
            hist_window = min(60, len(df))
            hist_df = df.iloc[-hist_window:]
            hist_times = list(hist_df.index)
            hist_closes = [float(x) for x in hist_df["Close"].values]

            n_proj = int(timesfm_res.get("horizon", forecast_horizon))
            fut_times = [hist_times[-1]] + get_future_dates(hist_times[-1], interval, n_bars=n_proj)

            q10_s = [last_close] + list(timesfm_res["q10"])
            q20_s = [last_close] + list(timesfm_res["q20"])
            q30_s = [last_close] + list(timesfm_res["q30"])
            q40_s = [last_close] + list(timesfm_res["q40"])
            q50_s = [last_close] + list(timesfm_res["q50"])
            q60_s = [last_close] + list(timesfm_res["q60"])
            q70_s = [last_close] + list(timesfm_res["q70"])
            q80_s = [last_close] + list(timesfm_res["q80"])
            q90_s = [last_close] + list(timesfm_res["q90"])

            fig_fan = go.Figure()
            fig_fan.add_trace(
                go.Scatter(
                    x=hist_times,
                    y=hist_closes,
                    mode="lines",
                    name="Historical Close",
                    line=dict(color="#E2E8F0", width=2.2),
                )
            )
            fig_fan.add_trace(
                go.Scatter(
                    x=fut_times + fut_times[::-1],
                    y=q90_s + q10_s[::-1],
                    fill="toself",
                    fillcolor="rgba(56, 189, 248, 0.12)",
                    line=dict(color="rgba(56, 189, 248, 0.30)", width=1),
                    name="80% Band (q10–q90)",
                )
            )
            fig_fan.add_trace(
                go.Scatter(
                    x=fut_times + fut_times[::-1],
                    y=q80_s + q20_s[::-1],
                    fill="toself",
                    fillcolor="rgba(56, 189, 248, 0.18)",
                    line=dict(color="rgba(56, 189, 248, 0.0)"),
                    name="60% Band (q20–q80)",
                )
            )
            fig_fan.add_trace(
                go.Scatter(
                    x=fut_times + fut_times[::-1],
                    y=q70_s + q30_s[::-1],
                    fill="toself",
                    fillcolor="rgba(56, 189, 248, 0.28)",
                    line=dict(color="rgba(129, 140, 248, 0.55)", width=1, dash="dot"),
                    name="40% Core Band (q30–q70)",
                )
            )
            fig_fan.add_trace(
                go.Scatter(
                    x=fut_times + fut_times[::-1],
                    y=q60_s + q40_s[::-1],
                    fill="toself",
                    fillcolor="rgba(56, 189, 248, 0.38)",
                    line=dict(color="rgba(56, 189, 248, 0.0)"),
                    name="20% Tight Band (q40–q60)",
                )
            )
            fig_fan.add_trace(
                go.Scatter(
                    x=fut_times,
                    y=q50_s,
                    mode="lines+markers",
                    name="TimesFM-3 Median (q50)",
                    line=dict(color="#38BDF8", width=3),
                    marker=dict(size=5, color="#38BDF8"),
                )
            )

            dec_fmt = f".{get_price_decimals(last_close, symbol)}f"
            fig_fan.update_layout(
                height=420,
                template="plotly_dark",
                margin=dict(l=20, r=20, t=25, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                xaxis_title="Time",
                yaxis_title=f"{symbol} Price",
                yaxis=dict(tickformat=dec_fmt),
            )
            st.plotly_chart(fig_fan, use_container_width=True)

        with col_attr:
            st.markdown("#### 🧠 Multivariate & Cross-Asset Attribution")
            st.caption("How much each technical & macro channel contributed to the forecast:")
            attr_dict = timesfm_res.get("covariate_attribution", {})
            if attr_dict:
                sorted_attr = sorted(attr_dict.items(), key=lambda x: x[1], reverse=True)
                labels = [k for k, _ in sorted_attr][::-1]
                vals = [round(v * 100.0, 1) for _, v in sorted_attr][::-1]

                fig_attr = go.Figure(
                    go.Bar(
                        x=vals,
                        y=labels,
                        orientation="h",
                        marker=dict(
                            color=["#00E676" if "🌐" in lbl else "#38BDF8" for lbl in labels]
                        ),
                        text=[f"{v:.1f}%" for v in vals],
                        textposition="auto",
                    )
                )
                fig_attr.update_layout(
                    height=385,
                    template="plotly_dark",
                    margin=dict(l=10, r=15, t=10, b=20),
                    xaxis_title="Attention & XReg Contribution (%)",
                )
                st.plotly_chart(fig_attr, use_container_width=True)

        # Cross-Asset Macro Drivers Table
        cross_corrs = timesfm_res.get("cross_asset_correlations", [])
        if cross_corrs:
            st.markdown(f"#### 🌐 Live Cross-Asset Macro Drivers for `{asset_meta['name']}`")
            st.caption(
                "TimesFM-3's Full Cross-Variate Attention automatically tracks these correlated global macro markets "
                "to detect lead-lag spillovers:"
            )
            st.dataframe(pd.DataFrame(cross_corrs), use_container_width=True, hide_index=True)

        st.markdown("#### 📋 Multi-Horizon Quantile Forecast Schedule")
        h_table = timesfm_res.get("horizon_table", [])
        if h_table:
            st.dataframe(pd.DataFrame(h_table), use_container_width=True, hide_index=True)

    # ------------------------------------------------------------
    # TAB 3: AI CONSENSUS & SIGNAL RADAR (7 ENGINES)
    # ------------------------------------------------------------
    with tab_consensus:
        master_signal = {
            "action": bias if bias != "NEUTRAL" else "HOLD",
            "conviction": consensus.get("directional_conviction", 0.5),
            "entry": last_close,
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "position_size_usd": pos_size_usd,
        }
        execution_data = {
            "master_trade_signal": master_signal,
            "component_scores": consensus.get("component_scores", {}),
            "consensus_report": consensus,
            "df": df,
        }
        render_execution_dashboard(execution_data, symbol, interval)

    # ------------------------------------------------------------
    # TAB 4: PAPER TRADING PORTFOLIO & MONTE CARLO STRESS TEST
    # ------------------------------------------------------------
    with tab_portfolio:
        st.markdown(f"### 💼 Paper Trading Account & Risk Stress Simulator — `{asset_meta['name']} ({symbol})`")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Available Cash", f"${pf_summary['cash']:,.2f}")
        p2.metric("Total Portfolio Value", f"${pf_summary['total_value']:,.2f}")
        p3.metric(
            "Unrealized PnL",
            f"${pf_summary['unrealized_pnl']:,.2f}",
            delta=f"{pf_summary['unrealized_pnl']:+,.2f}",
        )
        p4.metric("Active Open Positions", len(pf_summary["active_positions"]))

        st.divider()

        col_order, col_positions = st.columns([1.0, 1.6])
        with col_order:
            st.markdown("#### 📝 Place Custom Paper Order")
            default_qty = pos_breakdown["recommended_qty"]
            order_qty = st.number_input(
                f"Quantity ({asset_meta.get('unit_label', symbol)})",
                min_value=0.0001,
                value=max(float(default_qty), 0.0001),
                step=0.01,
                format="%.4f",
                key="custom_order_qty",
            )
            st.caption(
                f"Estimated Notional Value: **${order_qty * last_close:,.2f}** at Market (`{fmt_last}`)"
            )
            b_col, s_col = st.columns(2)
            with b_col:
                if st.button("🟢 Execute BUY", use_container_width=True, key="tab_pf_buy"):
                    ok_b, msg_b = pf.buy_asset(symbol, order_qty, last_close)
                    if ok_b:
                        st.success(msg_b)
                        st.rerun()
                    else:
                        st.error(msg_b)
            with s_col:
                if st.button("🔴 Execute SELL / SHORT", use_container_width=True, key="tab_pf_sell"):
                    ok_s, msg_s = pf.sell_asset(symbol, order_qty, last_close)
                    if ok_s:
                        st.success(msg_s)
                        st.rerun()
                    else:
                        st.error(msg_s)

        with col_positions:
            st.markdown("#### 📊 Active Open Positions")
            if len(pf_summary["active_positions"]) > 0:
                df_pos = pd.DataFrame(pf_summary["active_positions"])
                st.dataframe(df_pos, use_container_width=True, hide_index=True)
            else:
                st.info("No active positions held. Use the 1-Click AI Trade bar or Order Ticket to open a position.")

        st.divider()

        # Merton Jump-Diffusion Monte Carlo Tail-Risk Stress Test
        st.markdown("#### 🎲 Merton Jump-Diffusion Monte Carlo Tail-Risk Stress Test (1,000 Paths)")
        st.caption(
            "Simulates future price paths incorporating sudden market jumps (Poisson shocks) to compute "
            "your 95% Value-at-Risk (VaR) and Expected Shortfall (CVaR)."
        )
        jd_res = run_jump_diffusion_mc(df_feat, days=forecast_horizon, simulations=500)
        if jd_res:
            j1, j2, j3, j4 = st.columns(4)
            j1.metric("Expected Mean Price", format_price(jd_res["mean_expected"], symbol))
            j2.metric(
                "95% Value-at-Risk (Floor)",
                format_price(jd_res["var_95"], symbol),
                f"{((jd_res['var_95'] - last_close)/last_close)*100:.2f}%",
                delta_color="inverse",
            )
            j3.metric(
                "95% CVaR (Crash Tail Avg)",
                format_price(jd_res["cvar_95"], symbol),
                f"{((jd_res['cvar_95'] - last_close)/last_close)*100:.2f}%",
                delta_color="inverse",
            )
            j4.metric("Poisson Jump Intensity (λ)", f"{jd_res['lambda_jump']*100:.1f}% / bar")


# Render dashboard within Fragment
render_dashboard()

st.sidebar.divider()
st.sidebar.caption("🔒 *Chart zoom, drawings, and active tabs persist across live auto-refreshes.*")
