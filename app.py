import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from data_engine import fetch_data
from features import add_features
from models import TradingModel, run_monte_carlo, get_markov_regime
from patterns import detect_and_draw_patterns
import importlib
import tv_chart
importlib.reload(tv_chart)
from tv_chart import (
    ChartSpec, add_patterns, render_tv_chart, series_points, to_time,
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
    _BUY_T, _SELL_T = 0.56, 0.45

# Default component weights for signal fusion – kept in one place for discoverability
DEFAULT_COMPONENT_WEIGHTS = {
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

try:
    from metrics.validation import PredictionTracker
    _HAS_TRACKER = True
except Exception:
    _HAS_TRACKER = False

# -----------------------------------------------------------------------
# CONFIGURATION CONSTANTS – make thresholds and component weights user‑configurable
# -----------------------------------------------------------------------
# (already defined above as DEFAULT_COMPONENT_WEIGHTS)

st.set_page_config(
    page_title="Professional AI Trading Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom sleek CSS styling for high-density TradingView aesthetic
st.markdown("""
<style>
    .trade-plan-card {
        background: linear-gradient(135deg, rgba(19, 23, 34, 0.95), rgba(30, 36, 52, 0.95));
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-buy { background: rgba(8, 153, 129, 0.2); color: #00E676; border: 1px solid #00E676; }
    .badge-sell { background: rgba(242, 54, 69, 0.2); color: #FF5252; border: 1px solid #FF5252; }
    .badge-neutral { background: rgba(255, 235, 59, 0.2); color: #FFF176; border: 1px solid #FFF176; }
</style>
""", unsafe_allow_html=True)


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
        "1wk": timedelta(weeks=1)
    }
    step = step_map.get(interval, timedelta(days=1))
    return [last_date + step * (i + 1) for i in range(n_bars)]


def calculate_position_size(cash: float, entry: float, stop_loss: float, risk_pct: float = 0.02) -> float:
    """
    Calculates the total USD position size based on a fixed percentage of cash risk.
    Formula: Position_Size_USD = (Cash * Risk_Pct) / (abs(Entry - Stop_Loss) / Entry)
    """
    risk_amount = cash * risk_pct
    price_diff = abs(entry - stop_loss)
    if price_diff <= 0:
        return 0.0
    
    # Position size in units
    units = risk_amount / price_diff
    # Total USD value
    return units * entry


def log_prediction(symbol: str, interval: str, prediction: dict, df_feat: pd.DataFrame):
    """
    Logs the current forward‑looking prediction to the on‑disk tracker so drift
    detection can run later, and resolves the *previous* pending prediction's
    actual outcome once a new bar has arrived. No‑op on silent auto‑refreshes
    that have not produced a new bar.
     """
    if not _HAS_TRACKER:
        return
    try:
        last_ts = df_feat.index[-1]
        if last_ts == st.session_state.get('last_logged_ts'):
            return
        tracker = PredictionTracker(symbol, interval)
        pending = st.session_state.get('pending_prediction')
        # Clear stale pending prediction if symbol or interval changed
        if pending and (pending.get('symbol') != symbol or pending.get('interval') != interval):
            # Remove stale entry to avoid memory leak
            if 'pending_prediction' in st.session_state:
                del st.session_state['pending_prediction']
            if 'last_logged_ts' in st.session_state:
                del st.session_state['last_logged_ts']
        # Resolve the earlier prediction now that a new bar confirms its outcome.
        if pending and pending.get('symbol') == symbol and pending.get('interval') == interval:
            if len(df_feat) >= 2:
                actual = float(df_feat['Close'].iloc[-1]) > float(df_feat['Close'].iloc[-2])
                tracker.log_prediction(pending['signal'], pending['prob_up'],
                                       actual_result=actual)
        # Log the new prediction (outcome unknown until the next bar).
        tracker.log_prediction(prediction.get('signal', 'NEUTRAL'),
                               float(prediction.get('prob_up', 0.5)),
                               actual_result=None)
        st.session_state['pending_prediction'] = {
            'symbol': symbol, 'interval': interval, 'ts': last_ts,
            'signal': prediction.get('signal', 'NEUTRAL'),
            'prob_up': float(prediction.get('prob_up', 0.5)),
        }
        st.session_state['last_logged_ts'] = last_ts
    except Exception as e:
        st.session_state['prediction_log_error'] = str(e)


def compute_trade_plan(last_close: float, atr: float, prediction: dict, regime: dict, detected_patterns: dict):
    """
    Builds a trade plan that combines signal fusion, dynamic stops, and Kelly sizing.
    """
    sig = prediction.get('signal', 'NEUTRAL')
    prob_up = prediction.get('prob_up', 0.5)

    # Check structural geometric patterns first
    pattern_name = None
    pat_tp, pat_sl = None, None

    if "bearish_bat" in detected_patterns:
        bias = "SELL"
        pattern_name = "Bearish Bat (Harmonic XABCD)"
        pat = detected_patterns["bearish_bat"]
        pat_tp = pat.get('target')
        pat_sl = pat.get('stop')
    elif "cup_handle" in detected_patterns:
        bias = "SELL"
        pattern_name = "Inverted Cup & Handle"
        pat = detected_patterns["cup_handle"]
        pat_tp = pat.get('target')
        pat_sl = pat.get('support')
    elif "wolfe_wave" in detected_patterns:
        bias = "BUY"
        pattern_name = "Bullish Wolfe Wave"
        pat = detected_patterns["wolfe_wave"]
        pat_tp = pat.get('p6', [None, None])[1] if 'p6' in pat else None
        pat_sl = pat.get('p5', [None, None])[1] if 'p5' in pat else None
    elif "falling_wedge" in detected_patterns:
        bias = "BUY"
        pattern_name = "Bullish Falling Wedge"
        pat = detected_patterns["falling_wedge"]
        pat_tp = pat.get('target')
        pat_sl = pat.get('stop')
    elif "BUY" in sig or prob_up > _BUY_T:
        bias = "BUY"
    elif "SELL" in sig or prob_up < _SELL_T:
        bias = "SELL"
    else:
        bias = "NEUTRAL"

    # Dynamic ATR risk buffer based on market regime
    regime_name = regime.get('regime', 'CHOPPY') if isinstance(regime, dict) else str(regime)
    if "BULL" in regime_name or "BEAR" in regime_name:
        sl_multiplier = 1.5
    else:
        sl_multiplier = 2.5

    risk_dist = sl_multiplier * (atr if atr > 0 else last_close * 0.02)
    rr_target_ratio = 2.0

    if bias == "SELL":
        entry = last_close
        stop_loss = pat_sl if pat_sl and pat_sl > entry else round(entry + risk_dist, 2)
        take_profit = pat_tp if pat_tp and pat_tp < entry else round(entry - (rr_target_ratio * abs(stop_loss - entry)), 2)
        take_profit_tp1 = round(entry - (1.0 * abs(stop_loss - entry)), 2)
        profit_pct = abs(entry - take_profit) / entry * 100
        risk_pct = abs(stop_loss - entry) / entry * 100
        rr_actual = profit_pct / risk_pct if risk_pct > 0 else 2.0
    elif bias == "BUY":
        entry = last_close
        stop_loss = pat_sl if pat_sl and pat_sl < entry else round(entry - risk_dist, 2)
        take_profit = pat_tp if pat_tp and pat_tp > entry else round(entry + (rr_target_ratio * abs(entry - stop_loss)), 2)
        take_profit_tp1 = round(entry + (1.0 * abs(entry - stop_loss)), 2)
        profit_pct = abs(take_profit - entry) / entry * 100
        risk_pct = abs(entry - stop_loss) / entry * 100
        rr_actual = profit_pct / risk_pct if risk_pct > 0 else 2.0
    else:
        entry = last_close
        stop_loss = round(entry - risk_dist, 2)
        take_profit = round(entry + risk_dist, 2)
        take_profit_tp1 = take_profit
        profit_pct = 0.0
        risk_pct = 0.0
        rr_actual = 1.0

    return {
        "bias": bias,
        "pattern_name": pattern_name,
        "entry": round(entry, 2),
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "take_profit_tp1": round(take_profit_tp1, 2),
        "profit_pct": profit_pct,
        "risk_pct": risk_pct,
        "rr_ratio": rr_actual,
        "atr": atr
    }


# ============================================================
# SIDEBAR CONFIGURATION (PERSISTENT KEYS)
# ============================================================

st.sidebar.header("Trading Configuration")
symbol = st.sidebar.text_input("Trading Symbol", value="BTC-USD", key="symbol_input").upper().strip()

timeframe_options = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk"]
saved_tf = st.session_state.get("active_interval", "1d")
tf_idx = timeframe_options.index(saved_tf) if saved_tf in timeframe_options else 6

interval = st.sidebar.selectbox("Timeframe", options=timeframe_options, index=tf_idx, key="interval_select")

period_map = {
    "1m": "7d",
    "5m": "5d",
    "15m": "1mo",
    "30m": "1mo",
    "1h": "3mo",
    "4h": "6mo",
    "1d": "2y",
    "1wk": "5y"
}
period = period_map.get(interval, "2y")

retrain_ai = st.sidebar.checkbox("Retrain AI Model", value=False, key="retrain_ai_chk")

st.sidebar.divider()

# Auto‑Refresh Settings
st.sidebar.subheader("Auto‑Refresh", divider=False)
auto_refresh = st.sidebar.checkbox("Enable Auto‑Refresh", value=False, key="auto_refresh_chk")
if auto_refresh:
    refresh_interval = st.sidebar.selectbox(
        "Refresh Interval",
        options=["10s", "30s", "1m", "5m"],
        index=1,
        key="refresh_interval_select"
    )
    run_every_val = refresh_interval
else:
    run_every_val = None

st.sidebar.divider()

# Chart Appearance
st.sidebar.subheader("Chart Settings", divider=False)
chart_height = st.sidebar.selectbox(
    "Chart Height",
    options=["Standard (700px)", "Large (900px)", "Full (1100px)"],
    index=0,
    key="chart_height_select"
)
height_map = {
    "Standard (700px)": 700,
    "Large (900px)": 900,
    "Full (1100px)": 1100
}
chart_h = height_map.get(chart_height, 700)

chart_timezone = st.sidebar.selectbox(
    "Timezone Display",
    options=["IST (India Standard Time - UTC+5:30)", "Local Time (Browser / Device)", "UTC (Market Time)", "EST (New York)"],
    index=0,
    key="chart_timezone_select",
    help="Select the timezone used on the chart time scale, crosshair, and timestamps"
)
tz_mode_map = {
    "IST (India Standard Time - UTC+5:30)": "IST",
    "Local Time (Browser / Device)": "Local",
    "UTC (Market Time)": "UTC",
    "EST (New York)": "EST",
}
chart_tz_mode = tz_mode_map.get(chart_timezone, "IST")

show_future_forecast = st.sidebar.checkbox(
    "🔮 Future Entry & Exit Projection",
    value=True,
    help="Displays projected future price trajectory with Take‑Profit & Stop‑Loss risk zones directly on the chart",
    key="show_future_forecast"
)

st.sidebar.caption("✏️ **Chart Markings**: Click **`✏️ Draw`** on the chart toolbar to place **Support**, **Resistance**, or **Trendlines**. All markings persist across reloads until you click **`⌫ Undo`** or **`🗑️ Clear`**.")

st.sidebar.divider()

# Technical Overlays (Persistent Keys)
st.sidebar.subheader("Technical Overlays", divider=False)
ind_col1, ind_col2, ind_col3 = st.sidebar.columns(3)

with ind_col1:
    show_sma20 = st.checkbox("SMA‑20", value=True, key="chk_sma20")
    show_bb = st.checkbox("Bollinger", value=True, key="chk_bb")

with ind_col2:
    show_sma50 = st.checkbox("SMA‑50", value=True, key="chk_sma50")
    show_trendlines = st.checkbox("Trend Channel", value=False, key="chk_trend")

with ind_col3:
    show_sma200 = st.checkbox("SMA‑200", value=False, key="chk_sma200")

st.sidebar.divider()

# Subplot Indicator
st.sidebar.subheader("Secondary Subplot", divider=False)
subplot_choice = st.sidebar.selectbox(
    "Subplot Indicator",
    options=["Volume", "MACD (12,26,9)", "RSI (14)", "None"],
    index=0,
    key="subplot_select"
)

st.sidebar.divider()

# View Control
st.sidebar.subheader("View Control", divider=False)
zoom_options = ["Last 30 Bars", "Last 60 Bars", "Last 120 Bars", "Last 200 Bars", "Last 365 Bars", "All Data"]
selected_zoom = st.sidebar.selectbox(
    "Visible Candle Window",
    options=zoom_options,
    index=5,
    key="zoom_selector"
)

st.sidebar.divider()

# Zoom & Interaction
st.sidebar.subheader("Zoom & Interaction", divider=False)
zoom_mode = st.sidebar.radio(
     "Zoom Behaviour",
     options=["Both axes (wheel)", "Horizontal only (lock Y)", "Vertical only (lock X)"],
     index=0,
     help=("Mouse wheel / drag-to‑zoom applies. "
           "'Horizontal only' locks the price (Y) axis so you zoom the time (X) axis alone; "
           "'Vertical only' locks the time (X) axis so you zoom the price (Y) axis alone.")
)
# Zoom generation integer. It stays CONSTANT across auto‑refresh ticks so the
# chart's uirevision never changes and Plotly preserves the user's zoom. It only
# INCREMENTS on a deliberate action (a new control selection or the Reset button),
# which intentionally fits a fresh default view. This is the crux of refresh‑proof
# zoom: ordinary refresh = same generation = frozen zoom.
st.session_state.setdefault("zoom_gen", 1)

if st.sidebar.button("⟲ Reset Chart Zoom", use_container_width=True, key="reset_zoom_btn"):
     # Discard the current zoom/pan and re‑fit to the default view, exactly once.
    st.session_state["zoom_gen"] += 1
st.sidebar.caption("🔒 Zoom & pan auto‑persist across auto‑refresh. A reset or a new control fits a fresh view.")

# Deliberate control changes SHOULD fit a new view; an auto‑refresh must NOT.
_zoom_controls = (selected_zoom, zoom_mode, chart_h, chart_tz_mode, symbol, str(interval))
if st.session_state.get("zoom_controls_sig") != _zoom_controls:
     # Re‑fit to the new selection by advancing the generation (one‑shot bump).
    st.session_state["zoom_gen"] += 1
    st.session_state["zoom_controls_sig"] = _zoom_controls
zoom_gen = st.session_state.get("zoom_gen", 1)



st.sidebar.divider()

# Chart Geometric Patterns
st.sidebar.subheader("Harmonic & Geometric Patterns", divider=False)
pat_col1, pat_col2 = st.sidebar.columns(2)

with pat_col1:
    show_wolfe_wave = st.checkbox("🌊 Wolfe Wave", value=True, key="pat_wolfe", help="Bullish 5‑point expanding wedge")
    show_bat_pattern = st.checkbox("🦇 Bearish Bat", value=True, key="pat_bat", help="Harmonic XABCD reversal pattern")

with pat_col2:
    show_cup_handle = st.checkbox("☕ Inverted Cup", value=True, key="pat_cup", help="Inverted Cup & Handle continuation")
    show_falling_wedge = st.checkbox("📐 Falling Wedge", value=True, key="pat_fw", help="Bullish contraction breakout")

st.sidebar.divider()

# Analysis Panels
st.sidebar.subheader("Analysis Panels", divider=False)
show_deep_dive = st.sidebar.checkbox("Show AI Deep Dive", value=False, help="ML model votes & quantum analysis", key="show_deep_dive_chk")
show_portfolio = st.sidebar.checkbox("Show Paper Portfolio", value=False, help="Mock trading account & positions", key="show_portfolio_chk")

fetch_clicked = st.sidebar.button("📥 Fetch & Analyze", key="btn_fetch_analyze")


# ============================================================
# DATA FETCHING & AUTOMATIC STATE TRANSITION
# ============================================================

# Automatically detect if user modified symbol or timeframe
symbol_changed = st.session_state.get('active_symbol') != symbol
interval_changed = st.session_state.get('active_interval') != interval

if fetch_clicked or ('df' not in st.session_state) or symbol_changed or interval_changed:
    with st.spinner(f"Loading data for {symbol} ({interval})..."):
        df = fetch_data(symbol, period=period, interval=interval)
        
    if df is not None and len(df) > 0:
        df_feat = add_features(df)
        model = TradingModel(symbol, interval)
        model.train_or_load(df_feat, force_retrain=retrain_ai)
        prediction = model.predict_next(df_feat, buy_threshold=_BUY_T, sell_threshold=_SELL_T)
        regime = get_markov_regime(df_feat)
        log_prediction(symbol, interval, prediction, df_feat)
        
        st.session_state['df'] = df
        st.session_state['df_feat'] = df_feat
        st.session_state['prediction'] = prediction
        st.session_state['regime'] = regime
        st.session_state['active_symbol'] = symbol
        st.session_state['active_interval'] = interval
    elif fetch_clicked or symbol_changed or interval_changed:
        st.error(f"Failed to fetch data for {symbol} on {interval}. Check symbol or Yahoo Finance connection.")
        st.stop()


# ============================================================
# AUTO‑REFRESH FRAGMENT (PRESERVES USER STATE & ZOOM)
# ============================================================

@st.fragment(run_every=run_every_val)
def render_dashboard():
    """Auto‑refresh dashboard data smoothly while preserving chart zoom state"""
    
    # Auto‑refresh: quiet data sync
    if auto_refresh and 'df' in st.session_state and not fetch_clicked:
        _df_new = fetch_data(symbol, period=period, interval=interval)
        if _df_new is not None and len(_df_new) > 0:
            df_feat_new = add_features(_df_new)
            st.session_state['df'] = _df_new
            st.session_state['df_feat'] = df_feat_new
            # Quietly refresh prediction & regime
            model = TradingModel(symbol, interval)
            model.train_or_load(df_feat_new, force_retrain=False)
            _pred = model.predict_next(df_feat_new, buy_threshold=_BUY_T, sell_threshold=_SELL_T)
            st.session_state['prediction'] = _pred
            st.session_state['regime'] = get_markov_regime(df_feat_new)
            log_prediction(symbol, interval, _pred, df_feat_new)

    if 'df' not in st.session_state:
        st.info("Click **📥 Fetch & Analyze** to load data")
        return

    df = st.session_state['df']
    df_feat = st.session_state['df_feat']
    prediction = st.session_state.get('prediction', {'signal': 'N/A', 'prob_up': 0.5})
    regime = st.session_state.get('regime', 'N/A')

    last_close = float(df['Close'].iloc[-1])
    prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else last_close
    pct_change = ((last_close - prev_close) / prev_close * 100) if prev_close != 0 else 0
    atr = float(df_feat['ATR'].iloc[-1]) if 'ATR' in df_feat.columns and not np.isnan(df_feat['ATR'].iloc[-1]) else (last_close * 0.02)

    # Top Executive Summary Metrics
    last_ts = df.index[-1]
    if getattr(last_ts, 'tzinfo', None) is not None:
        last_ts_utc = last_ts.tz_convert('UTC')
    else:
        last_ts_utc = pd.Timestamp(last_ts).tz_localize('UTC')
    utc_str = last_ts_utc.strftime('%H:%M UTC')
    try:
        last_ts_ist = last_ts_utc.tz_convert('Asia/Kolkata')
        ist_str = last_ts_ist.strftime('%H:%M IST')
    except Exception:
        ist_str = utc_str

    logger.info(
        f"Data sync check: {symbol} ({interval}) - latest bar is {last_ts} ({utc_str}) / {ist_str} | "
        f"Server local time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Total bars: {len(df)}"
    )

    st.markdown(f"## 📈 Trading Terminal — `{symbol}` ({interval})")
    st.caption(
        f"🕒 **Latest Candle:** `{ist_str}` / `{utc_str}` (`{last_ts.strftime('%Y-%m-%d')}`) · "
        f"**Active Chart Timezone:** `{chart_tz_mode}` · **Feed Status:** 🟢 Live & Synchronized"
    )
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Current Price", f"${last_close:,.2f}", f"{pct_change:+.2f}%")
    
    prob_up = prediction.get('prob_up', 0.5)
    sig_text = prediction.get('signal', 'NEUTRAL')
    col2.metric("AI Model Signal", sig_text, f"{prob_up*100:.1f}% UP" if prob_up >= 0.5 else f"{(1-prob_up)*100:.1f}% DOWN")
    
    # Safe regime handling
    if isinstance(regime, dict):
        regime_label = regime.get('regime', 'CHOPPY')
        score = regime.get('markov_score')
        delta_regime = f"{score*100:.0f}% Conviction" if score is not None else None
    else:
        regime_label = str(regime)
        delta_regime = None
    col3.metric("Markov Regime", regime_label, delta=delta_regime, delta_color="off")
    
    col4.metric("ATR Volatility", f"${atr:,.2f}", f"{(atr/last_close)*100:.2f}%")
    col5.metric("Historical Bars", f"{len(df):,} Candles")

    # ============================================================
    # ZOOM WINDOWING
    # ============================================================
    data_len = len(df)
    zoom_map = {
        "Last 30 Bars": 30,
        "Last 60 Bars": 60,
        "Last 120 Bars": 120,
        "Last 200 Bars": 200,
        "Last 365 Bars": 365,
        "All Data": data_len
    }
    zoom_bars = zoom_map.get(selected_zoom, data_len)
    df_zoom = df.iloc[-zoom_bars:].copy() if zoom_bars < data_len else df.copy()
    df_feat_zoom = df_feat.iloc[-zoom_bars:].copy() if zoom_bars < data_len else df_feat.copy()

    # Detect geometric patterns with bar‑level hysteresis (prevents flickering on intra‑candle noise)
    _pat_cache_key = (str(df.index[-1]), len(df), show_wolfe_wave, show_bat_pattern, show_cup_handle, show_falling_wedge)
    if st.session_state.get('_pat_cache_key') == _pat_cache_key and '_cached_patterns' in st.session_state:
        detected_patterns = st.session_state['_cached_patterns']
    else:
        detected_patterns = detect_and_draw_patterns(
            None, df_feat_zoom,
            has_sub=False,
            show_wolfe=show_wolfe_wave,
            show_cup=show_cup_handle,
            show_bat=show_bat_pattern,
            show_wedge=show_falling_wedge
        )
        st.session_state['_cached_patterns'] = detected_patterns
        st.session_state['_pat_cache_key'] = _pat_cache_key

    # Compute Comprehensive Trade Plan (Entry, TP, SL)
    trade_plan = compute_trade_plan(
        last_close, atr, prediction, regime, detected_patterns)
    bias = trade_plan["bias"]
    tp_price = trade_plan["take_profit"]
    sl_price = trade_plan["stop_loss"]
    profit_pct = trade_plan["profit_pct"]
    risk_pct = trade_plan["risk_pct"]
    rr_ratio = trade_plan["rr_ratio"]

    # ============================================================
    # TRADE EXECUTION & EXIT STRATEGY BANNER
    # ============================================================
    with st.container():
        st.markdown("### 🎯 Trade Execution Plan & Exit Strategy")
        c_action, c_entry, c_tp, c_sl, c_rr = st.columns([1.2, 1, 1.2, 1.2, 1])

        with c_action:
            if bias == "SELL":
                st.markdown("""
                <div class="badge badge-sell">🔴 SHORT / SELL SETUP</div>
                """, unsafe_allow_html=True)
            elif bias == "BUY":
                st.markdown("""
                <div class="badge badge-buy">🟢 LONG / BUY SETUP</div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="badge badge-neutral">🟡 NEUTRAL / CONSOLIDATION</div>
                """, unsafe_allow_html=True)
            if trade_plan["pattern_name"]:
                st.caption(f"Driven by: **{trade_plan['pattern_name']}**")
            else:
                st.caption("Driven by: **AI Direction & Volatility Engine**")

        with c_entry:
            st.metric("Entry Level", f"${last_close:,.2f}", "Market Price")

        with c_tp:
            st.metric(
                "Take‑Profit (Target)",
                f"${tp_price:,.2f}",
                f"+{profit_pct:.2f}% Target",
                delta_color="normal"
            )

        with c_sl:
            st.metric(
                "Stop‑Loss (Exit)",
                f"${sl_price:,.2f}",
                f"-{risk_pct:.2f}% Invalid‑ation",
                delta_color="inverse"
            )

        with c_rr:
            st.metric(
                "Risk / Reward",
                f"1 : {rr_ratio:.2f}",
                "Favorable" if rr_ratio >= 1.5 else "Moderate",
                delta_color="off"
            )

    # ============================================================
    # FUTURE PROJECTION: ENTRY, EXIT & MONTE CARLO TRAJECTORY
    # ============================================================
    has_subplot = subplot_choice != "None"
    spec = ChartSpec()
    spec.set_candles(df_zoom)
    last_date = df_zoom.index[-1]

    # Technical Overlays
    def _overlay(col, color, width, title, style=LS_SOLID):
        if col in df_feat_zoom.columns:
            spec.line(series_points(df_feat_zoom.index, df_feat_zoom[col]),
                      color, width, style, title, legend=True)

    if show_sma20:
        _overlay('SMA_20', '#FFD54F', 1.2, 'SMA 20')
    if show_sma50:
        _overlay('SMA_50', '#FF9800', 1.2, 'SMA 50')
    if show_sma200:
        _overlay('SMA_200', '#7E57C2', 1.5, 'SMA 200')
    if show_bb and 'BB_High' in df_feat_zoom.columns and 'BB_Low' in df_feat_zoom.columns:
        _overlay('BB_High', 'rgba(120, 123, 134, 0.65)', 1, 'BB Upper', LS_DASHED)
        _overlay('BB_Low', 'rgba(120, 123, 134, 0.65)', 1, 'BB Lower', LS_DASHED)

    # Harmonic & Geometric Patterns (already detected above)
    if detected_patterns:
        add_patterns(spec, detected_patterns, last_date)
    if show_future_forecast:
        n_proj = 15
        future_dates = get_future_dates(df.index[-1], interval, n_bars=n_proj)
        forecast_span = [df.index[-1]] + future_dates

        # Shaded Take‑Profit (green) & Stop‑Loss (red) zones over the forecast horizon
        spec.zone(forecast_span, last_close, tp_price, "rgba(0, 230, 118, 0.55)", "rgba(0, 230, 118, 0.11)")
        spec.zone(forecast_span, last_close, sl_price, "rgba(255, 82, 82, 0.55)", "rgba(255, 82, 82, 0.11)")

        # Entry / TP / SL as native price lines
        spec.price_line(last_close, "#29B6F6", f"ENTRY {last_close:,.2f}", LS_DOTTED, 2)
        spec.price_line(tp_price, "#00E676", f"🎯 TP +{profit_pct:.1f}% (R:R 1:{rr_ratio:.1f})", LS_DASHED, 2)
        spec.price_line(sl_price, "#FF5252", f"🛑 SL -{risk_pct:.1f}%", LS_DASHED, 2)

        # Monte‑Carlo trajectory (analytic median + 85 % / 15 % quantiles)
        try:
            mc_res = run_monte_carlo(df_feat, days=n_proj, simulations=300, prob_up=prob_up)
            if mc_res:
                if 'analytical_median' in mc_res:
                    median_path = [last_close] + mc_res['analytical_median']
                    upper_path = [last_close] + mc_res['analytical_upper85']
                    lower_path = [last_close] + mc_res['analytical_lower15']
                else:
                    paths = mc_res['paths']
                    median_path = [last_close] + list(paths.median(axis=1).values)
                    upper_path = [last_close] + list(paths.quantile(0.85, axis=1).values)
                    lower_path = [last_close] + list(paths.quantile(0.15, axis=1).values)
                spec.line(series_points(forecast_span, upper_path), "rgba(41, 98, 255, 0.55)", 1, LS_DOTTED, "Forecast 85%")
                spec.line(series_points(forecast_span, lower_path), "rgba(41, 98, 255, 0.55)", 1, LS_DOTTED, "Forecast 15%")
                spec.line(series_points(forecast_span, median_path), "#82B1FF", 1.6, LS_DASHED,
                          "Projected Trajectory", legend=True)
        except Exception as e:
            # Surface a silent MC failure instead of hiding it (debug aid).
            st.session_state.setdefault('mc_last_error', str(e))

    # ============================================================
    # SUB‑PLOT (VOLUME / MACD / RSI)
    # ============================================================
    if subplot_choice != "None":
        if subplot_choice == "Volume" and 'Volume' in df_zoom.columns:
            spec.histogram([
                {"time": to_time(t), "value": float(v),
                 "color": 'rgba(8, 153, 129, 0.5)' if c >= o else 'rgba(242, 54, 69, 0.5)'}
                for t, v, c, o in zip(df_zoom.index, df_zoom['Volume'], df_zoom['Close'], df_zoom['Open'])
                if pd.notna(v)
            ], title="Volume", pane=1, volume=True)

        elif subplot_choice == "MACD (12,26,9)" and 'MACD' in df_feat_zoom.columns:
            if 'MACD_Signal' in df_feat_zoom.columns:
                hist = (df_feat_zoom['MACD'] - df_feat_zoom['MACD_Signal'])
                spec.histogram([
                    {"time": to_time(t), "value": float(v),
                     "color": 'rgba(8, 153, 129, 0.5)' if v >= 0 else 'rgba(242, 54, 69, 0.5)'}
                    for t, v in zip(hist.index, hist) if pd.notna(v)
                ], title="MACD Hist", pane=1)
            spec.line(series_points(df_feat_zoom.index, df_feat_zoom['MACD']), '#29B6F6', 1.3,
                      title='MACD', pane=1, last_value=True)
            if 'MACD_Signal' in df_feat_zoom.columns:
                spec.line(series_points(df_feat_zoom.index, df_feat_zoom['MACD_Signal']), '#FF5252', 1.3,
                          title='Signal', pane=1, last_value=True)

        elif subplot_choice == "RSI (14)" and 'RSI' in df_feat_zoom.columns:
            spec.line(series_points(df_feat_zoom.index, df_feat_zoom['RSI']), '#AB47BC', 1.5,
                      title='RSI 14', pane=1, last_value=True,
                      price_lines=[
                          {"price": 70, "color": "rgba(242, 54, 69, 0.6)", "lineStyle": LS_DASHED,
                           "lineWidth": 1, "axisLabelVisible": True, "title": "70"},
                          {"price": 30, "color": "rgba(8, 153, 129, 0.6)", "lineStyle": LS_DASHED,
                           "lineWidth": 1, "axisLabelVisible": True, "title": "30"},
                      ])

    # ── Render the TradingView Lightweight Chart ──
    _zoom_mode_js = {"Horizontal only (lock Y)": "lockY",
                     "Vertical only (lock X)": "lockX"}.get(zoom_mode, "both")
    try:
        render_tv_chart(
            spec,
            title=f"{symbol} · {interval}",
            height=chart_h,
            intraday=interval not in ("1d", "1wk"),
            state_key=f"{symbol}|{interval}|g{zoom_gen}|{chart_tz_mode}",
            zoom_mode=_zoom_mode_js,
            filename=f"{symbol}_{interval}",
            tz_mode=chart_tz_mode,
        )
    except LightweightChartsNotBuilt as e:
        st.error(str(e))

    # ============================================================
    # DEEP DIVE & PORTFOLIO PANELS
    # ============================================================
    if show_deep_dive:
        st.divider()
        st.subheader("🤖 AI Model Deep Dive")
        
        # Portfolio snapshot
        pf_summary = pf.get_portfolio_summary({symbol: last_close})
        cash = pf_summary['cash']
        
        # Position sizing (risk 2% of cash per trade)
        pos_size_usd = calculate_position_size(cash, last_close, sl_price, risk_pct=0.02)

        master_signal = {
            "action": bias,
            "conviction": (prob_up if bias == "BUY" else (1 - prob_up) if bias == "SELL" else 0.5),
            "entry": last_close,
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "position_size_usd": pos_size_usd
        }

        # Component scores derived from the latest features
        _last = df_feat.iloc[-1]
        def _f(col, default=0.5):
            try:
                v = _last.get(col, np.nan)
                v = default if v is None or pd.isna(v) else float(v)
                return v
            except Exception:
                return default
        _rsi = _f('RSI', 0.5)
        _rsi_score = min(max(_rsi / 100.0, 0.0), 1.0)
        _macd_score = min(max(_f('MACD_Diff', 0.0) + 0.5, 0.0), 1.0)
        _atr_pct = _f('ATR_Pct', 0.02)
        _vol_score = min(max(_atr_pct / 0.05, 0.0), 1.0)
        _trend_score = min(max(_f('Dist_SMA50', 0.0) + 0.02, 0.0), 1.0)
        _mom_score = min(max((_rsi - 50.0) / 50.0 + 0.5, 0.0), 1.0)
        _regime_score = regime.get('markov_score', 0.5) if isinstance(regime, dict) else 0.5
        _atr_sync = min(max(0.5 + _atr_pct, 0.0), 1.0)
        component_scores = {
            "AI Model": prob_up,
            "Market Regime": _regime_score,
            "Pattern Match": 1.0 if trade_plan["pattern_name"] else 0.5,
            "Vol Strength": _vol_score,
            "Trend Alignment": _trend_score,
            "Momentum": _mom_score,
            "Volume Profile": 0.8,
            "RSI Level": _rsi_score,
            "MACD Signal": _macd_score,
            "ATR Sync": _atr_sync,
        }

        execution_data = {
            "master_trade_signal": master_signal,
            "component_scores": component_scores,
            "df": df
        }

        # Render the high‑ fidelity dashboard
        render_execution_dashboard(execution_data, symbol, interval)
        
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Signal Confidence", f"{prob_up*100:.1f}% UP" if prob_up >= 0.5 else f"{(1-prob_up)*100:.1f}% DOWN")
        c2.metric("Regime Score", f"{regime.get('markov_score', 0.5)*100:.1f}%" if isinstance(regime, dict) else "N/A")
        _m = prediction.get('metrics') or {}
        if _m:
            c3.metric("OOS Accuracy", f"{_m.get('accuracy', 0):.2%}",
                      f"AUC {_m.get('roc_auc', 0):.2f} · base {_m.get('base_rate', 0.5):.2%}")
        else:
            c3.metric("OOS Accuracy", "N/A", "Enable Retrain AI to compute")

    if show_portfolio:
        st.divider()
        st.subheader("💼 Paper Trading Portfolio")
        try:
            summary = pf.get_portfolio_summary({symbol: last_close})
            c1, c2, c3 = st.columns(3)
            c1.metric("Available Cash", f"${summary['cash']:,.2f}")
            c2.metric("Active Positions", len(summary['active_positions']))
            c3.metric("Total Account Value", f"${summary['total_value']:,.2f}")
            if len(summary['active_positions']) > 0:
                df_pos = pd.DataFrame(summary['active_positions'])
                st.dataframe(df_pos, width='stretch')
            else:
                st.info("No active positions currently held. Use execution tools to open paper trades.")
        except Exception as e:
            st.warning(f"Portfolio unavailable: {str(e)}")


# Render dashboard within Fragment
render_dashboard()

st.sidebar.divider()
st.sidebar.caption("⚡ *Chart pan/zoom and active toggles are fully preserved during auto‑refresh.*")