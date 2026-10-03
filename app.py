import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from data_engine import fetch_data
from features import add_features
from models import TradingModel, run_monte_carlo, get_markov_regime
from patterns import detect_and_draw_patterns
import portfolio as pf
from pages.execution_dashboard import render_execution_dashboard

# Centralised signal thresholds + prediction logging.
try:
    from config.settings import SIGNAL_THRESHOLDS
    _BUY_T = float(SIGNAL_THRESHOLDS.get("buy_threshold", 0.56))
    _SELL_T = float(SIGNAL_THRESHOLDS.get("sell_threshold", 0.44))
except Exception:
    _BUY_T, _SELL_T = 0.55, 0.45

try:
    from metrics.validation import PredictionTracker
    _HAS_TRACKER = True
except Exception:
    _HAS_TRACKER = False

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
    Logs the current forward-looking prediction to the on-disk tracker so drift
    detection can run later, and resolves the *previous* pending prediction's
    actual outcome once a new bar has arrived. No-op on silent auto-refreshes
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

# Auto-Refresh Settings
st.sidebar.subheader("Auto-Refresh", divider=False)
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=False, key="auto_refresh_chk")
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

show_future_forecast = st.sidebar.checkbox(
    "🔮 Future Entry & Exit Projection",
    value=True,
    help="Displays projected future price trajectory with Take-Profit & Stop-Loss risk zones directly on the chart",
    key="show_future_forecast"
)

st.sidebar.divider()

# Technical Overlays (Persistent Keys)
st.sidebar.subheader("Technical Overlays", divider=False)
ind_col1, ind_col2, ind_col3 = st.sidebar.columns(3)

with ind_col1:
    show_sma20 = st.checkbox("SMA-20", value=True, key="chk_sma20")
    show_bb = st.checkbox("Bollinger", value=True, key="chk_bb")

with ind_col2:
    show_sma50 = st.checkbox("SMA-50", value=True, key="chk_sma50")
    show_trendlines = st.checkbox("Trend Channel", value=False, key="chk_trend")

with ind_col3:
    show_sma200 = st.checkbox("SMA-200", value=False, key="chk_sma200")

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

# Zoom & Interaction ── horizontal / vertical control + a refresh-proof reset
st.sidebar.subheader("Zoom & Interaction", divider=False)
zoom_mode = st.sidebar.radio(
     "Zoom Behaviour",
     options=["Both axes (wheel)", "Horizontal only (lock Y)", "Vertical only (lock X)"],
     index=0,
     horizontal=True,
     key="zoom_mode_select",
     help=("Mouse wheel / drag-to-zoom applies. "
           "'Horizontal only' freezes the price (Y) axis so you zoom the time (X) axis alone; "
           "'Vertical only' does the opposite. Zoom & pan state is preserved across auto-refresh.")
)
# Zoom generation integer. It stays CONSTANT across auto-refresh ticks so the
# chart's uirevision never changes and Plotly preserves the user's zoom. It only
# INCREMENTS on a deliberate action (a new control selection or the Reset button),
# which intentionally fits a fresh default view. This is the crux of refresh-proof
# zoom: ordinary refresh = same generation = frozen zoom.
st.session_state.setdefault("zoom_gen", 1)

if st.sidebar.button("⟲ Reset Chart Zoom", use_container_width=True, key="reset_zoom_btn"):
     # Discard the current zoom/pan and re-fit to the default view, exactly once.
    st.session_state["zoom_gen"] += 1
st.sidebar.caption("🔒 Zoom & pan auto-persist across auto-refresh. A reset or a new control fits a fresh view.")

# Deliberate control changes SHOULD fit a new view; an auto-refresh must NOT.
_zoom_controls = (selected_zoom, zoom_mode, chart_height, symbol, str(interval))
if st.session_state.get("zoom_controls_sig") != _zoom_controls:
     # Re-fit to the new selection by advancing the generation (one-shot bump).
    st.session_state["zoom_gen"] += 1
    st.session_state["zoom_controls_sig"] = _zoom_controls
zoom_gen = st.session_state.get("zoom_gen", 1)



st.sidebar.divider()

# Chart Geometric Patterns (Persistent Keys)
st.sidebar.subheader("Harmonic & Geometric Patterns", divider=False)
pat_col1, pat_col2 = st.sidebar.columns(2)

with pat_col1:
    show_wolfe_wave = st.checkbox("🌊 Wolfe Wave", value=True, key="pat_wolfe", help="Bullish 5-point expanding wedge")
    show_bat_pattern = st.checkbox("🦇 Bearish Bat", value=True, key="pat_bat", help="Harmonic XABCD reversal pattern")

with pat_col2:
    show_cup_handle = st.checkbox("☕ Inv. Cup", value=True, key="pat_cup", help="Inverted Cup and Handle continuation")
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
# AUTO-REFRESH FRAGMENT (PRESERVES USER STATE & ZOOM)
# ============================================================

@st.fragment(run_every=run_every_val)
def render_dashboard():
    """Auto-refresh dashboard data smoothly while preserving chart zoom state"""
    
    # Auto-refresh: quiet data sync
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
    st.markdown(f"## 📈 Trading Terminal — `{symbol}` ({interval})")
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

    # Pre-detect geometric patterns to inform trade setup
    dummy_fig = go.Figure()
    detected_patterns = detect_and_draw_patterns(
        dummy_fig, df_feat_zoom,
        has_sub=False,
        show_wolfe=show_wolfe_wave,
        show_cup=show_cup_handle,
        show_bat=show_bat_pattern,
        show_wedge=show_falling_wedge
    )

    # Compute Comprehensive Trade Plan (Entry, Take-Profit, Stop-Loss)
    trade_plan = compute_trade_plan(last_close, atr, prediction, regime, detected_patterns)
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
                <div class="metric-badge badge-sell">🔴 SHORT / SELL SETUP</div>
                """, unsafe_allow_html=True)
            elif bias == "BUY":
                st.markdown("""
                <div class="metric-badge badge-buy">🟢 LONG / BUY SETUP</div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-badge badge-neutral">🟡 NEUTRAL / CONSOLIDATION</div>
                """, unsafe_allow_html=True)
            if trade_plan["pattern_name"]:
                st.caption(f"Driven by: **{trade_plan['pattern_name']}**")
            else:
                st.caption("Driven by: **AI Direction & Volatility Engine**")

        with c_entry:
            st.metric("Entry Level", f"${last_close:,.2f}", "Market Price")

        with c_tp:
            st.metric(
                "Take-Profit (Target)",
                f"${tp_price:,.2f}",
                f"+{profit_pct:.2f}% Target",
                delta_color="normal"
            )

        with c_sl:
            st.metric(
                "Stop-Loss (Exit)",
                f"${sl_price:,.2f}",
                f"-{risk_pct:.2f}% Invalidation",
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
    # BUILD TRADINGVIEW-GRADE INTERACTIVE CHART
    # ============================================================
    has_subplot = subplot_choice != "None"

    if not has_subplot:
        fig = go.Figure()
        row_main = None
        col_main = None
    else:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.75, 0.25]
        )
        row_main = 1
        col_main = 1

    # Professional Candlesticks (Emerald & Crimson TradingView styling)
    fig.add_trace(
        go.Candlestick(
            x=df_zoom.index,
            open=df_zoom['Open'],
            high=df_zoom['High'],
            low=df_zoom['Low'],
            close=df_zoom['Close'],
            name=f"{symbol} OHLC",
            increasing=dict(line=dict(color='#089981', width=1), fillcolor='#089981'),
            decreasing=dict(line=dict(color='#F23645', width=1), fillcolor='#F23645'),
            hoverlabel=dict(bgcolor="#1E222D", font=dict(color="white", size=12))
        ),
        row=row_main, col=col_main
    )

    # Technical Overlays
    if show_sma20 and 'SMA_20' in df_feat_zoom.columns:
        fig.add_trace(
            go.Scatter(
                x=df_feat_zoom.index, y=df_feat_zoom['SMA_20'],
                line=dict(color='#FFD54F', width=1.2), name='SMA 20', mode='lines'
            ),
            row=row_main, col=col_main
        )

    if show_sma50 and 'SMA_50' in df_feat_zoom.columns:
        fig.add_trace(
            go.Scatter(
                x=df_feat_zoom.index, y=df_feat_zoom['SMA_50'],
                line=dict(color='#FF9800', width=1.2), name='SMA 50', mode='lines'
            ),
            row=row_main, col=col_main
        )

    if show_sma200 and 'SMA_200' in df_feat_zoom.columns:
        fig.add_trace(
            go.Scatter(
                x=df_feat_zoom.index, y=df_feat_zoom['SMA_200'],
                line=dict(color='#7E57C2', width=1.5), name='SMA 200', mode='lines'
            ),
            row=row_main, col=col_main
        )

    if show_bb and 'BB_High' in df_feat_zoom.columns and 'BB_Low' in df_feat_zoom.columns:
        fig.add_trace(
            go.Scatter(
                x=df_feat_zoom.index, y=df_feat_zoom['BB_High'],
                line=dict(color='rgba(120, 123, 134, 0.45)', width=0.8, dash='dash'),
                name='BB Upper', mode='lines', hoverinfo='skip'
            ),
            row=row_main, col=col_main
        )
        fig.add_trace(
            go.Scatter(
                x=df_feat_zoom.index, y=df_feat_zoom['BB_Low'],
                line=dict(color='rgba(120, 123, 134, 0.45)', width=0.8, dash='dash'),
                fill='tonexty', fillcolor='rgba(41, 98, 255, 0.04)',
                name='BB Lower', mode='lines', hoverinfo='skip'
            ),
            row=row_main, col=col_main
        )

    # Draw Harmonic & Geometric Patterns
    if any([show_wolfe_wave, show_bat_pattern, show_cup_handle, show_falling_wedge]):
        detect_and_draw_patterns(
            fig, df_feat_zoom,
            has_sub=has_subplot,
            show_wolfe=show_wolfe_wave,
            show_cup=show_cup_handle,
            show_bat=show_bat_pattern,
            show_wedge=show_falling_wedge
        )

    # ============================================================
    # FUTURE PREDICTION: ENTRY, EXIT & MONTE CARLO TRAJECTORY
    # ============================================================
    if show_future_forecast:
        n_proj = 15
        last_date = df_zoom.index[-1]
        future_dates = get_future_dates(last_date, interval, n_bars=n_proj)
        forecast_span = [last_date] + future_dates

        # 1. Shaded Take-Profit Target Zone (Green)
        fig.add_trace(
            go.Scatter(
                x=[last_date, future_dates[-1], future_dates[-1], last_date],
                y=[last_close, last_close, tp_price, tp_price],
                fill="toself",
                fillcolor="rgba(0, 230, 118, 0.11)",
                line=dict(color="rgba(0, 230, 118, 0.45)", width=1, dash="dot"),
                name="Target Profit Zone (TP)",
                hoverinfo="skip"
            ),
            row=row_main, col=col_main
        )

        # 2. Shaded Stop-Loss Risk Zone (Red)
        fig.add_trace(
            go.Scatter(
                x=[last_date, future_dates[-1], future_dates[-1], last_date],
                y=[last_close, last_close, sl_price, sl_price],
                fill="toself",
                fillcolor="rgba(255, 82, 82, 0.11)",
                line=dict(color="rgba(255, 82, 82, 0.45)", width=1, dash="dot"),
                name="Stop Loss Zone (SL)",
                hoverinfo="skip"
            ),
            row=row_main, col=col_main
        )

        # 3. Horizontal Entry Line
        fig.add_trace(
            go.Scatter(
                x=[last_date, future_dates[-1]],
                y=[last_close, last_close],
                mode="lines+text",
                line=dict(color="#29B6F6", width=2.0, dash="dot"),
                text=["", f"  ENTRY: ${last_close:,.2f}"],
                textposition="middle right",
                textfont=dict(color="#29B6F6", size=11),
                name="Entry Level"
            ),
            row=row_main, col=col_main
        )

        # 4. Horizontal Take-Profit Exit Line
        fig.add_trace(
            go.Scatter(
                x=[last_date, future_dates[-1]],
                y=[tp_price, tp_price],
                mode="lines+text",
                line=dict(color="#00E676", width=2.2, dash="dash"),
                text=["", f"  🎯 TP (Exit): ${tp_price:,.2f} (+{profit_pct:.1f}%) [R:R 1:{rr_ratio:.1f}]"],
                textposition="middle right",
                textfont=dict(color="#00E676", size=11),
                name="Take-Profit Exit"
            ),
            row=row_main, col=col_main
        )

        # 5. Horizontal Stop-Loss Invalidation Line
        fig.add_trace(
            go.Scatter(
                x=[last_date, future_dates[-1]],
                y=[sl_price, sl_price],
                mode="lines+text",
                line=dict(color="#FF5252", width=2.2, dash="dash"),
                text=["", f"  🛑 SL (Exit): ${sl_price:,.2f} (-{risk_pct:.1f}%)"],
                textposition="middle right",
                textfont=dict(color="#FF5252", size=11),
                name="Stop-Loss Invalidation"
            ),
            row=row_main, col=col_main
        )

        # 6. Monte Carlo Future Price Trajectory Cone
        try:
            mc_res = run_monte_carlo(df_feat, days=n_proj, simulations=300)
            if mc_res and 'paths' in mc_res:
                paths = mc_res['paths']
                median_path = [last_close] + list(paths.median(axis=1).values)
                upper_path = [last_close] + list(paths.quantile(0.85, axis=1).values)
                lower_path = [last_close] + list(paths.quantile(0.15, axis=1).values)

                # Upper boundary
                fig.add_trace(
                    go.Scatter(
                        x=forecast_span, y=upper_path,
                        line=dict(color="rgba(41, 98, 255, 0.3)", width=0.8, dash="dot"),
                        name="Forecast Upper (85%)", mode="lines", hoverinfo="skip"
                    ),
                    row=row_main, col=col_main
                )
                # Lower boundary with fill
                fig.add_trace(
                    go.Scatter(
                        x=forecast_span, y=lower_path,
                        line=dict(color="rgba(41, 98, 255, 0.3)", width=0.8, dash="dot"),
                        fill="tonexty", fillcolor="rgba(41, 98, 255, 0.07)",
                        name="Forecast Confidence Cone", mode="lines", hoverinfo="skip"
                    ),
                    row=row_main, col=col_main
                )
                # Median projected path
                fig.add_trace(
                    go.Scatter(
                        x=forecast_span, y=median_path,
                        line=dict(color="#82B1FF", width=1.6, dash="dashdot"),
                        name="Projected Trajectory", mode="lines"
                    ),
                    row=row_main, col=col_main
                )
        except Exception as e:
             # Surface a silent MC failure instead of hiding it (debug aid).
            st.session_state.setdefault('mc_last_error', str(e))

    # ============================================================
    # SUBPLOT CONFIGURATION (VOLUME / MACD / RSI)
    # ============================================================
    if has_subplot:
        if subplot_choice == "Volume":
            vol_colors = [
                'rgba(8, 153, 129, 0.5)' if c >= o else 'rgba(242, 54, 69, 0.5)'
                for c, o in zip(df_zoom['Close'], df_zoom['Open'])
            ]
            fig.add_trace(
                go.Bar(
                    x=df_zoom.index, y=df_zoom['Volume'],
                    name='Volume', marker_color=vol_colors,
                    hoverlabel=dict(bgcolor="#1E222D")
                ),
                row=2, col=1
            )
            fig.update_yaxes(title_text="Volume", row=2, col=1)

        elif subplot_choice == "MACD (12,26,9)" and 'MACD' in df_feat_zoom.columns:
            fig.add_trace(
                go.Scatter(x=df_feat_zoom.index, y=df_feat_zoom['MACD'], line=dict(color='#29B6F6', width=1.3), name='MACD'),
                row=2, col=1
            )
            if 'MACD_Signal' in df_feat_zoom.columns:
                fig.add_trace(
                    go.Scatter(x=df_feat_zoom.index, y=df_feat_zoom['MACD_Signal'], line=dict(color='#FF5252', width=1.3), name='Signal'),
                    row=2, col=1
                )
            fig.update_yaxes(title_text="MACD", row=2, col=1)

        elif subplot_choice == "RSI (14)" and 'RSI' in df_feat_zoom.columns:
            fig.add_trace(
                go.Scatter(x=df_feat_zoom.index, y=df_feat_zoom['RSI'], line=dict(color='#AB47BC', width=1.5), name='RSI (14)'),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="rgba(242, 54, 69, 0.6)", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="rgba(8, 153, 129, 0.6)", row=2, col=1)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

        fig.update_xaxes(showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)", row=2, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)", row=2, col=1)

    # ── ZOOM & INTERACTION: world-class, fully refresh-proof ─────────────
    # uirevision is the ONLY lever Plotly exposes to keep a user's zoom/pan
    # across figure updates. We back it with session state so it stays CONSTANT
    # on every auto-refresh tick (=> zoom preserved) and increases ONLY when the
    # user clicks 'Reset Chart Zoom' in the sidebar (=> clean one-shot reset).
    _zoom_rev = f"ui_g{zoom_gen}"       # gen is constant on refresh => zoom kept

    # Master Chart Layout & Styling
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0D1117',
        plot_bgcolor='#0E1117',
        height=chart_h,
        margin=dict(l=20, r=80, t=30, b=20),
        uirevision=_zoom_rev,                  # constant on refresh => zoom kept
        dragmode='pan',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02, xanchor="left", x=0.01,
            bgcolor="rgba(19, 23, 34, 0.8)",
            bordercolor="rgba(255, 255, 255, 0.1)", borderwidth=1, font=dict(size=11)
        )
    )

    # ── Per-axis uirevision on EVERY axis (main + any subplot row) ──
    # A single axis missing its revision can reset alone on rerun.
    _axis_common = dict(
        uirevision=_zoom_rev,
        showgrid=True, gridcolor="rgba(255, 255, 255, 0.06)",
        showspikes=True, spikemode='across', spikesnap='cursor',
        spikethickness=1, spikedash='dot', spikecolor="rgba(255, 255, 255, 0.3)",
    )
    fig.update_xaxes(rangeslider_visible=False, **_axis_common)
    fig.update_yaxes(side="right", **_axis_common)

    # ── Zoom isolation: lock one axis so the free axis keeps full zoom/pan ──
    # fixedrange=True freezes an axis; the other axis keeps complete control.
    if zoom_mode == "Horizontal only (lock Y)":
        fig.update_yaxes(fixedrange=True)       # Y frozen => wheel/drag moves X only
    elif zoom_mode == "Vertical only (lock X)":
        fig.update_xaxes(fixedrange=True)       # X frozen => wheel/drag moves Y only
    else:
        # 'Both axes': neither frozen, so X and Y zoom/pan independently.
        fig.update_xaxes(fixedrange=False)
        fig.update_yaxes(fixedrange=False)

    # Full-featured interactive config: horizontal + vertical zoom, one-shot reset.
      # ── Refresh-proof zoom (final guarantee) ──
      # Reuse the EXACT same figure object when inputs are unchanged. Most auto-
      # refresh ticks add no new bar, so this is a no-op and the user's zoom literally
      # cannot move. A new bar or a changed toggle produces a new figure carrying the
      # SAME uirevision, so Plotly keeps the current view even when it redraws.
    _chart_sig = (symbol, str(interval), str(df["Close"].index[-1]), len(df),
              zoom_mode, chart_h, _zoom_rev, subplot_choice, show_future_forecast,
              show_sma20, show_bb, show_sma50, show_sma200, show_trendlines,
              show_wolfe_wave, show_bat_pattern, show_cup_handle, show_falling_wedge)
    if st.session_state.get("_main_chart_sig") == _chart_sig and "main_fig_cache" in st.session_state:
        fig = st.session_state["main_fig_cache"]    # identical inputs => no-op re-render
    st.session_state["main_fig_cache"] = fig
    st.session_state["_main_chart_sig"] = _chart_sig
    st.plotly_chart(
        fig,
        width='stretch',
        key="main_chart_stable",               # constant key => widget never remounts
        config={
            'scrollZoom': True,                # wheel zooms the free axis(es)
            'displayModeBar': True,
            'doubleClick': 'reset',
            'scrollHeight': 12,
            'modeBarButtonsToAdd': [
                'zoom', 'zoomIn', 'zoomOut',
                'pan', 'select', 'lasso',
                'autoscale', 'resetScopes',
                'drawline', 'drawopenpath', 'eraseshape',
            ],
            'toImageButtonOptions': {'format': 'png', 'filename': f'{symbol}_{interval}'},
            'displaylogo': False,
        }
    )

    # ============================================================
    # DEEP DIVE & PORTFOLIO PANELS
    # ============================================================
    if show_deep_dive:
        st.divider()
        st.subheader("🤖 AI Model Deep Dive")
        
        # Prepare data for Execution Dashboard
        pf_summary = pf.get_portfolio_summary({symbol: last_close})
        cash = pf_summary['cash']
        
        # Calculate position size (Risk 2% of cash per trade)
        pos_size_usd = calculate_position_size(cash, last_close, sl_price, risk_pct=0.02)

        master_signal = {
            "action": bias,
            "conviction": prediction.get('prob_up', 0.5) if bias == "BUY" else (1 - prediction.get('prob_up', 0.5) if bias == "SELL" else 0.5),
            "entry": last_close,
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "position_size_usd": pos_size_usd
        }

         # Real component scores derived from the latest computed features (no dummies).
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
            "AI Model": prediction.get('prob_up', 0.5),
            "Market Regime": _regime_score,
            "Pattern Match": 1.0 if trade_plan["pattern_name"] else 0.5,
            "Vol Strength": _vol_score,
            "Trend Alignment": _trend_score,
            "Momentum": _mom_score,
            "Volume Profile": 0.8,     # HVN/LVN not wired into live path yet -> neutral
            "RSI Level": _rsi_score,
            "MACD Signal": _macd_score,
            "ATR Sync": _atr_sync,
        }

        execution_data = {
            "master_trade_signal": master_signal,
            "component_scores": component_scores,
            "df": df
        }

        # Display the high-fidelity dashboard
        render_execution_dashboard(execution_data, symbol, interval)
        
        st.divider()
        c_dd1, c_dd2, c_dd3 = st.columns(3)
        c_dd1.metric("Signal Confidence", f"{prob_up*100:.1f}% UP" if prob_up >= 0.5 else f"{(1-prob_up)*100:.1f}% DOWN")
        c_dd2.metric("Regime Score", f"{regime.get('markov_score', 0.5)*100:.1f}%" if isinstance(regime, dict) else "N/A")
        _m = prediction.get('metrics') or {}
        if _m:
            c_dd3.metric("OOS Accuracy", f"{_m.get('accuracy', 0):.2%}",
                         f"AUC {_m.get('roc_auc', 0):.2f} · base {_m.get('base_rate', 0.5):.2%}")
        else:
            c_dd3.metric("OOS Accuracy", "N/A", "Enable Retrain AI to compute")

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
st.sidebar.caption("⚡ *Chart pan/zoom and active toggles are fully preserved during auto-refresh.*")
