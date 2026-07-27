import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf

from data_engine import fetch_data
from features import add_features
from models import TradingModel, run_monte_carlo, get_markov_regime
from patterns import detect_and_draw_patterns
import portfolio as pf

st.set_page_config(page_title="Professional Trading Chart", layout="wide")

# ============================================================
# SIDEBAR CONFIGURATION
# ============================================================
st.sidebar.header("Trading Configuration")
symbol = st.sidebar.text_input("Trading Symbol", value="BTC-USD", key="symbol_input")
interval = st.sidebar.selectbox("Timeframe", options=["1d", "1h", "5m"], index=0, key="interval_select")

period_map = {"1d": "2y", "1h": "1mo", "5m": "5d"}
period = period_map[interval]

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

# Chart Settings
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

st.sidebar.divider()

# Technical Indicators
st.sidebar.subheader("Technical Indicators", divider=False)
ind_col1, ind_col2, ind_col3 = st.sidebar.columns(3)

with ind_col1:
    show_sma20 = st.checkbox("SMA-20", value=True, key=f"chk_sma20_{symbol}_{interval}")
    show_bb = st.checkbox("Bollinger", value=True, key=f"chk_bb_{symbol}_{interval}")

with ind_col2:
    show_sma50 = st.checkbox("SMA-50", value=True, key=f"chk_sma50_{symbol}_{interval}")
    show_trendlines = st.checkbox("Trend Channel", value=False, key=f"chk_trend_{symbol}_{interval}")

with ind_col3:
    show_sma200 = st.checkbox("SMA-200", value=False, key=f"chk_sma200_{symbol}_{interval}")

st.sidebar.divider()

# Subplot Selection
st.sidebar.subheader("Secondary Indicator", divider=False)
subplot_choice = st.sidebar.selectbox(
    "Subplot",
    options=["MACD (12,26,9)", "RSI (14)", "Volume", "None"],
    index=0,
    key="subplot_select"
)

st.sidebar.divider()

# Zoom/View Control
st.sidebar.subheader("View Control", divider=False)
zoom_options = ["Last 7", "Last 14", "Last 30", "Last 60", "Last 90", "Last 120", "Last 200", "Last 365", "All Data"]
saved_zoom_idx = st.session_state.get(f"zoom_preset_idx_{symbol}_{interval}", 8)
if saved_zoom_idx >= len(zoom_options):
    saved_zoom_idx = 8

selected_zoom = st.sidebar.selectbox(
    "Time Window",
    options=zoom_options,
    index=saved_zoom_idx,
    key=f"zoom_selector_{symbol}_{interval}"
)
new_zoom_idx = zoom_options.index(selected_zoom)
st.session_state[f"zoom_preset_idx_{symbol}_{interval}"] = new_zoom_idx

st.sidebar.divider()

# Chart Patterns
st.sidebar.subheader("Chart Patterns", divider=False)
st.sidebar.caption("Auto-detects selected patterns and draws on chart")
pat_col1, pat_col2 = st.sidebar.columns(2)

with pat_col1:
    show_wolfe_wave = st.checkbox(
        "🌊 Wolfe Wave",
        value=False,
        key=f"pat_wolfe_{symbol}_{interval}",
        help="Bullish 5-point pattern"
    )
    show_bat_pattern = st.checkbox(
        "🦇 Bearish Bat",
        value=False,
        key=f"pat_bat_{symbol}_{interval}",
        help="Harmonic XABCD pattern"
    )

with pat_col2:
    show_cup_handle = st.checkbox(
        "☕ Inv. Cup",
        value=False,
        key=f"pat_cup_{symbol}_{interval}",
        help="Inverted cup & handle"
    )
    show_falling_wedge = st.checkbox(
        "📐 Falling Wedge",
        value=False,
        key=f"pat_fw_{symbol}_{interval}",
        help="Bullish wedge pattern"
    )

st.sidebar.divider()

# Analysis Panels
st.sidebar.subheader("Analysis Panels", divider=False)
show_deep_dive = st.sidebar.checkbox("Show AI Deep Dive", value=False, help="ML models & quantum analysis", key="show_deep_dive_chk")
show_portfolio = st.sidebar.checkbox("Show Portfolio", value=False, help="Trading account & positions", key="show_portfolio_chk")

# ============================================================
# DATA FETCHING & PROCESSING
# ============================================================
fetch_clicked = st.sidebar.button("📥 Fetch & Analyze")

if fetch_clicked or ('df' not in st.session_state):
    if fetch_clicked:
        with st.spinner(f"Fetching data for {symbol}..."):
            df = fetch_data(symbol, period=period, interval=interval)
    else:
        df = fetch_data(symbol, period=period, interval=interval)
    
    if df is not None and len(df) > 0:
        df_feat = add_features(df)
        model = TradingModel(symbol, interval)
        model.train_or_load(df_feat, force_retrain=retrain_ai)
        prediction = model.predict_next(df_feat)
        regime = get_markov_regime(df_feat)
        
        st.session_state['df'] = df
        st.session_state['df_feat'] = df_feat
        st.session_state['prediction'] = prediction
        st.session_state['regime'] = regime
        st.session_state['symbol'] = symbol
    elif fetch_clicked:
        st.error(f"Failed to fetch data for {symbol}")
        st.stop()

# ============================================================
# AUTO-REFRESH FRAGMENT (preserves chart zoom)
# ============================================================
@st.fragment(run_every=run_every_val)
def render_dashboard():
    """Auto-refresh data while preserving chart zoom state"""
    
    # Auto-refresh: fetch new data quietly
    if auto_refresh and 'df' in st.session_state:
        _df_new = fetch_data(symbol, period=period, interval=interval)
        if _df_new is not None and len(_df_new) > 0:
            st.session_state['df'] = _df_new
            st.session_state['df_feat'] = add_features(_df_new)
    
    # Get data from session state
    if 'df' not in st.session_state:
        st.info("Click **📥 Fetch & Analyze** to load data")
        return
    
    df = st.session_state['df']
    df_feat = st.session_state['df_feat']
    prediction = st.session_state.get('prediction', {'signal': 'N/A', 'prob_up': 0.5})
    regime = st.session_state.get('regime', 'N/A')
    
    last_close = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2] if len(df) > 1 else last_close
    pct_change = ((last_close - prev_close) / prev_close * 100) if prev_close != 0 else 0
    
    # ============================================================
    # MAIN CONTENT
    # ============================================================
    
    # Top Metrics
    st.markdown("## 📈 Professional Trading Chart")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${last_close:.2f}", f"{pct_change:+.2f}%")
    col2.metric("AI Signal", prediction.get('signal', 'N/A'), f"{prediction.get('prob_up', 0.5)*100:.1f}% UP")
    col3.metric("Markov Regime", regime)
    col4.metric("Data Points", len(df))
    
    # ============================================================
    # BUILD THE CHART
    # ============================================================
    
    # Determine data range based on zoom selection
    data_len = len(df)
    zoom_map = {
        "Last 7": 7, "Last 14": 14, "Last 30": 30, "Last 60": 60,
        "Last 90": 90, "Last 120": 120, "Last 200": 200, "Last 365": 365,
        "All Data": data_len
    }
    zoom_bars = zoom_map.get(selected_zoom, data_len)
    df_zoom = df.iloc[-zoom_bars:].copy() if zoom_bars < data_len else df.copy()
    df_feat_zoom = df_feat.iloc[-zoom_bars:].copy() if zoom_bars < data_len else df_feat.copy()
    
    # Create candlestick chart with optional subplot
    if subplot_choice == "None":
        fig = go.Figure(data=[
            go.Candlestick(
                x=df_zoom.index,
                open=df_zoom['Open'],
                high=df_zoom['High'],
                low=df_zoom['Low'],
                close=df_zoom['Close'],
                name="OHLC"
            )
        ])
        has_subplot = False
    else:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3]
        )
        fig.add_trace(
            go.Candlestick(
                x=df_zoom.index,
                open=df_zoom['Open'],
                high=df_zoom['High'],
                low=df_zoom['Low'],
                close=df_zoom['Close'],
                name="OHLC"
            ),
            row=1, col=1
        )
        has_subplot = True
    
    # Add Technical Indicators to main chart
    if show_sma20 and 'SMA_20' in df_feat_zoom.columns:
        fig.add_trace(
            go.Scatter(x=df_feat_zoom.index, y=df_feat_zoom['SMA_20'],
                      line=dict(color='yellow', width=1), name='SMA-20', mode='lines'),
            row=1 if has_subplot else None, col=1 if has_subplot else None
        )
    
    if show_sma50 and 'SMA_50' in df_feat_zoom.columns:
        fig.add_trace(
            go.Scatter(x=df_feat_zoom.index, y=df_feat_zoom['SMA_50'],
                      line=dict(color='orange', width=1), name='SMA-50', mode='lines'),
            row=1 if has_subplot else None, col=1 if has_subplot else None
        )
    
    if show_sma200 and 'SMA_50' in df_feat_zoom.columns:
        fig.add_trace(
            go.Scatter(x=df_feat_zoom.index, y=df_feat_zoom.get('SMA_50', df_feat_zoom['SMA_50']),
                      line=dict(color='purple', width=1), name='SMA-200', mode='lines'),
            row=1 if has_subplot else None, col=1 if has_subplot else None
        )
    
    if show_bb:
        if 'BB_High' in df_feat_zoom.columns and 'BB_Low' in df_feat_zoom.columns:
            fig.add_trace(
                go.Scatter(x=df_feat_zoom.index, y=df_feat_zoom['BB_High'],
                          line=dict(color='gray', width=0.5, dash='dash'), name='BB Upper', mode='lines'),
                row=1 if has_subplot else None, col=1 if has_subplot else None
            )
            fig.add_trace(
                go.Scatter(x=df_feat_zoom.index, y=df_feat_zoom['BB_Low'],
                          line=dict(color='gray', width=0.5, dash='dash'), name='BB Lower',
                          fill='tonexty', mode='lines'),
                row=1 if has_subplot else None, col=1 if has_subplot else None
            )
    
    # Add Subplot Indicator
    if has_subplot:
        if subplot_choice == "MACD (12,26,9)" and 'MACD' in df_feat_zoom.columns:
            fig.add_trace(
                go.Scatter(x=df_feat_zoom.index, y=df_feat_zoom['MACD'],
                          line=dict(color='white', width=1), name='MACD', mode='lines'),
                row=2, col=1
            )
            if 'MACD_Signal' in df_feat_zoom.columns:
                fig.add_trace(
                    go.Scatter(x=df_feat_zoom.index, y=df_feat_zoom['MACD_Signal'],
                              line=dict(color='red', width=1), name='Signal', mode='lines'),
                    row=2, col=1
                )
        elif subplot_choice == "RSI (14)" and 'RSI' in df_feat_zoom.columns:
            fig.add_trace(
                go.Scatter(x=df_feat_zoom.index, y=df_feat_zoom['RSI'],
                          line=dict(color='cyan', width=1), name='RSI', mode='lines'),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        elif subplot_choice == "Volume":
            fig.add_trace(
                go.Bar(x=df_zoom.index, y=df_zoom['Volume'], name='Volume',
                      marker_color='rgba(0,100,255,0.3)'),
                row=2, col=1
            )
    
    # Add Chart Patterns
    if any([show_wolfe_wave, show_bat_pattern, show_cup_handle, show_falling_wedge]):
        detect_and_draw_patterns(
            fig, df_feat_zoom,
            has_sub=has_subplot,
            show_wolfe=show_wolfe_wave,
            show_cup=show_cup_handle,
            show_bat=show_bat_pattern,
            show_wedge=show_falling_wedge
        )
    
    # Update layout
    fig.update_layout(
        template='plotly_dark',
        height=chart_h,
        xaxis_rangeslider_visible=False,
        uirevision=f"tvpro_{symbol}_{interval}",
        dragmode='pan',
        hovermode='x unified'
    )
    
    if has_subplot:
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text=subplot_choice, row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True, key=f"main_chart_{symbol}_{interval}")
    
    # ============================================================
    # ANALYSIS PANELS (Optional)
    # ============================================================
    
    if show_deep_dive:
        st.subheader("🤖 AI Deep Dive")
        col1, col2 = st.columns(2)
        col1.metric("Signal Confidence", f"{prediction.get('prob_up', 0.5)*100:.1f}%")
        col2.metric("Quantile Signals", str(prediction.get('quantile_signals', 'N/A')))
        st.markdown(f"**Regime Filter:** {regime}")
    
    if show_portfolio:
        st.subheader("💼 Portfolio & Positions")
        try:
            port_state = pf.load_portfolio()
            summary = pf.get_portfolio_summary({symbol: last_close})
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Cash", f"${summary['cash']:,.2f}")
            c2.metric("Active Positions", len(summary['active_positions']))
            c3.metric("Total Value", f"${summary['total_value']:,.2f}")
            
            if len(summary['active_positions']) > 0:
                df_pos = pd.DataFrame(summary['active_positions'])
                st.dataframe(df_pos, use_container_width=True)
        except Exception as e:
            st.warning(f"Portfolio data unavailable: {str(e)}")

# Render the dashboard (auto-refreshing if enabled)
render_dashboard()

st.sidebar.divider()
st.sidebar.caption("*Chart zoom is preserved during auto-refresh*")
