import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
import json
from datetime import datetime

from data_engine import fetch_data
from features import add_features
from models import TradingModel, run_monte_carlo, get_markov_regime
import portfolio as pf

# ============================================================
# PAGE CONFIGURATION - INTERNATIONAL STANDARD UI
# ============================================================
st.set_page_config(
    page_title="Universal Trading Predictor - TradingView Professional",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit default elements
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .viewerBadge_container {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# International header - Professional
st.markdown("""
<div style="text-align: center; padding: 10px 0; border-bottom: 2px solid #1f77b4;">
    <h1 style="margin: 0; color: #ffffff;">Universal Trading Predictor</h1>
    <p style="margin: 0; color: #cccccc; font-size: 12px;">Professional Trading Platform with AI-Powered Analysis</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR - INSTRUMENT & SETTINGS PANEL (TradingView Style)
# ============================================================
with st.sidebar:
    st.header("⚙️ Trading Configuration", divider=True)
    
    # Symbol Input with precision
    symbol = st.text_input(
        "Trading Symbol",
        value="BTC-USD",
        placeholder="e.g., BTC-USD, AAPL, GC=F",
        help="Enter any tradable symbol"
    )
    
    # Timeframe Selection
    interval = st.selectbox(
        "Timeframe",
        options=["1d", "1h", "5m"],
        index=0,
        format_func=lambda x: {"1d": "Daily (1D)", "1h": "Hourly (1H)", "5m": "5 Minutes (5M)"}.get(x, x)
    )
    period_map = {"1d": "2y", "1h": "1mo", "5m": "5d"}
    period = period_map[interval]
    
    st.divider()
    
    # Data Refresh Controls
    col1, col2 = st.columns(2)
    with col1:
        fetch_clicked = st.button("🔄 Refresh", type="primary", use_container_width=True)
    with col2:
        force_retrain = st.checkbox("Retrain AI", value=False, help="Force rebuild ML models")
    
    # Auto-refresh settings
    st.divider()
    st.subheader("Auto-Refresh", divider=False)
    auto_refresh = st.checkbox("Enable Auto-Refresh")
    if auto_refresh:
        refresh_interval = st.selectbox(
            "Refresh Interval",
            options=["10s", "30s", "1m", "5m"],
            index=1
        )
        run_every_val = refresh_interval
        st.info(f"📡 Updates every {refresh_interval}")
    else:
        run_every_val = None
    
    st.divider()
    
    # Chart Settings
    st.subheader("Chart Settings", divider=False)
    
    chart_height = st.selectbox(
        "Chart Height",
        options=["Compact (500px)", "Standard (700px)", "Large (900px)", "Full (1100px)"],
        index=1
    )
    height_map = {
        "Compact (500px)": 500,
        "Standard (700px)": 700,
        "Large (900px)": 900,
        "Full (1100px)": 1100
    }
    chart_h = height_map.get(chart_height, 700)
    
    st.divider()
    
    # Technical Indicators
    st.subheader("Technical Indicators", divider=False)
    
    # Column-based layout for checkboxes
    ind_col1, ind_col2, ind_col3 = st.columns(3)
    
    with ind_col1:
        show_sma20 = st.checkbox("SMA-20", value=True, key=f"chk_sma20_{symbol}_{interval}")
        show_bb = st.checkbox("Bollinger Bands", value=True, key=f"chk_bb_{symbol}_{interval}")
    
    with ind_col2:
        show_sma50 = st.checkbox("SMA-50", value=True, key=f"chk_sma50_{symbol}_{interval}")
        show_trendlines = st.checkbox("Trend Channel", value=True, key=f"chk_trend_{symbol}_{interval}")
    
    with ind_col3:
        show_sma200 = st.checkbox("SMA-200", value=True, key=f"chk_sma200_{symbol}_{interval}")
    
    st.divider()
    
    # Subplot Selection
    st.subheader("Subplot Indicator", divider=False)
    subplot_choice = st.selectbox(
        "Secondary Chart",
        options=["MACD (12,26,9)", "RSI (14)", "Volume", "None"],
        index=0
    )
    
    st.divider()
    
    # Zoom Preservation Controls
    st.subheader("View Control", divider=False)
    zoom_options = [
        "Last 7", "Last 14", "Last 30", "Last 60",
        "Last 90", "Last 120", "Last 200", "Last 365", "All Data"
    ]
    saved_zoom_idx = st.session_state.get(f"zoom_preset_idx_{symbol}_{interval}", 3)
    if saved_zoom_idx >= len(zoom_options):
        saved_zoom_idx = 3
    
    selected_zoom = st.selectbox(
        "Time Window",
        options=zoom_options,
        index=saved_zoom_idx,
        key=f"zoom_selector_{symbol}_{interval}"
    )
    new_zoom_idx = zoom_options.index(selected_zoom)
    
    prev_zoom_key = f"prev_zoom_preset_{symbol}_{interval}"
    zoom_preset_changed = st.session_state.get(prev_zoom_key) != selected_zoom
    st.session_state[prev_zoom_key] = selected_zoom
    st.session_state[f"zoom_preset_idx_{symbol}_{interval}"] = new_zoom_idx


# ============================================================
# DATA FETCHING & PROCESSING
# ============================================================
# Check if symbol OR interval changed, or first boot, or manual fetch button clicked
need_refetch = (
    fetch_clicked or 
    'df' not in st.session_state or 
    st.session_state.get('symbol') != symbol or 
    st.session_state.get('interval') != interval
)

if need_refetch:
    if fetch_clicked:
        with st.spinner(f"📊 Fetching market data for {symbol} ({interval})..."):
            df = fetch_data(symbol, period=period, interval=interval)
    else:
        df = fetch_data(symbol, period=period, interval=interval)
        
    if df is not None and len(df) > 0:
        with st.spinner(f"🔧 Engineering indicators..."):
            df_feat = add_features(df)
        with st.spinner(f"🤖 Running AI analysis..."):
            model = TradingModel(symbol, interval)
            model.train_or_load(df_feat, force_retrain=force_retrain)
            prediction = model.predict_next(df_feat)
            regime = get_markov_regime(df_feat)
            
        st.session_state['df'] = df
        st.session_state['df_feat'] = df_feat
        st.session_state['prediction'] = prediction
        st.session_state['regime'] = regime
        st.session_state['symbol'] = symbol
        st.session_state['interval'] = interval
    elif fetch_clicked or need_refetch:
        st.error(f"Failed to fetch data for {symbol}. Please check the symbol and try again.")
        st.stop()


# ============================================================
# DASHBOARD FRAGMENT - REAL-TIME RENDERING
# ============================================================
@st.fragment(run_every=run_every_val)
def render_main_chart():
    """Primary chart rendering with TradingView-style layout"""
    
    # Auto-refresh logic
    if auto_refresh and 'df' in st.session_state and not fetch_clicked:
        df = fetch_data(symbol, period=period, interval=interval)
        if df is not None and len(df) > 0:
            df_feat = add_features(df)
            model = TradingModel(symbol, interval)
            model.train_or_load(df_feat, force_refrain=False)
            prediction = model.predict_next(df_feat)
            regime = get_markov_regime(df_feat)
            
            st.session_state['df'] = df
            st.session_state['df_feat'] = df_feat
            st.session_state['prediction'] = prediction
            st.session_state['regime'] = regime

    if (
        'df' in st.session_state and 
        st.session_state.get('symbol') == symbol and 
        st.session_state.get('interval') == interval
    ):
        # Get data from session state
        df = st.session_state['df']
        df_feat = st.session_state['df_feat']
        prediction = st.session_state['prediction']
        regime = st.session_state['regime']
        
        # Extract current market data with precision
        last_close = float(df['Close'].iloc[-1])
        prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else last_close
        pct_change = ((last_close - prev_close) / prev_close) * 100.0
        
        # Load trade history for markers
        port_state = pf.load_portfolio()
        trade_history = port_state.get("history", [])
        symbol_trades = [t for t in trade_history if t.get("symbol") == symbol]
        
        # Extract AI signals with quantile targets
        quant_info = prediction.get('quantile_signals', {})
        q10 = quant_info.get('q10_price', last_close * 0.96)
        q50 = quant_info.get('q50_price', last_close)
        q90 = quant_info.get('q90_price', last_close * 1.04)
        
        sig = prediction.get('signal', 'NEUTRAL')
        prob_up = prediction.get('prob_up', 0.5) * 100.0
        
        # Determine trading recommendation
        if sig == "BUY (UP)":
            target_price = q90
            stop_price = q10
            rec_color = "green"
            rec_emoji = "🟢"
            rec_text = f"BUY - Entry: {format_price(symbol, last_close)} | Target: {format_price(symbol, target_price)} | Stop: {format_price(symbol, stop_price)}"
        elif sig == "SELL (DOWN)":
            target_price = q10
            stop_price = q90
            rec_color = "red"
            rec_emoji = "🔴"
            rec_text = f"SELL - Entry: {format_price(symbol, last_close)} | Target: {format_price(symbol, target_price)} | Stop: {format_price(symbol, stop_price)}"
        else:
            target_price = q90
            stop_price = q10
            rec_color = "orange"
            rec_emoji = "🟡"
            rec_text = f"HOLD - Range: {format_price(symbol, q10)} → {format_price(symbol, q90)}"
        
        # ================================================================
        # HEADER SECTION - INSTRUMENT INFO & SIGNAL
        # ================================================================
        header_col1, header_col2, header_col3, header_col4 = st.columns([2, 2, 2, 2])
        
        with header_col1:
            st.metric(
                "Price (OHLC)",
                format_price(symbol, last_close),
                f"{pct_change:+.3f}%"
            )
        
        with header_col2:
            reg_name = regime.get('regime', 'NEUTRAL')
            trend_icon = "📈" if "BULL" in reg_name else "📉" if "BEAR" in reg_name else "↔️"
            st.metric("Market Regime", f"{trend_icon} {reg_name}")
        
        with header_col3:
            sig_display = sig.split("(")[0].strip()
            st.metric(
                "AI Signal",
                f"{rec_emoji} {sig_display}",
                f"{prob_up:.1f}% Probability"
            )
        
        with header_col4:
            st.metric(
                "Take-Profit",
                format_price(symbol, target_price),
                f"{((target_price - last_close) / last_close * 100):+.2f}%"
            )
        
        # Recommendation banner
        st.markdown(f"<div style='background-color: #{'26A69A20' if rec_color == 'green' else 'EF535020' if rec_color == 'red' else 'FFB81C20'}; border-left: 4px solid #{'26A69A' if rec_color == 'green' else 'EF5350' if rec_color == 'red' else 'FFB81C'}; padding: 12px; border-radius: 4px;'><b>{rec_emoji} {rec_text}</b></div>", unsafe_allow_html=True)
        
        st.divider()
        
        # ================================================================
        # MAIN TRADING CHART - TradingView SINGLE CHART LAYOUT
        # ================================================================
        st.markdown("### 📊 Professional Trading Chart")
        
        # Build chart with optional subplot
        has_subplot = subplot_choice != "None"
        if has_subplot:
            from plotly.subplots import make_subplots
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.72, 0.28]
            )
        else:
            fig = go.Figure()
        
        # Main candlestick trace
        candlestick = go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price",
            increasing_line_color='#26A69A',
            decreasing_line_color='#EF5350',
            increasing_fillcolor='#26A69A',
            decreasing_fillcolor='#EF5350'
        )
        if has_subplot:
            fig.add_trace(candlestick, row=1, col=1)
        else:
            fig.add_trace(candlestick)
        
        # Technical indicators
        if show_sma20 and 'SMA_20' in df_feat.columns:
            trace = go.Scatter(
                x=df_feat.index,
                y=df_feat['SMA_20'],
                line=dict(color='#FFD700', width=1.5),
                name='SMA 20',
                hovertemplate='<b>SMA 20</b><br>%{y:.8f}<extra></extra>'
            )
            fig.add_trace(trace, row=1, col=1) if has_subplot else fig.add_trace(trace)
        
        if show_sma50 and 'SMA_50' in df_feat.columns:
            trace = go.Scatter(
                x=df_feat.index,
                y=df_feat['SMA_50'],
                line=dict(color='#00FFFF', width=1.5),
                name='SMA 50',
                hovertemplate='<b>SMA 50</b><br>%{y:.8f}<extra></extra>'
            )
            fig.add_trace(trace, row=1, col=1) if has_subplot else fig.add_trace(trace)
        
        if show_sma200 and 'SMA_200' in df_feat.columns:
            trace = go.Scatter(
                x=df_feat.index,
                y=df_feat['SMA_200'],
                line=dict(color='#FF00FF', width=1.8),
                name='SMA 200',
                hovertemplate='<b>SMA 200</b><br>%{y:.8f}<extra></extra>'
            )
            fig.add_trace(trace, row=1, col=1) if has_subplot else fig.add_trace(trace)
        
        # Bollinger Bands
        if show_bb and 'BB_High' in df_feat.columns:
            trace_bb_h = go.Scatter(
                x=df_feat.index,
                y=df_feat['BB_High'],
                line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dash'),
                name='BB Upper',
                hovertemplate='<b>BB Upper</b><br>%{y:.8f}<extra></extra>'
            )
            trace_bb_l = go.Scatter(
                x=df_feat.index,
                y=df_feat['BB_Low'],
                line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dash'),
                name='BB Lower',
                hovertemplate='<b>BB Lower</b><br>%{y:.8f}<extra></extra>'
            )
            fig.add_trace(trace_bb_h, row=1, col=1) if has_subplot else fig.add_trace(trace_bb_h)
            fig.add_trace(trace_bb_l, row=1, col=1) if has_subplot else fig.add_trace(trace_bb_l)
        
        # Auto trend channel
        if show_trendlines and len(df) >= 20:
            try:
                eval_len = min(60, len(df))
                sub_df = df.tail(eval_len)
                x_nums = np.arange(eval_len)
                high_fit = np.polyfit(x_nums, sub_df['High'], 1)
                low_fit = np.polyfit(x_nums, sub_df['Low'], 1)
                
                line_high = high_fit[0] * x_nums + high_fit[1]
                line_low = low_fit[0] * x_nums + low_fit[1]
                
                trace_th = go.Scatter(
                    x=sub_df.index,
                    y=line_high,
                    line=dict(color='#FFA500', width=2, dash='dot'),
                    name='Trend Upper'
                )
                trace_tl = go.Scatter(
                    x=sub_df.index,
                    y=line_low,
                    line=dict(color='#1E90FF', width=2, dash='dot'),
                    name='Trend Lower'
                )
                fig.add_trace(trace_th, row=1, col=1) if has_subplot else fig.add_trace(trace_th)
                fig.add_trace(trace_tl, row=1, col=1) if has_subplot else fig.add_trace(trace_tl)
            except Exception:
                pass
        
        # Target and Stop-Loss lines
        fig.add_hline(
            y=target_price,
            line_width=2,
            line_dash="dash",
            line_color="#00FF7F",
            annotation_text=f"🎯 Target: {format_price(symbol, target_price)}",
            annotation_position="top right",
            row=1,
            col=1 if has_subplot else None
        )
        fig.add_hline(
            y=stop_price,
            line_width=2,
            line_dash="dash",
            line_color="#FF4500",
            annotation_text=f"🛡️ Stop: {format_price(symbol, stop_price)}",
            annotation_position="bottom right",
            row=1,
            col=1 if has_subplot else None
        )
        
        # Trade markers overlay
        if len(symbol_trades) > 0:
            buy_x, buy_y, buy_texts = [], [], []
            sell_x, sell_y, sell_texts = [], [], []
            
            chart_index = df.index
            price_range = df['High'].max() - df['Low'].min()
            marker_offset = price_range * 0.008
            
            for tr in symbol_trades:
                t_time = tr.get("timestamp")
                t_price = tr.get("price", 0.0)
                t_action = tr.get("action", "")
                t_qty = tr.get("quantity", 0.0)
                
                try:
                    t_dt = pd.to_datetime(t_time)
                    if t_dt.tzinfo is None:
                        t_dt = t_dt.tz_localize('UTC')
                    
                    idx_pos = chart_index.searchsorted(t_dt)
                    if idx_pos >= len(chart_index):
                        idx_pos = len(chart_index) - 1
                    elif idx_pos > 0:
                        before = abs(chart_index[idx_pos - 1] - t_dt)
                        after = abs(chart_index[idx_pos] - t_dt)
                        if before < after:
                            idx_pos = idx_pos - 1
                    snapped_dt = chart_index[idx_pos]
                    candle_low = float(df['Low'].iloc[idx_pos])
                    candle_high = float(df['High'].iloc[idx_pos])
                except Exception:
                    continue
                
                hover_info = f"<b>{t_action}</b><br>Price: {format_price(symbol, t_price)}<br>Qty: {t_qty:.8f}<br>Time: {t_time}"
                
                if "BUY" in t_action:
                    buy_x.append(snapped_dt)
                    buy_y.append(candle_low - marker_offset)
                    buy_texts.append(hover_info)
                elif "SELL" in t_action:
                    sell_x.append(snapped_dt)
                    sell_y.append(candle_high + marker_offset)
                    sell_texts.append(hover_info)
            
            if len(buy_x) > 0:
                tr_buy = go.Scatter(
                    x=buy_x,
                    y=buy_y,
                    mode='markers+text',
                    marker=dict(symbol='triangle-up', size=16, color='#00FF00', line=dict(width=2, color='white')),
                    text=['▲' for _ in buy_x],
                    textposition='bottom center',
                    textfont=dict(color='#00FF00', size=10),
                    name='BUY Trades',
                    hoverinfo='text',
                    hovertext=buy_texts
                )
                fig.add_trace(tr_buy, row=1, col=1) if has_subplot else fig.add_trace(tr_buy)
            
            if len(sell_x) > 0:
                tr_sell = go.Scatter(
                    x=sell_x,
                    y=sell_y,
                    mode='markers+text',
                    marker=dict(symbol='triangle-down', size=16, color='#FF0000', line=dict(width=2, color='white')),
                    text=['▼' for _ in sell_x],
                    textposition='top center',
                    textfont=dict(color='#FF0000', size=10),
                    name='SELL Trades',
                    hoverinfo='text',
                    hovertext=sell_texts
                )
                fig.add_trace(tr_sell, row=1, col=1) if has_subplot else fig.add_trace(tr_sell)
        
        # Add subplot indicator
        if has_subplot:
            if subplot_choice == "MACD (12,26,9)" and 'MACD' in df_feat.columns:
                macd_diff = df_feat['MACD_Diff'].fillna(0)
                colors = []
                for i in range(len(macd_diff)):
                    curr = macd_diff.iloc[i]
                    prev = macd_diff.iloc[i-1] if i > 0 else curr
                    if curr >= 0:
                        colors.append('#26A69A' if curr >= prev else '#B2DFDB')
                    else:
                        colors.append('#EF5350' if curr <= prev else '#FFCDD2')
                
                fig.add_trace(go.Bar(x=df_feat.index, y=macd_diff, marker_color=colors, name='MACD'), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat['MACD'], line=dict(color='#2962FF', width=1.5), name='MACD Line'), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat['MACD_Signal'], line=dict(color='#FF6D00', width=1.5), name='Signal'), row=2, col=1)
            
            elif subplot_choice == "RSI (14)" and 'RSI' in df_feat.columns:
                fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat['RSI'], line=dict(color='#9C27B0', width=1.8), name='RSI'), row=2, col=1)
                fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green", row=2, col=1)
            
            elif subplot_choice == "Volume" and 'Volume' in df.columns:
                vol_colors = ['#26A69A' if c >= o else '#EF5350' for c, o in zip(df['Close'], df['Open'])]
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=vol_colors, name='Volume'), row=2, col=1)
        
        # Chart layout - Zoom Preservation & Microscopic Precision
        stable_uirevision = f"tvpro_{symbol}_{interval}"
        
        layout_dict = dict(
            template='plotly_dark',
            height=chart_h,
            uirevision=stable_uirevision,
            margin=dict(l=50, r=50, t=40, b=50),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10)
            ),
            xaxis=dict(
                uirevision=stable_uirevision,
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.15)',
                type='date',
                fixedrange=False,
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikecolor='rgba(255,255,255,0.6)',
                spikethickness=1,
                spikedash='dot'
            ),
            yaxis=dict(
                uirevision=stable_uirevision,
                title='Price',
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.15)',
                fixedrange=False,
                side='right',
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikecolor='rgba(255,255,255,0.6)',
                spikethickness=1,
                spikedash='dot'
            ),
            hovermode='x unified',
            dragmode='zoom',
            spikedistance=-1
        )
        
        if has_subplot:
            layout_dict['yaxis2'] = dict(
                uirevision=stable_uirevision,
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.12)',
                fixedrange=False,
                side='right'
            )
            layout_dict['xaxis2'] = dict(
                uirevision=stable_uirevision,
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.15)',
                type='date',
                fixedrange=False,
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikecolor='rgba(255,255,255,0.6)',
                spikethickness=1,
                spikedash='dot'
            )
        
        # Apply zoom range if changed
        x_range = None
        if zoom_preset_changed:
            zoom_num_map = {
                "Last 7": 7, "Last 14": 14, "Last 30": 30, "Last 60": 60,
                "Last 90": 90, "Last 120": 120, "Last 200": 200, "Last 365": 365
            }
            n_candles = zoom_num_map.get(selected_zoom)
            if n_candles and len(df) > n_candles:
                x_range = [df.index[-n_candles], df.index[-1]]
        
        if x_range is not None:
            layout_dict['xaxis']['range'] = x_range
            layout_dict['xaxis']['uirevision'] = f"preset_{selected_zoom}"
            layout_dict['yaxis']['uirevision'] = f"preset_{selected_zoom}"
            layout_dict['uirevision'] = f"preset_{selected_zoom}"
            if has_subplot:
                layout_dict['xaxis2']['range'] = x_range
                layout_dict['xaxis2']['uirevision'] = f"preset_{selected_zoom}"
                layout_dict['yaxis2']['uirevision'] = f"preset_{selected_zoom}"
        
        fig.update_xaxes(rangeslider_visible=False)
        fig.update_layout(**layout_dict)
        
        # Plotly configuration - Professional drawing tools
        plotly_config = dict(
            displayModeBar=True,
            displaylogo=False,
            modeBarButtonsToAdd=['drawline', 'drawopenpath', 'drawcircle', 'drawrect', 'eraseshape'],
            modeBarButtonsToRemove=['lasso2d', 'select2d'],
            scrollZoom=True,
            doubleClick='reset+autosize',
            showTips=True,
            toImageButtonOptions=dict(
                format='png',
                filename=f'{symbol}_{interval}_chart',
                height=chart_h,
                width=1600,
                scale=2
            )
        )
        
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"main_chart_{symbol}_{interval}",
            config=plotly_config
        )
        
        st.divider()
        
        # ================================================================
        # TRADE EXECUTION PANEL
        # ================================================================
        st.markdown("### ⚡ Quick Trade Execution")
        
        trade_col1, trade_col2, trade_col3, trade_col4 = st.columns([2, 2, 2, 2])
        
        with trade_col1:
            trade_qty = st.number_input(
                "Quantity",
                min_value=0.0001,
                value=0.1,
                step=0.01,
                format="%.8f",
                key=f"qty_{symbol}_{interval}"
            )
        
        with trade_col2:
            with st.form(f"buy_form_{symbol}_{interval}", clear_on_submit=True):
                btn_buy = st.form_submit_button("🟩 BUY", use_container_width=True, type="primary")
                if btn_buy:
                    success, msg = pf.buy_asset(symbol, trade_qty, last_close)
                    if success:
                        st.success(msg)
                        st.rerun(scope="fragment")
                    else:
                        st.error(msg)
        
        with trade_col3:
            with st.form(f"sell_form_{symbol}_{interval}", clear_on_submit=True):
                btn_sell = st.form_submit_button("🟥 SELL", use_container_width=True, type="primary")
                if btn_sell:
                    success, msg = pf.sell_asset(symbol, trade_qty, last_close)
                    if success:
                        st.success(msg)
                        st.rerun(scope="fragment")
                    else:
                        st.error(msg)
        
        with trade_col4:
            current_prices = {symbol: last_close}
            summary = pf.get_portfolio_summary(current_prices)
            st.metric("Account Balance", f"${summary['total_value']:,.2f}", f"{(summary['unrealized_pnl'] / summary['total_value'] * 100) if summary['total_value'] > 0 else 0:.2f}%")


# ============================================================
# HELPER FUNCTION - PRECISION PRICE FORMATTING
# ============================================================
def format_price(symbol: str, price: float) -> str:
    """Format price with international precision standards"""
    # Crypto assets: 8 decimal places
    if any(x in symbol.upper() for x in ['-USD', 'BTC', 'ETH', 'DOGE', 'ADA', 'BNB']):
        return f"${price:,.8f}"
    # Forex & Commodities: 5 decimal places
    elif any(x in symbol.upper() for x in ['=', 'FX']):
        return f"${price:,.5f}"
    # Stocks & ETFs: 2 decimal places
    else:
        return f"${price:,.2f}"
        
        # ==========================================
        # TAB 1: MAIN EASY SUMMARY DASHBOARD
        # ==========================================
        with tab1:
            header_col1, header_col2 = st.columns([3, 1])
            header_col1.subheader(f"⚡ Executive Summary & Strategy ({symbol})")
            if header_col2.button("🔄 Refresh Data", type="primary", use_container_width=True, key="manual_refresh_btn_tab1"):
                with st.spinner("Fetching latest live market prices..."):
                    new_df = fetch_data(symbol, period=period, interval=interval)
                    if new_df is not None and len(new_df) > 0:
                        new_df_feat = add_features(new_df)
                        new_model = TradingModel(symbol, interval)
                        new_model.train_or_load(new_df_feat, force_retrain=False)
                        st.session_state['df'] = new_df
                        st.session_state['df_feat'] = new_df_feat
                        st.session_state['prediction'] = new_model.predict_next(new_df_feat)
                        st.session_state['regime'] = get_markov_regime(new_df_feat)
                        st.rerun(scope="fragment")
            
            # Top Decision Metric Cards
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Market Price", f"${last_close:,.2f}", f"{pct_change:+.2f}%")
            
            reg_name = regime.get('regime', 'CHOPPY')
            reg_color = "🟢" if "BULL" in reg_name else "🔴" if "BEAR" in reg_name else "🟡"
            c2.metric("Market Trend", f"{reg_color} {reg_name}")
            
            sig_color = "normal" if sig == "BUY (UP)" else "inverse" if sig == "SELL (DOWN)" else "off"
            c3.metric("AI Signal Recommendation", sig, f"{prob_up:.1f}% Probability UP", delta_color=sig_color)
            
            if sig == "BUY (UP)":
                c4.metric("🎯 Take-Profit Target", f"${target_price:,.2f}", f"+{((target_price-last_close)/last_close)*100:.2f}%")
            elif sig == "SELL (DOWN)":
                c4.metric("🎯 Take-Profit Target", f"${target_price:,.2f}", f"{((target_price-last_close)/last_close)*100:.2f}%")
            else:
                c4.metric("🎯 Target Range", f"${q10:,.2f} - ${q90:,.2f}")

            # Plain English Executive Guidance Banner
            if sig == "BUY (UP)":
                st.success(recommendation_banner)
            elif sig == "SELL (DOWN)":
                st.error(recommendation_banner)
            else:
                st.warning(recommendation_banner)

            st.divider()

            # Interactive Chart Header with Zoom & Indicator Controls (TradingView Style)
            st.markdown("### 📉 TradingView Precision Chart & Decision Suite")
            
            ctl_col1, ctl_col2, ctl_col3 = st.columns([3, 2, 2])
            
            zoom_options = [
                "Last 7 Candles", "Last 14 Candles", "Last 30 Candles", 
                "Last 60 Candles", "Last 90 Candles", "Last 120 Candles", 
                "Last 200 Candles", "Last 365 Candles", "Show All Data"
            ]
            saved_zoom_idx = st.session_state.get(f"zoom_preset_idx_{symbol}_{interval}", 3)
            if saved_zoom_idx >= len(zoom_options):
                saved_zoom_idx = 3
            
            selected_zoom = ctl_col1.selectbox(
                "🔍 Zoom Window",
                options=zoom_options,
                index=saved_zoom_idx,
                key=f"zoom_selector_{symbol}_{interval}"
            )
            new_zoom_idx = zoom_options.index(selected_zoom)
            
            prev_zoom_key = f"prev_zoom_preset_{symbol}_{interval}"
            zoom_preset_changed = st.session_state.get(prev_zoom_key) != selected_zoom
            st.session_state[prev_zoom_key] = selected_zoom
            st.session_state[f"zoom_preset_idx_{symbol}_{interval}"] = new_zoom_idx

            chart_height = ctl_col2.selectbox(
                "📏 Chart Height",
                options=["Compact (450px)", "Standard (650px)", "Large (850px)", "Full (1050px)"],
                index=1,
                key=f"chart_height_{symbol}_{interval}"
            )
            height_map = {"Compact (450px)": 450, "Standard (650px)": 650, "Large (850px)": 850, "Full (1050px)": 1050}
            chart_h = height_map.get(chart_height, 650)

            subplot_choice = ctl_col3.selectbox(
                "📊 Subplot Indicator",
                options=["MACD (12,26,9)", "RSI (14)", "Volume", "None"],
                index=0,
                key=f"subplot_choice_{symbol}_{interval}"
            )

            # Indicator Toggle Checkboxes Bar
            ind_col1, ind_col2, ind_col3, ind_col4, ind_col5 = st.columns(5)
            show_sma20 = ind_col1.checkbox("SMA-20 (Gold)", value=True, key=f"chk_sma20_{symbol}_{interval}")
            show_sma50 = ind_col2.checkbox("SMA-50 (Cyan)", value=True, key=f"chk_sma50_{symbol}_{interval}")
            show_sma200 = ind_col3.checkbox("SMA-200 (Magenta)", value=True, key=f"chk_sma200_{symbol}_{interval}")
            show_bb = ind_col4.checkbox("Bollinger Bands", value=True, key=f"chk_bb_{symbol}_{interval}")
            show_trendlines = ind_col5.checkbox("Auto Trend Channel", value=True, key=f"chk_trend_{symbol}_{interval}")

            # Compute x_range only when zoom preset explicitly changes
            zoom_num_map = {
                "Last 7 Candles": 7, "Last 14 Candles": 14, "Last 30 Candles": 30,
                "Last 60 Candles": 60, "Last 90 Candles": 90, "Last 120 Candles": 120,
                "Last 200 Candles": 200, "Last 365 Candles": 365
            }
            
            x_range = None
            if zoom_preset_changed:
                n_candles = zoom_num_map.get(selected_zoom)
                if n_candles and len(df) > n_candles:
                    x_range = [df.index[-n_candles], df.index[-1]]

            # Build Multi-Pane Subplot Chart (TradingView Layout)
            has_subplot = subplot_choice != "None"
            if has_subplot:
                from plotly.subplots import make_subplots
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.72, 0.28]
                )
            else:
                fig = go.Figure()

            # Main Candlestick Chart (Row 1)
            candlestick = go.Candlestick(
                x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name="Price",
                increasing_line_color='#26A69A', decreasing_line_color='#EF5350',
                increasing_fillcolor='#26A69A', decreasing_fillcolor='#EF5350'
            )
            if has_subplot:
                fig.add_trace(candlestick, row=1, col=1)
            else:
                fig.add_trace(candlestick)

            # Moving Averages
            if show_sma20 and 'SMA_20' in df_feat.columns:
                trace_sma20 = go.Scatter(x=df_feat.index, y=df_feat['SMA_20'], line=dict(color='#FFD700', width=1.5), name='SMA 20')
                fig.add_trace(trace_sma20, row=1, col=1) if has_subplot else fig.add_trace(trace_sma20)

            if show_sma50 and 'SMA_50' in df_feat.columns:
                trace_sma50 = go.Scatter(x=df_feat.index, y=df_feat['SMA_50'], line=dict(color='#00FFFF', width=1.5), name='SMA 50')
                fig.add_trace(trace_sma50, row=1, col=1) if has_subplot else fig.add_trace(trace_sma50)

            if show_sma200 and 'SMA_200' in df_feat.columns:
                trace_sma200 = go.Scatter(x=df_feat.index, y=df_feat['SMA_200'], line=dict(color='#FF00FF', width=1.8), name='SMA 200')
                fig.add_trace(trace_sma200, row=1, col=1) if has_subplot else fig.add_trace(trace_sma200)

            # Bollinger Bands
            if show_bb and 'BB_High' in df_feat.columns:
                trace_bb_h = go.Scatter(x=df_feat.index, y=df_feat['BB_High'], line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dash'), name='BB Upper')
                trace_bb_l = go.Scatter(x=df_feat.index, y=df_feat['BB_Low'], line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dash'), name='BB Lower')
                fig.add_trace(trace_bb_h, row=1, col=1) if has_subplot else fig.add_trace(trace_bb_h)
                fig.add_trace(trace_bb_l, row=1, col=1) if has_subplot else fig.add_trace(trace_bb_l)

            # Dynamic Auto Trend Channel (Linear Fit on local Highs and Lows)
            if show_trendlines and len(df) >= 20:
                try:
                    eval_len = min(60, len(df))
                    sub_df = df.tail(eval_len)
                    x_nums = np.arange(eval_len)
                    high_fit = np.polyfit(x_nums, sub_df['High'], 1)
                    low_fit = np.polyfit(x_nums, sub_df['Low'], 1)
                    
                    line_high = high_fit[0] * x_nums + high_fit[1]
                    line_low = low_fit[0] * x_nums + low_fit[1]

                    trace_th = go.Scatter(x=sub_df.index, y=line_high, line=dict(color='#FFA500', width=2, dash='dot'), name='Upper Trend Channel')
                    trace_tl = go.Scatter(x=sub_df.index, y=line_low, line=dict(color='#1E90FF', width=2, dash='dot'), name='Lower Trend Channel')
                    fig.add_trace(trace_th, row=1, col=1) if has_subplot else fig.add_trace(trace_th)
                    fig.add_trace(trace_tl, row=1, col=1) if has_subplot else fig.add_trace(trace_tl)
                except Exception:
                    pass

            # Overlay Next Target Lines
            fig.add_hline(
                y=target_price, line_width=2, line_dash="dash", line_color="#00FF7F",
                annotation_text=f"🎯 Target (Take-Profit): ${target_price:,.2f}", annotation_position="top right",
                row=1, col=1 if has_subplot else None
            )
            fig.add_hline(
                y=stop_price, line_width=2, line_dash="dash", line_color="#FF4500",
                annotation_text=f"🛡️ Stop-Loss Limit: ${stop_price:,.2f}", annotation_position="bottom right",
                row=1, col=1 if has_subplot else None
            )

            # Executed Trade Markers Overlay
            if len(symbol_trades) > 0:
                buy_x, buy_y, buy_texts = [], [], []
                sell_x, sell_y, sell_texts = [], [], []

                chart_index = df.index
                price_range = df['High'].max() - df['Low'].min()
                marker_offset = price_range * 0.008

                for tr in symbol_trades:
                    t_time = tr.get("timestamp")
                    t_price = tr.get("price", 0.0)
                    t_action = tr.get("action", "")
                    t_qty = tr.get("quantity", 0.0)
                    
                    try:
                        t_dt = pd.to_datetime(t_time)
                        if t_dt.tzinfo is None:
                            t_dt = t_dt.tz_localize('UTC')
                        
                        idx_pos = chart_index.searchsorted(t_dt)
                        if idx_pos >= len(chart_index):
                            idx_pos = len(chart_index) - 1
                        elif idx_pos > 0:
                            before = abs(chart_index[idx_pos - 1] - t_dt)
                            after = abs(chart_index[idx_pos] - t_dt)
                            if before < after:
                                idx_pos = idx_pos - 1
                        snapped_dt = chart_index[idx_pos]
                        candle_low = float(df['Low'].iloc[idx_pos])
                        candle_high = float(df['High'].iloc[idx_pos])
                    except Exception:
                        continue

                    hover_info = f"<b>{t_action}</b><br>Price: ${t_price:,.2f}<br>Qty: {t_qty}<br>Time: {t_time}"

                    if "BUY" in t_action:
                        buy_x.append(snapped_dt)
                        buy_y.append(candle_low - marker_offset)
                        buy_texts.append(hover_info)
                    elif "SELL" in t_action:
                        sell_x.append(snapped_dt)
                        sell_y.append(candle_high + marker_offset)
                        sell_texts.append(hover_info)

                if len(buy_x) > 0:
                    tr_buy = go.Scatter(
                        x=buy_x, y=buy_y, mode='markers+text',
                        marker=dict(symbol='triangle-up', size=16, color='#00FF00', line=dict(width=2, color='white')),
                        text=['▲ BUY' for _ in buy_x], textposition='bottom center', textfont=dict(color='#00FF00', size=9),
                        name='Executed BUY', hoverinfo='text', hovertext=buy_texts
                    )
                    fig.add_trace(tr_buy, row=1, col=1) if has_subplot else fig.add_trace(tr_buy)

                if len(sell_x) > 0:
                    tr_sell = go.Scatter(
                        x=sell_x, y=sell_y, mode='markers+text',
                        marker=dict(symbol='triangle-down', size=16, color='#FF0000', line=dict(width=2, color='white')),
                        text=['▼ SELL' for _ in sell_x], textposition='top center', textfont=dict(color='#FF0000', size=9),
                        name='Executed SELL', hoverinfo='text', hovertext=sell_texts
                    )
                    fig.add_trace(tr_sell, row=1, col=1) if has_subplot else fig.add_trace(tr_sell)

            # Subplot Row 2 Implementation (MACD / RSI / Volume)
            if subplot_choice == "MACD (12,26,9)" and 'MACD' in df_feat.columns:
                macd_diff = df_feat['MACD_Diff'].fillna(0)
                # 4-color TradingView histogram
                colors = []
                for i in range(len(macd_diff)):
                    curr = macd_diff.iloc[i]
                    prev = macd_diff.iloc[i-1] if i > 0 else curr
                    if curr >= 0:
                        colors.append('#26A69A' if curr >= prev else '#B2DFDB')  # Dark Green / Light Green
                    else:
                        colors.append('#EF5350' if curr <= prev else '#FFCDD2')  # Dark Red / Light Red
                        
                fig.add_trace(go.Bar(x=df_feat.index, y=macd_diff, marker_color=colors, name='MACD Hist'), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat['MACD'], line=dict(color='#2962FF', width=1.5), name='MACD'), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat['MACD_Signal'], line=dict(color='#FF6D00', width=1.5), name='Signal'), row=2, col=1)

            elif subplot_choice == "RSI (14)" and 'RSI' in df_feat.columns:
                fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat['RSI'], line=dict(color='#9C27B0', width=1.8), name='RSI (14)'), row=2, col=1)
                fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green", row=2, col=1)

            elif subplot_choice == "Volume" and 'Volume' in df.columns:
                vol_colors = ['#26A69A' if c >= o else '#EF5350' for c, o in zip(df['Close'], df['Open'])]
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=vol_colors, name='Volume'), row=2, col=1)

            # Layout & Precision Settings (TradingView Precision & Zoom Lock)
            stable_uirevision = f"chart_state_{symbol}_{interval}"

            layout_dict = dict(
                template='plotly_dark',
                height=chart_h,
                uirevision=stable_uirevision,
                margin=dict(l=40, r=40, t=30, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(
                    uirevision=stable_uirevision,
                    showgrid=True, gridcolor='rgba(128, 128, 128, 0.15)',
                    rangeslider=dict(visible=False),
                    type='date',
                    fixedrange=False,
                    showspikes=True, spikemode='across', spikesnap='cursor',
                    spikecolor='rgba(255,255,255,0.6)', spikethickness=1, spikedash='dot'
                ),
                yaxis=dict(
                    uirevision=stable_uirevision,
                    title='Price ($)',
                    showgrid=True, gridcolor='rgba(128, 128, 128, 0.15)',
                    fixedrange=False,
                    side='right',
                    showspikes=True, spikemode='across', spikesnap='cursor',
                    spikecolor='rgba(255,255,255,0.6)', spikethickness=1, spikedash='dot'
                ),
                xaxis_rangeslider_visible=False,
                hovermode='x unified',
                dragmode='zoom',
                spikedistance=-1
            )

            if has_subplot:
                layout_dict['yaxis2'] = dict(
                    uirevision=stable_uirevision,
                    showgrid=True, gridcolor='rgba(128, 128, 128, 0.12)',
                    fixedrange=False,
                    side='right'
                )
                layout_dict['xaxis2'] = dict(
                    uirevision=stable_uirevision,
                    showgrid=True, gridcolor='rgba(128, 128, 128, 0.15)',
                    rangeslider=dict(visible=False),
                    type='date',
                    fixedrange=False,
                    showspikes=True, spikemode='across', spikesnap='cursor',
                    spikecolor='rgba(255,255,255,0.6)', spikethickness=1, spikedash='dot'
                )

            # Apply range only on zoom preset change
            if x_range is not None and zoom_preset_changed:
                layout_dict['xaxis']['range'] = x_range
                layout_dict['xaxis']['uirevision'] = f"preset_{selected_zoom}"
                layout_dict['yaxis']['uirevision'] = f"preset_{selected_zoom}"
                layout_dict['uirevision'] = f"preset_{selected_zoom}"
                if has_subplot:
                    layout_dict['xaxis2']['range'] = x_range
                    layout_dict['xaxis2']['uirevision'] = f"preset_{selected_zoom}"
                    layout_dict['yaxis2']['uirevision'] = f"preset_{selected_zoom}"

            fig.update_xaxes(rangeslider_visible=False)
            fig.update_xaxes(rangeslider=dict(visible=False))
            fig.update_layout(**layout_dict)

            # Modebar configuration for full drawing and zoom interactivity
            plotly_config = dict(
                displayModeBar=True,
                displaylogo=False,
                modeBarButtonsToAdd=[
                    'drawline', 'drawopenpath', 'drawcircle', 'drawrect',
                    'eraseshape', 'toggleSpikelines'
                ],
                modeBarButtonsToRemove=['lasso2d', 'select2d'],
                scrollZoom=True,
                doubleClick='reset+autosize',
                showTips=True,
                toImageButtonOptions=dict(
                    format='png', filename=f'{symbol}_{interval}_tradingview_chart',
                    height=chart_h, width=1600, scale=2
                )
            )
            st.plotly_chart(fig, use_container_width=True, key=f"main_price_chart_{symbol}_{interval}", config=plotly_config)

            # Quick Trade Execution Bar directly on Home Tab
            st.divider()
            st.subheader(f"⚡ One-Click Trade Execution ({symbol} - {interval})")
            
            q_col1, q_col2, q_col3 = st.columns([2, 2, 2])
            trade_qty = q_col1.number_input("Quantity", min_value=0.0001, value=0.1, step=0.1, key=f"quick_qty_{symbol}_{interval}")
            
            with q_col2:
                with st.form("buy_trade_form", clear_on_submit=True):
                    btn_buy = st.form_submit_button("🟩 BUY NOW", use_container_width=True, type="primary")
                    if btn_buy:
                        success, msg = pf.buy_asset(symbol, trade_qty, last_close)
                        if success:
                            st.success(msg)
                            st.rerun(scope="fragment")
                        else:
                            st.error(msg)

            with q_col3:
                with st.form("sell_trade_form", clear_on_submit=True):
                    btn_sell = st.form_submit_button("🟥 SELL NOW", use_container_width=True, type="primary")
                    if btn_sell:
                        success, msg = pf.sell_asset(symbol, trade_qty, last_close)
                        if success:
                            st.success(msg)
                            st.rerun(scope="fragment")
                        else:
                            st.error(msg)

        # ==========================================
        # TAB 2: AI MODEL & QUANTUM DEEP DIVE
        # ==========================================
        with tab2:
            st.header("🧠 Multi-Model Ultra Stacking & Quantum Science Matrix")
            
            # 5-Model Ultra Stacking Breakdown
            st.subheader("1. 5-Model Machine Learning Votes")
            votes = prediction.get('model_votes', {})
            ml_meta_prob = prediction.get('ml_prob_up', 0.5) * 100.0
            
            if votes:
                v_cols = st.columns(6)
                v_cols[0].metric("XGBoost", f"{votes.get('XGBoost', 0.5)*100:.1f}%")
                v_cols[1].metric("LightGBM", f"{votes.get('LightGBM', 0.5)*100:.1f}%")
                v_cols[2].metric("Random Forest", f"{votes.get('RandomForest', 0.5)*100:.1f}%")
                v_cols[3].metric("Extra Trees", f"{votes.get('ExtraTrees', 0.5)*100:.1f}%")
                v_cols[4].metric("Deep MLP Net", f"{votes.get('DeepMLP', 0.5)*100:.1f}%")
                v_cols[5].metric("🎯 Meta-Stacker", f"{ml_meta_prob:.1f}%")

            st.divider()

            # Quantum Signals Matrix
            q_fft = prediction.get('quantum_signals', {})
            q_hawkes = prediction.get('hawkes_signals', {})
            q_maxent = prediction.get('maxent_signals', {})
            q_hmm = prediction.get('hmm_signals', {})
            
            st.subheader("2. Quantum Signal Science Metrics")
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("FFT Dominant Cycle", f"{q_fft.get('dominant_period', 0)} candles", f"Phase: {q_fft.get('cycle_phase', 'N/A')}")
            q2.metric("Hawkes Volatility Intensity", f"{q_hawkes.get('intensity', 0.0):.2f}", f"{q_hawkes.get('volatility_regime', 'NORMAL')}")
            q3.metric("Jaynes MaxEnt Entropy", f"{q_maxent.get('maxent_entropy', 1.0):.3f}", "Shannon Information Density")
            q4.metric("HMM Latent State", f"{q_hmm.get('latent_state', 'N/A')}", f"Score: {q_hmm.get('hmm_score', 0.5):.2f}")
            
            st.divider()

            # Merton Jump-Diffusion Monte Carlo Simulation
            st.subheader("3. Merton Jump-Diffusion Monte Carlo Simulation (1,000 Paths)")
            with st.spinner("Running Merton Jump-Diffusion simulation..."):
                mc_results = run_monte_carlo(df_feat, days=20, simulations=1000)
                
            if mc_results:
                mc_fig = go.Figure()
                paths = mc_results['paths']
                for i in range(min(60, paths.shape[1])):
                    mc_fig.add_trace(go.Scatter(y=paths.iloc[:, i], mode='lines', line=dict(color='rgba(0, 150, 255, 0.08)'), showlegend=False))
                    
                mean_path = paths.mean(axis=1)
                mc_fig.add_trace(go.Scatter(y=mean_path, mode='lines', line=dict(color='cyan', width=3), name='Mean Path'))
                
                var_line = [mc_results['var_95']] * len(mean_path)
                mc_fig.add_trace(go.Scatter(y=var_line, mode='lines', line=dict(color='red', width=2, dash='dash'), name='95% VaR Floor'))
                
                mc_fig.update_layout(
                    title=f"Merton Jump-Diffusion Price Projections (20 Candles Ahead) | Jump Intensity: {mc_results['lambda_jump']:.2f}",
                    template='plotly_dark',
                    height=400,
                    uirevision=f"mc_zoom_{symbol}",
                    xaxis=dict(uirevision=f"mc_zoom_{symbol}"),
                    yaxis=dict(uirevision=f"mc_zoom_{symbol}"),
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(mc_fig, use_container_width=True, key=f"mc_price_chart_{symbol}")
                
                mc_c1, mc_c2, mc_c3 = st.columns(3)
                mc_c1.metric("Mean Expected Price", f"${mc_results['mean_expected']:,.2f}")
                mc_c2.metric("95% Value-at-Risk (VaR)", f"${mc_results['var_95']:,.2f}")
                mc_c3.metric("95% Expected Shortfall", f"${mc_results['cvar_95']:,.2f}")

        # ==========================================
        # TAB 3: PAPER TRADING PORTFOLIO
        # ==========================================
        with tab3:
            st.header("💼 Paper Trading Account & History")
            
            current_prices = {symbol: last_close}
            for sym in port_state["positions"].keys():
                if sym != symbol:
                    try:
                        current_prices[sym] = yf.Ticker(sym).fast_info.last_price
                    except Exception:
                        pass
                        
            summary = pf.get_portfolio_summary(current_prices)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Available Cash", f"${summary['cash']:,.2f}")
            c2.metric("Unrealized PnL", f"${summary['unrealized_pnl']:,.2f}")
            c3.metric("Total Account Value", f"${summary['total_value']:,.2f}")
            
            st.divider()
            
            st.subheader("Active Positions")
            if len(summary['active_positions']) > 0:
                df_pos = pd.DataFrame(summary['active_positions'])
                df_pos['Quantity'] = df_pos['Quantity'].round(6)
                df_pos['Avg Price'] = df_pos['Avg Price'].apply(lambda x: f"${x:,.2f}")
                df_pos['Current Price'] = df_pos['Current Price'].apply(lambda x: f"${x:,.2f}")
                df_pos['Current Value'] = df_pos['Current Value'].apply(lambda x: f"${x:,.2f}")
                df_pos['Unrealized PnL'] = df_pos['Unrealized PnL'].apply(lambda x: f"${x:,.2f}")
                df_pos['PnL %'] = df_pos['PnL %'].apply(lambda x: f"{x:.2f}%")
                
                def color_position_type(val):
                    if val == 'LONG': return 'color: green'
                    elif val == 'SHORT': return 'color: red'
                    return ''
                    
                st.dataframe(df_pos.style.map(color_position_type, subset=['Type']), use_container_width=True)
            else:
                st.info("No active open positions.")

            st.divider()
            st.subheader(f"Trade Execution History ({symbol})")
            if len(symbol_trades) > 0:
                df_hist = pd.DataFrame(symbol_trades)[::-1]
                df_hist['price'] = df_hist['price'].apply(lambda x: f"${x:,.2f}")
                df_hist['total'] = df_hist['total'].apply(lambda x: f"${x:,.2f}")
                df_hist['realized_pnl'] = df_hist['realized_pnl'].apply(lambda x: f"${x:,.2f}")
                
                def color_action(val):
                    if 'BUY' in str(val): return 'color: green'
                    elif 'SELL' in str(val): return 'color: red'
                    return ''
                    
                st.dataframe(df_hist.style.map(color_action, subset=['action']), use_container_width=True)
            else:
                st.info("No trade history for this symbol yet.")

# Execute fragment
render_dashboard()
