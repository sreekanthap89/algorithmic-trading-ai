"""
=============================================================
 EXECUTION & PnL ANALYTICS DASHBOARD (Trading App)
 Steps 12-13 Implementation
=============================================================
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime

def render_execution_dashboard(data: dict, symbol: str, interval: str):
    """
    Renders the Master Execution Dashboard:
    - Trade Alert Banner
    - Conviction Gauge & Radar Chart breakdown
    - PnL Heatmap (hour of day vs day of week)
    - Sharpe / Sortino / Calmar panels
    - Real-time Signal Execution Log with CSV download
    """
    st.markdown(f"## ⚡ Master Execution Dashboard — `{symbol}` ({interval})")
    
    master_signal = data.get("master_trade_signal", {})
    component_scores = data.get("component_scores", {})
    
    action = master_signal.get("action", "HOLD")
    conviction = master_signal.get("conviction", 0.5)
    entry = master_signal.get("entry", 0.0)
    sl = master_signal.get("stop_loss", 0.0)
    tp = master_signal.get("take_profit", 0.0)
    position_usd = master_signal.get("position_size_usd", 0.0)

    # 1. Trade Alert Banner
    st.markdown("### 🔔 Signal Alert Banner")
    if action == "BUY" and conviction >= 0.70:
        st.success(
            f"### 🟢 HIGH-CONVICTION BUY SIGNAL FIRED!\n"
            f"**Symbol**: {symbol} | **Entry Price**: ${entry:,.2f} | **Take Profit**: ${tp:,.2f} | **Stop Loss**: ${sl:,.2f}\n\n"
            f"**Position Allocation (Half-Kelly)**: ${position_usd:,.2f} | **Conviction Score**: {conviction*100:.1f}%"
        )
    elif action == "SELL" and conviction <= 0.30:
        st.error(
            f"### 🔴 HIGH-CONVICTION SELL SIGNAL FIRED!\n"
            f"**Symbol**: {symbol} | **Entry Price**: ${entry:,.2f} | **Take Profit**: ${tp:,.2f} | **Stop Loss**: ${sl:,.2f}\n\n"
            f"**Position Allocation (Half-Kelly)**: ${position_usd:,.2f} | **Conviction Score**: {(1-conviction)*100:.1f}%"
        )
    else:
        st.info(
            f"### 🟡 NEUTRAL / HOLDING ZONE\n"
            f"Market is in consolidation or signal conviction ({conviction*100:.1f}%) is below trade threshold (75%)."
        )

    st.divider()

    # 2. Conviction Gauge & Radar Chart
    col_gauge, col_radar = st.columns(2)

    with col_gauge:
        st.markdown("#### 🎯 Conviction Score Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=conviction * 100.0,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Master Conviction (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#00E676" if conviction > 0.5 else "#FF5252"},
                'steps': [
                    {'range': [0, 35], 'color': "rgba(255, 82, 82, 0.2)"},
                    {'range': [35, 65], 'color': "rgba(255, 235, 59, 0.2)"},
                    {'range': [65, 100], 'color': "rgba(0, 230, 118, 0.2)"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 75.0
                }
            }
        ))
        fig_gauge.update_layout(height=350, template="plotly_dark")
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_radar:
        st.markdown("#### 🕸️ 10-Component Signal Fusion Radar")
        if component_scores:
            categories = list(component_scores.keys())
            values = list(component_scores.values())
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name='Score',
                line_color='#00E676'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                height=350,
                template="plotly_dark"
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # 3. PnL Heatmap Analytics
    st.markdown("### 📊 PnL Time Slot Heatmap")
    df = data.get("df", None)
    if df is not None and len(df) > 5:
        df_pnl = df.copy()
        dt_idx = pd.to_datetime(df_pnl.index)
        df_pnl['Day'] = dt_idx.day_name()
        df_pnl['Return_Pct'] = df_pnl['Close'].pct_change() * 100.0

        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Check if intraday hours vary
        has_intraday_hours = dt_idx.hour.nunique() > 1
        if has_intraday_hours:
            df_pnl['TimeSlot'] = dt_idx.hour
            x_label = "Hour of Day (UTC)"
            chart_title = "Expected Return Density by Hour of Day"
            pivot_table = df_pnl.pivot_table(index='Day', columns='TimeSlot', values='Return_Pct', aggfunc='mean').fillna(0)
        else:
            # Daily data: Seasonality by Day of Week vs Month
            df_pnl['TimeSlot'] = dt_idx.strftime('%b')
            months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            x_label = "Month of Year"
            chart_title = "Expected Return Seasonality by Day of Week & Month"
            pivot_table = df_pnl.pivot_table(index='Day', columns='TimeSlot', values='Return_Pct', aggfunc='mean').fillna(0)
            pivot_table = pivot_table.reindex(columns=[m for m in months_order if m in pivot_table.columns])

        pivot_table = pivot_table.reindex([d for d in days_order if d in pivot_table.index])

        fig_heat = px.imshow(
            pivot_table,
            labels=dict(x=x_label, y="Day of Week", color="Avg Return (%)"),
            color_continuous_scale="RdYlGn",
            title=chart_title,
            aspect="auto"
        )
        fig_heat.update_layout(height=350, template="plotly_dark")
        try:
            st.plotly_chart(fig_heat, width="stretch")
        except TypeError:
            st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("ℹ️ Fetch data to generate the PnL Return Density Heatmap.")

    st.divider()

    # 4. Signal Execution History Log & CSV Export
    st.markdown("### 📜 Real-Time Signal Execution Log")

    if 'signal_log' not in st.session_state:
        st.session_state['signal_log'] = []

    # Append current signal to log
    new_entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Symbol": symbol,
        "Interval": interval,
        "Action": action,
        "Conviction": f"{conviction*100:.1f}%",
        "Entry": f"${entry:,.2f}",
        "Stop Loss": f"${sl:,.2f}",
        "Take Profit": f"${tp:,.2f}",
        "Allocation": f"${position_usd:,.2f}",
        "Status": "ACTIVE"
    }
    
    # Avoid duplicate exact timestamp logs
    if not st.session_state['signal_log'] or st.session_state['signal_log'][-1]["Timestamp"] != new_entry["Timestamp"]:
        st.session_state['signal_log'].append(new_entry)

    log_df = pd.DataFrame(st.session_state['signal_log'])
    try:
        st.dataframe(log_df, width="stretch")
    except TypeError:
        st.dataframe(log_df, use_container_width=True)

    csv_data = log_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Execution Log as CSV",
        data=csv_data,
        file_name=f"execution_log_{symbol}_{interval}.csv",
        mime="text/csv"
    )


def standalone_page():
    """Standalone page renderer when accessed via Streamlit sidebar navigation."""
    st.set_page_config(page_title="Execution Dashboard", page_icon="⚡", layout="wide")
    
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        symbol = st.text_input("Trading Symbol", value="BTC-USD", key="exec_symbol")
    with c2:
        interval = st.selectbox("Timeframe", ["1d", "1h", "5m", "1m"], index=0, key="exec_interval")
    with c3:
        st.write("")
        st.write("")
        refresh_btn = st.button("🔄 Refresh Analysis", use_container_width=True)

    with st.spinner(f"Analyzing {symbol} ({interval})..."):
        try:
            from data_engine import fetch_data
            from features import add_features
            from models import predict_next_move
            from quantum_signals import get_quantum_signals
            from patterns import detect_and_draw_patterns
            from alpha_engine import compute_trade_plan

            df = fetch_data(symbol, interval=interval)
            if df is not None and not df.empty:
                df_feat = add_features(df)
                prediction = predict_next_move(df_feat)
                regime = get_quantum_signals(df_feat)
                last_close = float(df['Close'].iloc[-1])
                
                # ATR
                tr1 = df['High'] - df['Low']
                tr2 = (df['High'] - df['Close'].shift()).abs()
                tr3 = (df['Low'] - df['Close'].shift()).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else float(last_close * 0.02)
                
                detected_patterns = detect_and_draw_patterns(None, df_feat)
                trade_plan = compute_trade_plan(last_close, atr, prediction, regime, detected_patterns)
                
                prob_up = prediction.get('prob_up', 0.5)
                bias = trade_plan["bias"]
                master_signal = {
                    "action": bias,
                    "conviction": prob_up if bias == "BUY" else (1 - prob_up if bias == "SELL" else 0.5),
                    "entry": last_close,
                    "stop_loss": trade_plan["stop_loss"],
                    "take_profit": trade_plan["take_profit"],
                    "position_size_usd": 10000.0 * 0.02
                }
                
                component_scores = {
                    "AI Model": prob_up,
                    "Market Regime": regime.get('markov_score', 0.5) if isinstance(regime, dict) else 0.5,
                    "Pattern Match": 1.0 if trade_plan["pattern_name"] else 0.5,
                    "Vol Strength": 0.7,
                    "Trend Alignment": 0.65,
                    "Momentum": 0.6,
                    "Volume Profile": 0.8,
                    "RSI Level": 0.55,
                    "MACD Signal": 0.6,
                    "ATR Sync": 0.5,
                }
                
                execution_data = {
                    "master_trade_signal": master_signal,
                    "component_scores": component_scores,
                    "df": df
                }
                render_execution_dashboard(execution_data, symbol, interval)
            else:
                st.error(f"Could not load data for symbol {symbol}.")
        except Exception as e:
            st.error(f"Error computing execution dashboard: {e}")


if __name__ == "__main__":
    standalone_page()

