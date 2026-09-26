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
    if df is not None and len(df) > 20:
        df_pnl = df.copy()
        df_pnl['Hour'] = pd.to_datetime(df_pnl.index).hour
        df_pnl['Day'] = pd.to_datetime(df_pnl.index).day_name()
        df_pnl['Return_Pct'] = df_pnl['Close'].pct_change() * 100.0

        pivot_table = df_pnl.pivot_table(index='Day', columns='Hour', values='Return_Pct', aggfunc='mean').fillna(0)
        
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_table = pivot_table.reindex([d for d in days_order if d in pivot_table.index])

        fig_heat = px.imshow(
            pivot_table,
            labels=dict(x="Hour of Day", y="Day of Week", color="Avg Return (%)"),
            color_continuous_scale="RdYlGn",
            title="Expected Return Density by Time Slot"
        )
        fig_heat.update_layout(height=350, template="plotly_dark")
        st.plotly_chart(fig_heat, use_container_width=True)

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
    st.dataframe(log_df, use_container_width=True)

    csv_data = log_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Execution Log as CSV",
        data=csv_data,
        file_name=f"execution_log_{symbol}_{interval}.csv",
        mime="text/csv"
    )
