"""
=============================================================
 EXECUTION & AI CONSENSUS ANALYTICS DASHBOARD (Trading App)
 Steps 12-13 Implementation + TimesFM-3 Consensus Upgrade
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
    Renders the Master Execution & AI Consensus Dashboard:
      - Plain-English Signal Alert Banner
      - 7-Engine AI Voting Breakdown Cards (if consensus_report present)
      - Conviction Score Gauge & 10-Pillar Signal Fusion Radar Chart
      - PnL Seasonality / Time-Slot Heatmap
      - Real-time Signal Execution Log with CSV download
    """
    st.markdown(f"## ⚡ AI Consensus & Execution Center — `{symbol}` ({interval})")

    master_signal = data.get("master_trade_signal", {})
    component_scores = data.get("component_scores", {})
    consensus = data.get("consensus_report", {})

    action = master_signal.get("action", "HOLD")
    conviction = float(master_signal.get("conviction", 0.5))
    entry = float(master_signal.get("entry", 0.0))
    sl = float(master_signal.get("stop_loss", 0.0))
    tp = float(master_signal.get("take_profit", 0.0))
    position_usd = float(master_signal.get("position_size_usd", 0.0))

    from config.market_catalog import format_price
    fmt_entry = format_price(entry, symbol)
    fmt_tp = format_price(tp, symbol)
    fmt_sl = format_price(sl, symbol)

    # 1. Plain-English Trade Alert Banner
    if action == "BUY" and conviction >= 0.56:
        st.success(
            f"### 🟢 {consensus.get('verdict', 'BULLISH BUY SIGNAL')} FIRED!\n"
            f"**Plain-English Summary:** {consensus.get('explanation', 'AI models project upward price momentum.')}\n\n"
            f"**Entry**: `{fmt_entry}` · **Target (Take-Profit)**: `{fmt_tp}` · "
            f"**Safety Exit (Stop-Loss)**: `{fmt_sl}` · **Suggested Size (2% Risk)**: `${position_usd:,.2f}`"
        )
    elif action == "SELL" and conviction >= 0.56:
        st.error(
            f"### 🔴 {consensus.get('verdict', 'BEARISH SELL SIGNAL')} FIRED!\n"
            f"**Plain-English Summary:** {consensus.get('explanation', 'AI models project downward price pressure.')}\n\n"
            f"**Entry**: `{fmt_entry}` · **Target (Take-Profit)**: `{fmt_tp}` · "
            f"**Safety Exit (Stop-Loss)**: `{fmt_sl}` · **Suggested Size (2% Risk)**: `${position_usd:,.2f}`"
        )
    else:
        st.info(
            f"### 🟡 WAIT / HOLD ZONE — PROTECT CAPITAL\n"
            f"**Plain-English Summary:** {consensus.get('explanation', 'Market signals are mixed or consolidating. Wait for a higher-conviction breakout.')}"
        )

    # 2. Institutional Accuracy & Meta-Labeling Verification Strip
    comp_oos = float(consensus.get("composite_oos_accuracy", 0.65)) * 100.0
    meta_rel = float(consensus.get("meta_label_prob", 0.65)) * 100.0
    bma_w = consensus.get("bma_weights", {})
    top_bma = max(bma_w.items(), key=lambda kv: kv[1])[0] if bma_w else "TimesFM-3 Forecast"
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric(
        "Validated Out-of-Sample (OOS) Skill",
        f"{comp_oos:.1f}%",
        "Purged Walk-Forward Accuracy / AUC",
    )
    m_col2.metric(
        "Meta-Label Signal Reliability",
        f"{meta_rel:.1f}%",
        "López de Prado Confidence Filter",
    )
    m_col3.metric(
        "Highest-Skill Bayesian Driver",
        top_bma,
        f"Weight: {bma_w.get(top_bma, 0.25)*100:.1f}% (Dynamic OOS BMA)",
    )

    # 3. 7-Engine AI Consensus Voting Breakdown (Simple & Visual)
    model_votes = consensus.get("model_votes", [])
    if model_votes:
        st.markdown("### 🤝 How the 7 AI & Quantitative Engines Voted")
        st.caption(
            "Each independent engine analyzes the market from a different angle—combining Google Research's "
            "**TimesFM-3 Foundation Model**, tree ensembles, market regimes, and institutional volume flow."
        )
        cols = st.columns(len(model_votes))
        for idx, vote_item in enumerate(model_votes):
            v = vote_item["vote"]
            badge_color = "#00E676" if v == "BUY" else ("#FF5252" if v == "SELL" else "#FFD54F")
            bg_color = (
                "rgba(0, 230, 118, 0.10)"
                if v == "BUY"
                else ("rgba(255, 82, 82, 0.10)" if v == "SELL" else "rgba(255, 213, 79, 0.08)")
            )
            with cols[idx]:
                st.markdown(
                    f"""
                    <div style="background:{bg_color}; border:1px solid {badge_color}; border-radius:10px; padding:12px; min-height:175px;">
                        <div style="font-size:1.15rem; font-weight:700;">{vote_item['icon']} {vote_item['engine']}</div>
                        <div style="margin:6px 0; font-size:0.95rem; font-weight:800; color:{badge_color};">
                            {v} ({vote_item['prob_up']:.0f}% Bullish)
                        </div>
                        <div style="font-size:0.78rem; color:#CBD5E1; line-height:1.35;">
                            {vote_item['plain_english']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.divider()

    # 3. Conviction Gauge & 10-Pillar Radar Chart
    col_gauge, col_radar = st.columns(2)

    with col_gauge:
        st.markdown("#### 🎯 Master AI Conviction Meter")
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=conviction * 100.0,
                number={"suffix": "%"},
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": f"Directional Conviction ({action})"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {
                        "color": "#00E676"
                        if action == "BUY"
                        else ("#FF5252" if action == "SELL" else "#FFD54F")
                    },
                    "steps": [
                        {"range": [0, 45], "color": "rgba(255, 82, 82, 0.18)"},
                        {"range": [45, 60], "color": "rgba(255, 235, 59, 0.18)"},
                        {"range": [60, 100], "color": "rgba(0, 230, 118, 0.18)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 4},
                        "thickness": 0.75,
                        "value": 60.0,
                    },
                },
            )
        )
        fig_gauge.update_layout(height=350, template="plotly_dark", margin=dict(l=25, r=25, t=45, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_radar:
        st.markdown("#### 🕸️ 10-Pillar Signal Fusion Radar")
        if component_scores:
            categories = list(component_scores.keys())
            values = [float(v) for v in component_scores.values()]

            fig_radar = go.Figure()
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name="Pillar Score (0..1)",
                    line_color="#29B6F6",
                    fillcolor="rgba(41, 182, 246, 0.25)",
                )
            )
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                height=350,
                template="plotly_dark",
                margin=dict(l=40, r=40, t=45, b=20),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # 4. PnL Heatmap Analytics
    st.markdown("### 📊 Historical Return Seasonality Heatmap")
    st.caption("Shows which days of the week and hours/months historically deliver the strongest average returns.")
    df = data.get("df", None)
    if df is not None and len(df) > 5:
        df_pnl = df.copy()
        dt_idx = pd.to_datetime(df_pnl.index)
        df_pnl["Day"] = dt_idx.day_name()
        df_pnl["Return_Pct"] = df_pnl["Close"].pct_change() * 100.0

        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        has_intraday_hours = dt_idx.hour.nunique() > 1
        if has_intraday_hours:
            df_pnl["TimeSlot"] = dt_idx.hour
            x_label = "Hour of Day (UTC)"
            chart_title = "Average Return (%) by Day of Week & Hour of Day"
            pivot_table = (
                df_pnl.pivot_table(index="Day", columns="TimeSlot", values="Return_Pct", aggfunc="mean").fillna(0)
            )
        else:
            df_pnl["TimeSlot"] = dt_idx.strftime("%b")
            months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            x_label = "Month of Year"
            chart_title = "Average Return (%) by Day of Week & Month"
            pivot_table = (
                df_pnl.pivot_table(index="Day", columns="TimeSlot", values="Return_Pct", aggfunc="mean").fillna(0)
            )
            pivot_table = pivot_table.reindex(columns=[m for m in months_order if m in pivot_table.columns])

        pivot_table = pivot_table.reindex([d for d in days_order if d in pivot_table.index])

        fig_heat = px.imshow(
            pivot_table,
            labels=dict(x=x_label, y="Day of Week", color="Avg Return (%)"),
            color_continuous_scale="RdYlGn",
            title=chart_title,
            aspect="auto",
        )
        fig_heat.update_layout(height=340, template="plotly_dark")
        try:
            st.plotly_chart(fig_heat, width="stretch")
        except TypeError:
            st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("ℹ️ Fetch data to generate the Return Seasonality Heatmap.")

    st.divider()

    # 5. Signal Execution History Log & CSV Export
    st.markdown("### 📜 Real-Time Signal Audit Log")

    if "signal_log" not in st.session_state:
        st.session_state["signal_log"] = []

    new_entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Symbol": symbol,
        "Interval": interval,
        "Action": action,
        "Conviction": f"{conviction*100:.1f}%",
        "Entry": fmt_entry,
        "Stop Loss": fmt_sl,
        "Take Profit": fmt_tp,
        "Allocation": f"${position_usd:,.2f}",
        "Status": "ACTIVE",
    }

    if not st.session_state["signal_log"] or st.session_state["signal_log"][-1]["Timestamp"] != new_entry["Timestamp"]:
        st.session_state["signal_log"].append(new_entry)

    log_df = pd.DataFrame(st.session_state["signal_log"])
    try:
        st.dataframe(log_df, width="stretch")
    except TypeError:
        st.dataframe(log_df, use_container_width=True)

    safe_sym = (
        str(symbol)
        .replace("=", "_")
        .replace("^", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace("\\", "_")
    )
    csv_data = log_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Signal Audit Log as CSV",
        data=csv_data,
        file_name=f"execution_log_{safe_sym}_{interval}.csv",
        mime="text/csv",
        key=f"dl_exec_log_{safe_sym}_{interval}",
    )


def standalone_page():
    """Standalone page renderer when accessed via Streamlit sidebar navigation."""
    st.set_page_config(page_title="Execution Dashboard", page_icon="⚡", layout="wide")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        symbol = st.text_input("Trading Symbol", value="BTC-USD", key="exec_symbol").upper().strip()
    with c2:
        interval = st.selectbox("Timeframe", ["1d", "1h", "5m", "1m"], index=0, key="exec_interval")
    with c3:
        st.write("")
        st.write("")
        st.button("🔄 Refresh Analysis", use_container_width=True)

    with st.spinner(f"Analyzing {symbol} ({interval})..."):
        try:
            from data_engine import fetch_data
            from features import add_features
            from models import predict_next_move, get_markov_regime
            from patterns import detect_and_draw_patterns
            from ml.timesfm_engine import run_timesfm_forecast
            from quant_engine import build_ai_consensus_report

            df = fetch_data(symbol, interval=interval)
            if df is not None and not df.empty:
                df_feat = add_features(df)
                prediction = predict_next_move(df_feat, symbol=symbol, interval=interval)
                regime = get_markov_regime(df_feat)
                last_close = float(df["Close"].iloc[-1])
                atr = (
                    float(df_feat["ATR"].iloc[-1])
                    if "ATR" in df_feat.columns and pd.notna(df_feat["ATR"].iloc[-1])
                    else float(last_close * 0.02)
                )
                detected_patterns = detect_and_draw_patterns(None, df_feat, has_sub=False)
                timesfm_res = prediction.get("timesfm") or run_timesfm_forecast(
                    df_feat, symbol=symbol, interval=interval, horizon=15
                )
                consensus = build_ai_consensus_report(
                    df_feat=df_feat,
                    symbol=symbol,
                    interval=interval,
                    xgb_prediction=prediction,
                    regime=regime,
                    detected_patterns=detected_patterns,
                    timesfm_res=timesfm_res,
                )

                bias = consensus.get("direction_bias", "NEUTRAL")
                stop_dist = 2.0 * atr
                sl = last_close - stop_dist if bias != "SELL" else last_close + stop_dist
                tp = last_close + 2.0 * stop_dist if bias != "SELL" else last_close - 2.0 * stop_dist

                master_signal = {
                    "action": bias if bias != "NEUTRAL" else "HOLD",
                    "conviction": consensus.get("directional_conviction", 0.5),
                    "entry": last_close,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "position_size_usd": consensus.get("kelly_sizing", {}).get("position_usd", 2000.0),
                }

                execution_data = {
                    "master_trade_signal": master_signal,
                    "component_scores": consensus.get("component_scores", {}),
                    "consensus_report": consensus,
                    "df": df,
                }
                render_execution_dashboard(execution_data, symbol, interval)
            else:
                st.error(f"Could not load data for symbol {symbol}.")
        except Exception as e:
            st.error(f"Error computing execution dashboard: {e}")


if __name__ == "__main__":
    standalone_page()
