import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf

from data_engine import fetch_data
from features import add_features
from models import TradingModel, run_monte_carlo, get_markov_regime
import portfolio as pf

st.set_page_config(page_title="Universal Trading Predictor", layout="wide")

st.title("📈 Universal Trading Predictor (Maximal Edge)")
st.markdown("""
This app applies **Machine Learning**, **Markov Chains**, and **Monte Carlo Simulations** 
to any tradable asset. *Note: 100% accuracy is impossible in finance. The goal is to 
maximize statistical edge and strictly manage risk.*
""")

# Sidebar inputs
st.sidebar.header("Configuration")
symbol = st.sidebar.text_input("Asset Symbol (e.g., BTC-USD, GC=F, AAPL, NG=F)", value="BTC-USD")
interval = st.sidebar.selectbox("Timeframe", options=["1d", "1h", "5m"], index=0)
period_map = {"1d": "2y", "1h": "1mo", "5m": "5d"}
period = period_map[interval]

force_retrain = st.sidebar.checkbox("Force Retrain AI Model", value=False)

st.sidebar.divider()
st.sidebar.header("Auto-Refresh")
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=False)

if auto_refresh:
    refresh_interval = st.sidebar.selectbox("Refresh Interval", options=["10s", "30s", "1m", "5m"], index=1)
    run_every_val = refresh_interval
    st.sidebar.info(f"Smooth auto-refresh enabled: {run_every_val}")
else:
    run_every_val = None

fetch_clicked = st.sidebar.button("Fetch & Analyze Data")

# First boot or forced manual fetch
if fetch_clicked or ('df' not in st.session_state):
    if fetch_clicked:
        with st.spinner(f"Fetching data for {symbol}..."):
            df = fetch_data(symbol, period=period, interval=interval)
    else:
        df = fetch_data(symbol, period=period, interval=interval)
        
    if df is not None and len(df) > 0:
        if fetch_clicked:
            with st.spinner("Calculating technical indicators..."):
                df_feat = add_features(df)
            with st.spinner("Running AI Models..."):
                model = TradingModel(symbol, interval)
                model.train_or_load(df_feat, force_retrain=force_retrain)
                prediction = model.predict_next(df_feat)
                regime = get_markov_regime(df_feat)
        else:
            df_feat = add_features(df)
            model = TradingModel(symbol, interval)
            model.train_or_load(df_feat, force_retrain=force_retrain)
            prediction = model.predict_next(df_feat)
            regime = get_markov_regime(df_feat)
            
        st.session_state['df'] = df
        st.session_state['df_feat'] = df_feat
        st.session_state['prediction'] = prediction
        st.session_state['regime'] = regime
        st.session_state['symbol'] = symbol
    elif fetch_clicked:
        st.error("Failed to fetch data or empty dataset returned. Check the symbol.")
        st.stop()


# Define the fragment that updates automatically without reloading the whole page!
@st.fragment(run_every=run_every_val)
def render_dashboard():
    # If auto-refresh is on, fetch the latest data QUIETLY inside the fragment
    if auto_refresh and 'df' in st.session_state and not fetch_clicked:
        df = fetch_data(symbol, period=period, interval=interval)
        if df is not None and len(df) > 0:
            df_feat = add_features(df)
            model = TradingModel(symbol, interval)
            model.train_or_load(df_feat, force_retrain=False)
            prediction = model.predict_next(df_feat)
            regime = get_markov_regime(df_feat)
            
            st.session_state['df'] = df
            st.session_state['df_feat'] = df_feat
            st.session_state['prediction'] = prediction
            st.session_state['regime'] = regime
            st.session_state['symbol'] = symbol

    if 'df' in st.session_state and st.session_state['symbol'] == symbol:
        df = st.session_state['df']
        df_feat = st.session_state['df_feat']
        prediction = st.session_state['prediction']
        regime = st.session_state['regime']
        
        last_close = df['Close'].iloc[-1]
        
        # Create Tabs
        tab1, tab2 = st.tabs(["🤖 AI Prediction Engine", "💼 Paper Trading Portfolio"])
        
        with tab1:
            # 1. Top Metrics
            prev_close = df['Close'].iloc[-2]
            pct_change = ((last_close - prev_close) / prev_close) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"{last_close:.2f}", f"{pct_change:.2f}%")
            col2.metric("Markov Regime Filter", regime)
            
            sig = prediction.get('signal', 'ERROR')
            prob_up = prediction.get('prob_up', 0.5) * 100
            color = "normal" if sig == "NEUTRAL" else "inverse" 
            col3.metric("AI Signal (Next Candle)", sig, f"{prob_up:.1f}% probability UP", delta_color=color if sig=="SELL (DOWN)" else "normal")
            
            # 2. Main Chart
            st.subheader(f"Price Action & Technicals ({symbol})")
            fig = go.Figure(data=[go.Candlestick(x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            name="Price")])
            
            if 'BB_High' in df_feat.columns:
                fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat['BB_High'], line=dict(color='gray', width=1, dash='dash'), name='BB High'))
                fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat['BB_Low'], line=dict(color='gray', width=1, dash='dash'), name='BB Low'))
                
            fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark', height=500, uirevision=f"{symbol}_{interval}", dragmode='pan')
            st.plotly_chart(fig, use_container_width=True)
            
            # 3. Monte Carlo Risk Management
            st.subheader("Monte Carlo Risk Analysis (Value at Risk)")
            st.markdown("Simulating 1,000 possible future price paths based on historical volatility.")
            
            with st.spinner("Running 1,000 Monte Carlo simulations..."):
                mc_results = run_monte_carlo(df_feat, days=20, simulations=1000)
                
            if mc_results:
                mc_fig = go.Figure()
                paths = mc_results['paths']
                for i in range(50):
                    mc_fig.add_trace(go.Scatter(y=paths[i], mode='lines', line=dict(color='rgba(0,100,255,0.1)'), showlegend=False))
                    
                mean_path = paths.mean(axis=1)
                mc_fig.add_trace(go.Scatter(y=mean_path, mode='lines', line=dict(color='white', width=3), name='Mean Expected'))
                
                mc_fig.update_layout(title="Future Price Paths (20 Candles)", template='plotly_dark', height=400, uirevision=f"{symbol}_{interval}", dragmode='pan')
                st.plotly_chart(mc_fig, use_container_width=True)
                
                st.info(f"**Risk Management Output:** The 95% Confidence Value-at-Risk (Worst Case) is **{mc_results['var_95']:.2f}**. Consider placing your hard stop-loss near this level.")
            else:
                st.warning("Not enough data to run Monte Carlo simulations.")
                
        with tab2:
            st.header("💼 Paper Trading Account")
            
            # Load state and fetch current prices for active positions to calculate accurate PnL
            port_state = pf.load_portfolio()
            current_prices = {symbol: last_close} # Always inject the currently viewed symbol
            
            # Fetch prices for other holdings (lazy load)
            for sym in port_state["positions"].keys():
                if sym != symbol:
                    try:
                        current_prices[sym] = yf.Ticker(sym).fast_info.last_price
                    except:
                        pass
                        
            summary = pf.get_portfolio_summary(current_prices)
            
            # 1. Account Summary
            c1, c2, c3 = st.columns(3)
            c1.metric("Available Cash", f"${summary['cash']:,.2f}")
            c2.metric("Unrealized PnL", f"${summary['unrealized_pnl']:,.2f}", f"{(summary['unrealized_pnl'] / (summary['total_value'] - summary['unrealized_pnl']) * 100) if summary['total_value'] > summary['cash'] else 0:.2f}%")
            c3.metric("Total Account Value", f"${summary['total_value']:,.2f}")
            
            st.divider()
            
            # 2. Execution Panel
            st.subheader(f"Trade Execution: {symbol}")
            st.write(f"**Current Price:** ${last_close:,.4f}")
            
            with st.form("trade_form"):
                col_a, col_b = st.columns(2)
                qty = col_a.number_input(f"Quantity of {symbol}", min_value=0.0, value=1.0, step=0.1)
                
                # Action buttons
                buy_btn = col_b.form_submit_button("🟩 BUY (Open Long / Close Short)")
                sell_btn = col_b.form_submit_button("🟥 SELL (Open Short / Close Long)")
                
                if buy_btn:
                    success, msg = pf.buy_asset(symbol, qty, last_close)
                    if success:
                        st.success(msg)
                        st.rerun(scope="fragment") # Refresh just this fragment
                    else:
                        st.error(msg)
                        
                if sell_btn:
                    success, msg = pf.sell_asset(symbol, qty, last_close)
                    if success:
                        st.success(msg)
                        st.rerun(scope="fragment")
                    else:
                        st.error(msg)
                        
            st.divider()
            
            # 3. Active Positions
            st.subheader("Active Positions")
            if len(summary['active_positions']) > 0:
                df_pos = pd.DataFrame(summary['active_positions'])
                # Format columns nicely
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
                st.info("No active positions.")
                
            st.divider()
            
            # 4. Trade History for Selected Symbol
            st.subheader(f"Trade History ({symbol})")
            
            sym_history = [h for h in summary['history'] if h['symbol'] == symbol]
            if len(sym_history) > 0:
                # Reverse to show newest first
                df_hist = pd.DataFrame(sym_history)[::-1]
                df_hist['price'] = df_hist['price'].apply(lambda x: f"${x:,.2f}")
                df_hist['total'] = df_hist['total'].apply(lambda x: f"${x:,.2f}")
                df_hist['realized_pnl'] = df_hist['realized_pnl'].apply(lambda x: f"${x:,.2f}")
                
                # Highlight Buy/Sell
                def color_action(val):
                    if 'BUY' in str(val): return 'color: green'
                    elif 'SELL' in str(val): return 'color: red'
                    return ''
                    
                st.dataframe(df_hist.style.map(color_action, subset=['action']), use_container_width=True)
            else:
                st.info(f"No previous trades found for {symbol}.")


# Call the fragment function
render_dashboard()
