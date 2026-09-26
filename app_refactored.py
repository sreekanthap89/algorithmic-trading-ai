"""
Trading App - Main Application
Refactored with modular components and comprehensive logging.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from data_engine import fetch_data
from features import add_features
from models import TradingModel, run_monte_carlo, get_markov_regime
import portfolio as pf
from utils.logging_setup import get_logger
from config.settings import DATA_SOURCES, SIGNAL_THRESHOLDS
from metrics.validation import PredictionTracker
from ml.model_versioning import ModelStorage
from metrics.feature_importance import FeatureImportanceAnalyzer
from backtesting.backtest_engine import Backtest

logger = get_logger(__name__)

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

st.set_page_config(page_title="Universal Trading Predictor", layout="wide")

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

def render_sidebar() -> tuple:
    """Render sidebar configuration and return user inputs."""
    st.sidebar.header("⚙️ Asset Configuration")
    
    symbol = st.sidebar.text_input(
        "Asset Symbol (e.g., BTC-USD, GC=F, AAPL)", 
        value="BTC-USD"
    )
    interval = st.sidebar.selectbox(
        "Timeframe", 
        options=["1d", "1h", "5m"], 
        index=0
    )
    period = DATA_SOURCES[interval]["period"]
    
    force_retrain = st.sidebar.checkbox("Force Retrain AI Models", value=False)
    
    st.sidebar.divider()
    st.sidebar.header("⏱️ Live Data Auto-Refresh")
    auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=False)
    
    run_every_val = None
    if auto_refresh:
        refresh_interval = st.sidebar.selectbox(
            "Refresh Interval", 
            options=["10s", "30s", "1m", "5m"], 
            index=1
        )
        run_every_val = refresh_interval
        st.sidebar.info(f"Auto-refresh active: {run_every_val}")
    
    fetch_clicked = st.sidebar.button(
        "🔄 Refresh & Analyze Data", 
        type="primary", 
        use_container_width=True
    )
    
    st.sidebar.divider()
    st.sidebar.header("📊 Advanced Options")
    show_backtest = st.sidebar.checkbox("Show Backtesting Tab", value=False)
    show_metrics = st.sidebar.checkbox("Show Feature Importance", value=False)
    
    return symbol, interval, period, force_retrain, auto_refresh, run_every_val, fetch_clicked, show_backtest, show_metrics


# ============================================================================
# DATA & MODEL PROCESSING
# ============================================================================

from alpha_engine import MultiAlphaEngine, compute_bid_ask_imbalance
from simulation import run_heston_monte_carlo
from ml.stacking_ensemble import UltraStackingEnsemble
from quant_engine import generate_master_trade_signal, fuse_signals
from pages.execution_dashboard import render_execution_dashboard
from quantum_signals import compute_fft_cycles, compute_hawkes_excitation, compute_maxent_distribution, compute_hmm_regime
from features import add_volume_profile, compute_var

def load_and_process_data(symbol: str, interval: str, period: str, force_retrain: bool) -> dict:
    """Load data, engineer features, and train model."""
    logger.info(f"Loading data for {symbol} ({interval})")
    
    with st.spinner(f"Fetching market data for {symbol} ({interval})..."):
        df = fetch_data(symbol, period=period, interval=interval)
    
    if df is None or len(df) == 0:
        st.error(f"Failed to fetch data for {symbol}")
        st.stop()
    
    with st.spinner("Engineering technical indicators & volume profiles..."):
        df_feat = add_features(df)
    
    if df_feat is None or len(df_feat) < 40:
        st.error("Insufficient data for model training")
        st.stop()
    
    with st.spinner("Training AI ensemble & Stacking models..."):
        model = TradingModel(symbol, interval)
        model.train_or_load(df_feat, force_retrain=force_retrain)
        prediction = model.predict_next(df_feat)
        regime = get_markov_regime(df_feat)

        # Stacking Ensemble
        stacking = UltraStackingEnsemble(symbol, interval)
        stacking.train_or_load(df_feat, force_retrain=force_retrain)
        stacking_res = stacking.predict_next(df_feat)

        # Multi-Alpha Engine
        alpha_eng = MultiAlphaEngine(requires_n_confirms=2)
        alpha_res = alpha_eng.compute_alphas(df_feat)

        # Heston Monte Carlo
        heston_res = run_heston_monte_carlo(df_feat, days=20, simulations=2000)

        # Volume Profile & VaR
        vol_prof = add_volume_profile(df_feat)
        var_res = compute_var(df_feat)

        # Quantum signals
        fft_res = compute_fft_cycles(df_feat)
        hawkes_res = compute_hawkes_excitation(df_feat)
        maxent_res = compute_maxent_distribution(df_feat)

        # 10 Component Scores for Fusion Engine
        component_scores = {
            "data_quality": 0.95,
            "volume_profile": 0.65 if 'Close' in df_feat.columns and df_feat['Close'].iloc[-1] > vol_prof.get('poc', 0) else 0.45,
            "var_risk": float(np.clip(1.0 - var_res.get('historical_var', 0.05) * 5.0, 0.1, 0.9)),
            "monte_carlo": 0.60 if heston_res and heston_res['mean_expected'] > df_feat['Close'].iloc[-1] else 0.40,
            "markov_regime": regime.get('markov_score', 0.5),
            "multi_alpha": alpha_res.get('composite_conviction', 0.5),
            "microstructure": float(np.clip(1.0 - compute_bid_ask_imbalance(df_feat) * 0.5, 0.1, 0.9)),
            "deep_learning": prediction.get('prob_up', 0.5),
            "stacking_ensemble": stacking_res.get('stacking_conviction', 0.5),
            "quantum_signals": float(np.mean([
                fft_res.get('spectral_score', 0.5),
                hawkes_res.get('hawkes_score', 0.5),
                maxent_res.get('maxent_score', 0.5)
            ]))
        }

        # Master Trade Signal with Kelly position sizing
        master_trade_signal = generate_master_trade_signal(df_feat, component_scores)
    
    logger.info(f"Data loading complete for {symbol}_{interval}")
    
    return {
        "df": df,
        "df_feat": df_feat,
        "model": model,
        "prediction": prediction,
        "regime": regime,
        "stacking_res": stacking_res,
        "alpha_res": alpha_res,
        "heston_res": heston_res,
        "vol_prof": vol_prof,
        "var_res": var_res,
        "component_scores": component_scores,
        "master_trade_signal": master_trade_signal
    }



# ============================================================================
# CHART RENDERING
# ============================================================================

def render_chart(df: pd.DataFrame, df_feat: pd.DataFrame, prediction: dict, 
                 symbol: str, show_indicators: bool = True) -> go.Figure:
    """Create TradingView-style chart with indicators."""
    
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Price",
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350'
    ))
    
    if show_indicators and 'SMA_20' in df_feat.columns:
        # Moving averages
        fig.add_trace(go.Scatter(
            x=df_feat.index, y=df_feat['SMA_20'],
            mode='lines', name='SMA 20',
            line=dict(color='#FFD700', width=1.5)
        ))
        fig.add_trace(go.Scatter(
            x=df_feat.index, y=df_feat['SMA_50'],
            mode='lines', name='SMA 50',
            line=dict(color='#00FFFF', width=1.5)
        ))
        
        # Bollinger Bands
        if 'BB_High' in df_feat.columns:
            fig.add_trace(go.Scatter(
                x=df_feat.index, y=df_feat['BB_High'],
                mode='lines', name='BB Upper',
                line=dict(color='rgba(200,200,200,0.3)', width=1),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=df_feat.index, y=df_feat['BB_Low'],
                mode='lines', name='BB Lower',
                line=dict(color='rgba(200,200,200,0.3)', width=1),
                fill='tonexty', fillcolor='rgba(200,200,200,0.1)'
            ))
    
    fig.update_layout(
        title=f"{symbol} Price Chart",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
        height=600,
        template="plotly_dark"
    )
    
    return fig


# ============================================================================
# DASHBOARD TABS
# ============================================================================

def render_main_dashboard(data: dict, symbol: str, interval: str):
    """Main trading dashboard."""
    
    df = data["df"]
    df_feat = data["df_feat"]
    prediction = data["prediction"]
    regime = data["regime"]
    
    last_close = float(df['Close'].iloc[-1])
    prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else last_close
    pct_change = ((last_close - prev_close) / prev_close) * 100.0
    
    # Executive summary metrics
    sig = prediction.get('signal', 'NEUTRAL')
    prob_up = prediction.get('prob_up', 0.5)
    
    st.markdown("### 📊 Executive Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(
        "Current Price",
        f"${last_close:,.2f}",
        f"{pct_change:+.2f}%"
    )
    col2.metric(
        "Market Regime",
        f"{'🟢' if 'BULL' in regime.get('regime', '') else '🔴'} {regime.get('regime', 'CHOPPY')}"
    )
    col3.metric(
        "AI Signal",
        sig,
        f"{prob_up*100:.1f}%"
    )
    col4.metric(
        "Model Confidence",
        f"{abs(prob_up - 0.5) * 200:.1f}%"
    )
    
    # Recommendation banner
    st.divider()
    st.markdown("### 🎯 Trading Recommendation")
    
    quant_info = prediction.get('quantile_signals', {})
    q10 = quant_info.get('q10_price', last_close * 0.96)
    q90 = quant_info.get('q90_price', last_close * 1.04)
    
    if sig == "BUY (UP)":
        st.success(f"**🟢 BUY SIGNAL**\n"
                   f"- Entry: ${last_close:,.2f}\n"
                   f"- Take Profit: ${q90:,.2f} ({((q90-last_close)/last_close)*100:+.2f}%)\n"
                   f"- Stop Loss: ${q10:,.2f}")
    elif sig == "SELL (DOWN)":
        st.error(f"**🔴 SELL SIGNAL**\n"
                 f"- Entry: ${last_close:,.2f}\n"
                 f"- Take Profit: ${q10:,.2f} ({((q10-last_close)/last_close)*100:+.2f}%)\n"
                 f"- Stop Loss: ${q90:,.2f}")
    else:
        st.warning(f"**🟡 NEUTRAL**\n"
                   f"- Market consolidating between ${q10:,.2f} and ${q90:,.2f}")
    
    # Chart
    st.divider()
    st.markdown("### 📈 Price Chart & Indicators")
    fig = render_chart(df, df_feat, prediction, symbol)
    st.plotly_chart(fig, use_container_width=True)


def render_model_analytics(data: dict, symbol: str, interval: str):
    """Model analytics tab."""
    
    model = data["model"]
    df_feat = data["df_feat"]
    prediction = data["prediction"]
    
    st.markdown("### 🤖 Model Performance")
    
    # Model votes
    model_votes = prediction.get('model_votes', {})
    if model_votes:
        vote_df = pd.DataFrame(
            list(model_votes.items()),
            columns=['Model', 'Probability UP']
        )
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=vote_df['Model'],
            y=vote_df['Probability UP'],
            marker_color=['green' if v > 0.5 else 'red' for v in vote_df['Probability UP']]
        ))
        fig.update_layout(title="Base Model Predictions", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Signal breakdown
    st.markdown("### 📡 Signal Breakdown")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(
        "ML Ensemble",
        f"{prediction.get('ml_prob_up', 0.5)*100:.1f}%"
    )
    col2.metric(
        "Quantum Score",
        f"{prediction.get('quantum_signals', {}).get('quantum_score', 0.5)*100:.1f}%"
    )
    col3.metric(
        "Quantile Score",
        f"{prediction.get('quantile_signals', {}).get('quantile_score', 0.5)*100:.1f}%"
    )
    col4.metric(
        "Markov Score",
        f"{prediction.get('markov_regime', {}).get('markov_score', 0.5)*100:.1f}%"
    )


def render_portfolio_tab(symbol: str):
    """Portfolio management tab."""
    
    st.markdown("### 💼 Paper Trading Portfolio")
    
    port_state = pf.load_portfolio()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Available Cash", f"${port_state.get('cash', 0):,.2f}")
        st.metric("Number of Positions", len(port_state.get('positions', {})))
    
    with col2:
        total_value = port_state.get('cash', 0)
        for pos_symbol, pos_data in port_state.get('positions', {}).items():
            total_value += pos_data.get('quantity', 0) * pos_data.get('average_price', 0)
        st.metric("Total Portfolio Value", f"${total_value:,.2f}")
    
    # Positions
    st.markdown("#### Open Positions")
    positions = port_state.get('positions', {})
    if positions:
        pos_list = []
        for pos_symbol, pos_data in positions.items():
            pos_list.append({
                "Symbol": pos_symbol,
                "Quantity": pos_data.get('quantity', 0),
                "Avg Price": f"${pos_data.get('average_price', 0):.2f}"
            })
        st.dataframe(pd.DataFrame(pos_list), use_container_width=True)
    else:
        st.info("No open positions")


def render_backtesting_tab(data: dict, symbol: str, interval: str):
    """Backtesting tab."""
    
    st.markdown("### 📈 Strategy Backtesting")
    
    df = data["df"]
    df_feat = data["df_feat"]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        entry_threshold = st.slider("Entry Threshold", 0.5, 1.0, 0.56)
    with col2:
        exit_threshold = st.slider("Exit Threshold", 0.0, 0.5, 0.44)
    with col3:
        stop_loss_pct = st.slider("Stop Loss %", 0.01, 0.1, 0.05)
    
    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            df_test = df_feat.copy()
            df_test['prob_up'] = data["prediction"].get('prob_up', 0.5)
            
            backtest = Backtest(symbol, interval)
            results = backtest.run(
                df_test,
                signal_col='prob_up',
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                stop_loss_pct=stop_loss_pct
            )
            
            # Display results
            ret = results["returns"]
            risk = results["risk"]
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Return", f"{risk['total_return']*100:+.2f}%")
            col2.metric("Win Rate", f"{ret['win_rate']*100:.1f}%")
            col3.metric("Profit Factor", f"{ret['profit_factor']:.2f}")
            col4.metric("Max Drawdown", f"{risk['max_drawdown']*100:.2f}%")
            col5.metric("Sharpe Ratio", f"{risk['sharpe_ratio']:.2f}")
            
            # Equity curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=results['equity_curve'],
                mode='lines',
                fill='tozeroy',
                name='Equity'
            ))
            fig.update_layout(title="Equity Curve", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Save results
            filepath = backtest.save_results()
            st.success(f"Results saved to {filepath}")


def render_feature_importance_tab(data: dict, symbol: str, interval: str):
    """Feature importance tab."""
    
    st.markdown("### 🎯 Feature Importance Analysis")
    
    model = data["model"]
    df_feat = data["df_feat"]
    
    if st.button("Compute Feature Importance", type="primary"):
        with st.spinner("Analyzing feature importance..."):
            analyzer = FeatureImportanceAnalyzer(symbol, interval)
            
            # Use last portion of data for training
            train_df = df_feat.dropna(subset=model.feature_cols + ['Target']).copy()
            X = train_df[model.feature_cols]
            y = train_df['Target']
            
            analysis = analyzer.analyze_all_methods(
                X, y,
                model.models if hasattr(model, 'models') else {},
                model.meta_stacker if hasattr(model, 'meta_stacker') else None
            )
            
            # Display report
            report = analyzer.generate_importance_report()
            st.text(report)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    
    st.title("📈 Universal Trading Predictor (Smart Decision Suite)")
    st.markdown("AI-powered trading analysis with ML, quantum signals, and backtesting")
    
    # Sidebar
    symbol, interval, period, force_retrain, auto_refresh, run_every_val, fetch_clicked, show_backtest, show_metrics = render_sidebar()
    
    # Check if we need to fetch/refetch data
    need_refetch = (
        fetch_clicked or
        'data' not in st.session_state or
        st.session_state.get('symbol') != symbol or
        st.session_state.get('interval') != interval
    )
    
    if need_refetch:
        try:
            data = load_and_process_data(symbol, interval, period, force_retrain)
            st.session_state['data'] = data
            st.session_state['symbol'] = symbol
            st.session_state['interval'] = interval
        except Exception as e:
            st.error(f"Error loading data: {e}")
            logger.error(f"Data loading error: {e}", exc_info=True)
            st.stop()
    
    if 'data' not in st.session_state:
        st.info("👈 Configure settings and click 'Refresh & Analyze Data' to begin")
        return
    
    data = st.session_state['data']
    
    # Main tabs
    tabs = ["🏠 Dashboard", "⚡ Master Execution", "🤖 Model Analytics", "💼 Portfolio"]
    if show_backtest:
        tabs.append("📈 Backtesting")
    if show_metrics:
        tabs.append("🎯 Feature Importance")
    
    selected_tab = st.tabs(tabs)
    
    with selected_tab[0]:
        render_main_dashboard(data, symbol, interval)
    
    with selected_tab[1]:
        render_execution_dashboard(data, symbol, interval)

    with selected_tab[2]:
        render_model_analytics(data, symbol, interval)
    
    with selected_tab[3]:
        render_portfolio_tab(symbol)
    
    if show_backtest:
        with selected_tab[4]:
            render_backtesting_tab(data, symbol, interval)
    
    if show_metrics:
        tab_idx = 5 if show_backtest else 4
        with selected_tab[tab_idx]:
            render_feature_importance_tab(data, symbol, interval)


if __name__ == "__main__":
    main()

