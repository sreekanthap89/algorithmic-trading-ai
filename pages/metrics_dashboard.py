"""
Real-time metrics dashboard for model and portfolio performance monitoring.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
from typing import Dict

from config.settings import METRICS_DIR, BACKTEST_DIR
from metrics.validation import PredictionTracker
from ml.model_versioning import ModelRegistry
from utils.logging_setup import get_logger

logger = get_logger(__name__)

class MetricsDashboard:
    """Real-time metrics dashboard."""
    
    @staticmethod
    def render_prediction_metrics(symbol: str, interval: str):
        """Render prediction accuracy metrics."""
        st.subheader(f"📊 Prediction Metrics - {symbol} ({interval})")
        
        tracker = PredictionTracker(symbol, interval)
        df_predictions = tracker.load_historical_predictions()
        
        if df_predictions.empty:
            st.info("No prediction data available yet")
            return
        
        # Recent predictions
        st.markdown("**Recent Predictions**")
        recent = df_predictions.tail(20).copy()
        recent['timestamp'] = pd.to_datetime(recent['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(recent[['timestamp', 'signal', 'prob_up', 'actual_result']], use_container_width=True)
        
        # Drift analysis
        drift_metrics = tracker.calculate_drift(window=100)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Predictions", len(df_predictions))
        col2.metric("Drift Detected", "⚠️ Yes" if drift_metrics["drift_detected"] else "✓ No")
        col3.metric("Drift Magnitude", f"{drift_metrics['drift_magnitude']:.4f}")
        col4.metric("Recent Mean Prob", f"{drift_metrics['mean_prob_second_half']:.2f}")
        
        # Probability distribution chart
        if not df_predictions.empty:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df_predictions['prob_up'],
                nbinsx=20,
                name='Probability UP',
                opacity=0.7
            ))
            fig.update_layout(
                title="Prediction Probability Distribution",
                xaxis_title="Probability UP",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_model_versions(symbol: str, interval: str):
        """Render model version history."""
        st.subheader(f"🤖 Model Versions - {symbol} ({interval})")
        
        registry = ModelRegistry()
        versions = registry.list_versions(symbol, interval)
        
        if not versions:
            st.info("No model versions found")
            return
        
        # Version comparison table
        version_data = []
        for v in versions:
            metrics = v.get("metrics", {})
            version_data.append({
                "Version": v["version_id"],
                "Timestamp": v["timestamp"][:19],
                "Status": v.get("status", "active"),
                "Accuracy": f"{metrics.get('accuracy', 0):.4f}",
                "F1 Score": f"{metrics.get('f1_score', 0):.4f}",
                "ROC AUC": f"{metrics.get('roc_auc', 0):.4f}"
            })
        
        df_versions = pd.DataFrame(version_data)
        st.dataframe(df_versions, use_container_width=True)
        
        # Latest model metrics
        if versions:
            latest = versions[-1]
            st.markdown("**Latest Model Metrics**")
            metrics = latest.get("metrics", {})
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
            col2.metric("Precision", f"{metrics.get('precision', 0):.4f}")
            col3.metric("Recall", f"{metrics.get('recall', 0):.4f}")
            col4.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
            col5.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.4f}")
    
    @staticmethod
    def render_backtest_results():
        """Render recent backtest results."""
        st.subheader("📈 Recent Backtest Results")
        
        backtest_files = []
        if os.path.exists(BACKTEST_DIR):
            backtest_files = [f for f in os.listdir(BACKTEST_DIR) if f.endswith('.json')]
        
        if not backtest_files:
            st.info("No backtest results available")
            return
        
        # Select backtest to view
        selected_file = st.selectbox("Select backtest", sorted(backtest_files, reverse=True))
        
        if selected_file:
            filepath = os.path.join(BACKTEST_DIR, selected_file)
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            # Summary metrics
            ret = results["returns"]
            risk = results["risk"]
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Return", f"{risk['total_return']*100:.2f}%")
            col2.metric("Win Rate", f"{ret['win_rate']*100:.1f}%")
            col3.metric("Profit Factor", f"{ret['profit_factor']:.2f}")
            col4.metric("Max Drawdown", f"{risk['max_drawdown']*100:.2f}%")
            col5.metric("Sharpe Ratio", f"{risk['sharpe_ratio']:.2f}")
            
            # Equity curve
            if "equity_curve" in results:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=results['equity_curve'],
                    mode='lines',
                    name='Equity',
                    fill='tozeroy'
                ))
                fig.update_layout(
                    title="Equity Curve",
                    xaxis_title="Trades",
                    yaxis_title="Capital ($)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Trades table
            if "trades" in results and results["trades"]:
                st.markdown("**Trade History**")
                trades_df = pd.DataFrame(results["trades"])
                trades_df = trades_df[['entry_index', 'exit_index', 'entry_price', 'exit_price', 'pnl', 'pnl_pct']]
                st.dataframe(trades_df, use_container_width=True)
    
    @staticmethod
    def render_feature_importance(symbol: str, interval: str):
        """Render feature importance analysis."""
        st.subheader(f"🎯 Feature Importance - {symbol} ({interval})")
        
        importance_file = os.path.join(
            METRICS_DIR,
            f"{symbol}_{interval}_feature_importance.json"
        )
        
        if not os.path.exists(importance_file):
            st.info("No feature importance data available. Run model training first.")
            return
        
        try:
            with open(importance_file, 'r') as f:
                analysis = json.load(f)
            
            # Method selector
            methods = list(analysis.get("methods", {}).keys())
            selected_method = st.selectbox("Select importance method", methods)
            
            if selected_method:
                importances = analysis["methods"][selected_method]
                
                # Sort and display
                sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                features = [x[0] for x in sorted_imp[:15]]
                values = [x[1] for x in sorted_imp[:15]]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=values,
                    y=features,
                    orientation='h'
                ))
                fig.update_layout(
                    title=f"Top 15 Features - {selected_method}",
                    xaxis_title="Importance Score",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed table
                st.markdown("**All Features**")
                imp_df = pd.DataFrame(sorted_imp, columns=["Feature", "Importance"])
                st.dataframe(imp_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading feature importance: {e}")
            logger.error(f"Feature importance error: {e}")


def render_dashboard():
    """Main dashboard renderer."""
    st.set_page_config(page_title="Trading App - Real-time Metrics", layout="wide")
    
    st.title("📊 Trading App - Real-time Metrics Dashboard")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Predictions",
        "Model Versions",
        "Backtests",
        "Feature Importance"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Symbol", "BTC-USD")
        with col2:
            interval = st.selectbox("Interval", ["1d", "1h", "5m"])
        
        MetricsDashboard.render_prediction_metrics(symbol, interval)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Symbol (Models)", "BTC-USD")
        with col2:
            interval = st.selectbox("Interval (Models)", ["1d", "1h", "5m"])
        
        MetricsDashboard.render_model_versions(symbol, interval)
    
    with tab3:
        MetricsDashboard.render_backtest_results()
    
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Symbol (Features)", "BTC-USD")
        with col2:
            interval = st.selectbox("Interval (Features)", ["1d", "1h", "5m"])
        
        MetricsDashboard.render_feature_importance(symbol, interval)


if __name__ == "__main__":
    render_dashboard()
