# Trading App - Complete Usage Guide

## Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app_refactored.py
```

### 2. Basic Usage
1. Enter symbol (BTC-USD, AAPL, GC=F)
2. Select timeframe (1d, 1h, 5m)
3. Click "Refresh & Analyze Data"
4. View dashboard with AI signals
5. Optional: Run backtest, analyze features

---

## Advanced Features

### Model Versioning

**Automatic Version Tracking:**
Every time a model is trained, it's automatically versioned and saved.

```python
from ml.model_versioning import ModelRegistry

registry = ModelRegistry()
versions = registry.list_versions("BTC-USD", "1d")
# Output: [
#   {
#     "version_id": "v20240101_120000",
#     "timestamp": "2024-01-01T12:00:00",
#     "status": "active",
#     "metrics": {"accuracy": 0.85, ...}
#   }
# ]

# Get specific version
info = registry.get_model_info("BTC-USD", "1d", "v20240101_120000")
```

**Load Previous Model:**
```python
from ml.model_versioning import ModelStorage

storage = ModelStorage()
model_data = storage.load_model("BTC-USD", "1d", "v20240101_120000")
```

**Compare Versions:**
```python
comparison = storage.compare_versions(
    "BTC-USD", "1d",
    ["v20240101_120000", "v20240102_120000"]
)
# Compare metrics across versions
print(comparison)
```

---

### Model Validation & Metrics

**Track Predictions:**
```python
from metrics.validation import PredictionTracker

tracker = PredictionTracker("BTC-USD", "1d")

# Log each prediction
tracker.log_prediction(
    signal="BUY",
    prob_up=0.72,
    actual_result=True,  # Set after price moves
    metadata={"sentiment": "bullish"}
)

# Check for model drift
drift_metrics = tracker.calculate_drift(window=100)
if drift_metrics["drift_detected"]:
    print("⚠️ Model drift detected!")
    print(f"Drift magnitude: {drift_metrics['drift_magnitude']:.4f}")

# Load historical predictions
history_df = tracker.load_historical_predictions()
print(f"Total predictions: {len(history_df)}")
```

**Validate Model:**
```python
from metrics.validation import ModelValidator

metrics = ModelValidator.validate_model(y_true, y_pred, y_pred_proba)
# Returns:
# {
#   "accuracy": 0.82,
#   "precision": 0.80,
#   "recall": 0.84,
#   "f1_score": 0.82,
#   "roc_auc": 0.88,
#   "confusion_matrix": {
#     "true_negatives": 450,
#     "false_positives": 50,
#     "false_negatives": 100,
#     "true_positives": 400
#   },
#   "specificity": 0.90,
#   "sensitivity": 0.80
# }
```

---

### Feature Importance Analysis

**Compute Importance:**
```python
from metrics.feature_importance import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer("BTC-USD", "1d")

# Run comprehensive analysis
analysis = analyzer.analyze_all_methods(X, y, model_dict, meta_stacker)
# Analyzes using:
# - Tree-based importance (XGBoost, RandomForest)
# - Permutation importance
# - Correlation importance
```

**Get Top Features:**
```python
# Get top 10 by tree importance
top_features = analyzer.get_top_features(
    method="tree_importance",
    top_n=10
)
# Returns: [("RSI", 0.15), ("MACD", 0.12), ...]

# Get by permutation importance
top_perm = analyzer.get_top_features(
    method="permutation_importance",
    top_n=10
)
```

**Generate Report:**
```python
report = analyzer.generate_importance_report()
print(report)
# Output:
# ============================================================
# Feature Importance Report: BTC-USD (1d)
# Timestamp: 2024-01-02T15:30:00
# ============================================================
#
# TREE_IMPORTANCE
# ----------------------------------------
#  1. RSI                               0.1523
#  2. MACD                              0.1247
#  ...
```

---

### Backtesting

**Run Single Backtest:**
```python
from backtesting.backtest_engine import Backtest

backtest = Backtest(
    symbol="BTC-USD",
    interval="1d",
    initial_capital=100000,
    commission=0.001,
    slippage=0.0005
)

results = backtest.run(
    df,
    signal_col='prob_up',
    entry_threshold=0.56,
    exit_threshold=0.44,
    use_stops=True,
    stop_loss_pct=0.05
)

print(backtest.get_summary())
# Output:
# ============================================================
# BACKTEST SUMMARY: BTC-USD (1d)
# ============================================================
#
# CAPITAL:
#   Initial Capital:      $100,000.00
#   Final Capital:        $120,500.00
#   Net Return:           +20.50%
#
# TRADING STATISTICS:
#   Total Trades:         42
#   Winning Trades:       28
#   Losing Trades:        14
#   Win Rate:             66.67%
#   Average Win:          $850.50
#   Average Loss:         -$425.00
#   Profit Factor:        1.89
#
# RISK METRICS:
#   Max Drawdown:         -12.50%
#   Sharpe Ratio:         1.45
#   Sortino Ratio:        2.13

# Save results
filepath = backtest.save_results("my_backtest_20240101.json")
```

**Parameter Optimization:**
```python
from backtesting.optimizer import ParameterOptimizer

optimizer = ParameterOptimizer("BTC-USD", "1d")

best_result = optimizer.grid_search(
    df,
    signal_col='prob_up',
    entry_thresholds=[0.54, 0.55, 0.56, 0.57, 0.58],
    exit_thresholds=[0.42, 0.43, 0.44, 0.45, 0.46],
    stop_loss_pcts=[0.03, 0.05, 0.07],
    metric="sharpe_ratio"
)

print(f"Best parameters: {best_result['params']}")
print(f"Sharpe ratio: {best_result['metric_value']:.2f}")

# Get all results
results_df = optimizer.get_results_dataframe()
results_df.to_csv("optimization_results.csv")
```

**Analyze Trade Performance:**
```python
from metrics.validation import BacktestMetrics

trades = results["trades"]

returns_metrics = BacktestMetrics.calculate_returns(trades)
# {
#   "total_return": 20500,
#   "win_rate": 0.667,
#   "avg_win": 850.5,
#   "avg_loss": -425,
#   "profit_factor": 1.89,
#   ...
# }

equity_curve = results["equity_curve"]
risk_metrics = BacktestMetrics.calculate_risk_metrics(equity_curve)
# {
#   "max_drawdown": -0.125,
#   "sharpe_ratio": 1.45,
#   "sortino_ratio": 2.13,
#   "total_return": 0.205
# }
```

---

### Logging

**View Logs:**
```bash
# Recent activity
tail -f logs/trading_app.log

# Errors only
tail -f logs/errors.log

# Search for symbol
grep "BTC-USD" logs/trading_app.log

# With timestamp
grep "2024-01-02" logs/trading_app.log | head -20
```

**Custom Logging:**
```python
from utils.logging_setup import get_logger

logger = get_logger(__name__)

logger.debug("Debugging info")
logger.info(f"Processed {count} data points")
logger.warning("Insufficient data for analysis")
logger.error(f"Failed to fetch data: {error}")
```

---

### Configuration

**Modify Settings:**
Edit `config/settings.py`:

```python
# Change model parameters
MODEL_CONFIG["base_models"]["XGBoost"]["max_depth"] = 5

# Change ensemble weights
ENSEMBLE_WEIGHTS["ml_stacker"] = 0.50  # Increased from 0.45
ENSEMBLE_WEIGHTS["quantum_engine"] = 0.20  # Decreased from 0.25

# Change signal thresholds
SIGNAL_THRESHOLDS["buy_threshold"] = 0.60
SIGNAL_THRESHOLDS["sell_threshold"] = 0.40

# Change backtesting parameters
BACKTEST_CONFIG["commission"] = 0.002  # 0.2%
BACKTEST_CONFIG["slippage"] = 0.001  # 0.1%
```

---

### Real-time Metrics Dashboard

**Launch Dashboard:**
```bash
streamlit run pages/metrics_dashboard.py
```

**Features:**
1. **Predictions Tab**: View prediction history, drift analysis, probability distribution
2. **Model Versions Tab**: Compare model versions, view metrics
3. **Backtests Tab**: Browse and analyze backtest results
4. **Feature Importance Tab**: Visualize feature rankings

---

## Workflow Examples

### Example 1: Daily Strategy Monitoring

```python
from data_engine import fetch_data
from features import add_features
from models import TradingModel, get_markov_regime
from metrics.validation import PredictionTracker

def daily_monitoring():
    symbol = "BTC-USD"
    interval = "1d"
    
    # Fetch & analyze
    df = fetch_data(symbol, period="2y", interval=interval)
    df_feat = add_features(df)
    
    model = TradingModel(symbol, interval)
    model.train_or_load(df_feat, force_retrain=False)
    prediction = model.predict_next(df_feat)
    regime = get_markov_regime(df_feat)
    
    # Log prediction
    tracker = PredictionTracker(symbol, interval)
    tracker.log_prediction(
        signal=prediction['signal'],
        prob_up=prediction['prob_up']
    )
    
    # Check drift
    drift = tracker.calculate_drift(window=100)
    if drift["drift_detected"]:
        print("⚠️ Model needs retraining!")
    
    return prediction, regime

daily_monitoring()
```

### Example 2: Strategy Optimization

```python
from data_engine import fetch_data
from features import add_features
from backtesting.optimizer import ParameterOptimizer

def optimize_strategy():
    symbol = "BTC-USD"
    interval = "1h"
    
    # Prepare data
    df = fetch_data(symbol, period="1mo", interval=interval)
    df_feat = add_features(df)
    df_feat['prob_up'] = 0.55  # Your signal column
    
    # Optimize
    optimizer = ParameterOptimizer(symbol, interval)
    best = optimizer.grid_search(
        df_feat,
        signal_col='prob_up',
        entry_thresholds=np.arange(0.50, 0.65, 0.02),
        exit_thresholds=np.arange(0.35, 0.50, 0.02),
        stop_loss_pcts=[0.03, 0.05, 0.07, 0.10],
        metric="sharpe_ratio"
    )
    
    print(f"Best: {best['params']}")
    
    # Save results
    optimizer.save_results(f"optimization_{symbol}_{interval}.csv")

optimize_strategy()
```

### Example 3: Feature Analysis

```python
from data_engine import fetch_data
from features import add_features
from models import TradingModel
from metrics.feature_importance import FeatureImportanceAnalyzer

def analyze_features():
    symbol = "BTC-USD"
    interval = "1d"
    
    # Train model
    df = fetch_data(symbol, period="2y", interval=interval)
    df_feat = add_features(df)
    
    model = TradingModel(symbol, interval)
    model.train_or_load(df_feat, force_retrain=True)
    
    # Analyze importance
    analyzer = FeatureImportanceAnalyzer(symbol, interval)
    analysis = analyzer.analyze_all_methods(
        df_feat[model.feature_cols],
        df_feat['Target'],
        model.models,
        model.meta_stacker
    )
    
    # Get report
    report = analyzer.generate_importance_report()
    print(report)
    
    # Top features
    top = analyzer.get_top_features("tree_importance", 15)
    for feat, score in top:
        print(f"{feat}: {score:.4f}")

analyze_features()
```

---

## Troubleshooting

### Issue: "ImportError: No module named 'config'"
**Solution:**
```bash
# Ensure you're in the TRADING_APP directory
cd d:\DEV\PY\TRADING_APP

# Run with python path set
python -m streamlit run app_refactored.py
```

### Issue: "ValueError: not enough values to unpack"
**Solution:** Check that all required data columns are present:
```python
required = ['Open', 'High', 'Low', 'Close', 'Volume']
assert all(col in df.columns for col in required)
```

### Issue: "FileNotFoundError: logs directory"
**Solution:** Logging directory is auto-created. If issue persists:
```bash
mkdir logs
mkdir models
mkdir metrics
mkdir backtest_results
```

### Issue: "Model training fails with 'not enough data'"
**Solution:** Ensure you have at least 50 rows of data:
```python
assert len(df) >= 50, f"Need 50+ rows, got {len(df)}"
```

---

## Performance Tips

1. **Cache Data**: Use 1d interval for faster loading
2. **Batch Processing**: Process multiple symbols
3. **Limit Backtests**: Use walk-forward analysis instead of full optimizations
4. **Archive Models**: Regularly archive old versions
5. **Monitor Logs**: Check for warnings and errors

---

## Support

For issues or questions:
1. Check logs in `logs/trading_app.log`
2. Review IMPROVEMENTS.md
3. Verify requirements.txt matches installed packages
4. Test with sample data first

