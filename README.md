# 📈 Universal Trading Predictor (Maximal Edge)

**Universal Trading Predictor** is an AI-powered financial trading analysis and prediction application. It applies Machine Learning, Markov Chains, and Monte Carlo Simulations to any tradable asset to help traders maximize their statistical edge and strictly manage risk.

## 🚀 Features

*   **🤖 AI Prediction Engine:** Uses an XGBoost Classifier trained on technical indicators (RSI, MACD, Bollinger Bands, ATR, SMAs) to predict the probability of the next candle moving up or down.
*   **📊 Monte Carlo Risk Analysis:** Simulates 1,000 possible future price paths based on historical volatility to calculate the 95% Confidence Value at Risk (VaR), helping you determine worst-case scenarios for stop-loss placement.
*   **🔄 Markov Regime Filter:** Employs a Markov chain approach to classify the current market state (Bull Trend, Bear Trend, or Choppy/Range).
*   **💼 Paper Trading Portfolio:** A fully functional mock trading environment that allows you to buy and sell assets, track active positions, and monitor unrealized/realized PnL without risking real money.
*   **📈 Interactive Charts:** Beautiful, interactive Plotly charts showing price action and technical indicators.
*   **⚡ Auto-Refresh Dashboard:** Streamlit fragments enable smooth background data fetching and real-time dashboard updates without reloading the entire page.

## 🛠️ Technology Stack

*   **Frontend UI:** [Streamlit](https://streamlit.io/)
*   **Machine Learning:** [XGBoost](https://xgboost.readthedocs.io/), `scikit-learn`
*   **Data Source:** [yfinance](https://pypi.org/project/yfinance/) (Yahoo Finance API)
*   **Data Processing:** `pandas`, `numpy`
*   **Visualization:** Plotly

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sreekanthap89/algorithmic-trading-ai.git
    cd algorithmic-trading-ai
    ```

2.  **Install dependencies:**
    Ensure you have Python installed. Install the required packages using pip:
    ```bash
    pip install streamlit pandas numpy yfinance xgboost scikit-learn plotly joblib
    ```

## 🏃‍♂️ Usage

Run the Streamlit app locally:

```bash
streamlit run app.py
```

Once the app is running:
1.  Enter the ticker symbol of your choice (e.g., `BTC-USD`, `AAPL`, `GC=F`).
2.  Select the desired timeframe (`1d`, `1h`, `5m`).
3.  Click **Fetch & Analyze Data**.
4.  Review the AI signals, Monte Carlo simulations, and paper trade using the **Paper Trading Portfolio** tab.

## ⚠️ Disclaimer

100% accuracy is impossible in finance. This application is for educational and experimental purposes only. It is not financial advice. Always do your own research and manage your risk carefully before executing real trades.
