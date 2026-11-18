# Algorithmic Trading Features: Chapters 10–15
**Source:** *Trade Stocks and Commodities With the Insiders — Secrets of the COT Report* by Larry Williams (Wiley, 2005)  
**Scope:** Chapters 10 through 15 (Book pp. 99–204 / PDF pp. 115–221)  
**Agent:** `systems-timing`

---

## 1. Chapter 10: "A Unified Theory of COT Data"

### Feature 10.1: Commercials as a Percentage of Total Open Interest (CL/OI & CS/OI)
- **Signal Category:** Macro Regime / Market Concentration Setup
- **Formula & Logic:**
  $$\text{CL\%OI}_t = \frac{\text{Commercial Long Contracts}_t}{\text{Total Open Interest}_t} \times 100$$
  $$\text{CS\%OI}_t = \frac{\text{Commercial Short Contracts}_t}{\text{Total Open Interest}_t} \times 100$$
- **Inputs & Parameters:**
  - Weekly CFTC COT Legacy Report (Commercial Longs, Commercial Shorts, Total Open Interest).
- **Interpretation & Exact Rules:**
  - **The Scale Problem Resolved:** Commercial position size in isolation is misleading. If commercials hold 1,000 contracts when total OI is 5,000, they represent 20% of the market. If commercial longs remain at 1,000 contracts but total OI expands to 20,000, their market share drops to 5%, proving that the open interest expansion was driven entirely by speculative crowds, not commercials.
  - **Bearish Top Setup:** When $\text{CS\%OI}_t \ge 55\%$ (tested up to 60–64% in Japanese Yen), the smart money dominates short hedging across the entire market $\rightarrow$ High-probability major market top and impending decline.
  - **Bullish Bottom Setup:** When $\text{CS\%OI}_t < 30\%$, commercial shorting pressure has evaporated, indicating commercial hedgers have ceased hedging and transitioned to net accumulators $\rightarrow$ Major market bottom setup.

---

### Feature 10.2: WILLCO (Williams Commercial Index)
- **Signal Category:** Core COT Normalization Setup Oscillator
- **Formula & Logic:**
  Stochastic $\%K$ normalization applied to the Commercial Net Position relative to Total Open Interest over lookback window $N$:
  $$\text{RawNetOI}_t = \frac{\text{Commercial Longs}_t - \text{Commercial Shorts}_t}{\text{Total Open Interest}_t}$$
  $$\text{WILLCO}_t = \frac{\text{RawNetOI}_t - \min_{k \in [0, N-1]}(\text{RawNetOI}_{t-k})}{\max_{k \in [0, N-1]}(\text{RawNetOI}_{t-k}) - \min_{k \in [0, N-1]}(\text{RawNetOI}_{t-k})} \times 100$$
  *(Note: Williams notes the indicator can also be constructed directly on Commercial Shorts / OI or Commercial Longs / OI).*
- **Inputs & Parameters:**
  - Lookback Window $N = 26 \text{ weeks}$ (6-month default). Williams establishes that while 3-year (156-week) lookbacks were standard historically, modern fast-paced and volatile markets require a 26-week window across all commodities and financials.
- **Thresholds & Regimes:**
  - **Buy Trigger Zone:** $\text{WILLCO}_t \ge 75\% - 80\%$ (Peak conviction $\ge 90\%$). Indicates commercials control the dominant share of open contracts on the long side.
  - **Sell Trigger Zone:** $\text{WILLCO}_t \le 20\% - 25\%$. Indicates commercials are heavily net short relative to open interest.
  - **Behavioral Role:** "The commercials are trend enders." WILLCO is a setup indicator that identifies when a macro trend has exhausted its fundamental underpinning.

---

### Feature 10.3: ProGo (Professional Go Indicator)
- **Signal Category:** Intraday Smart Money vs. Public Flow / Divergence Indicator
- **Formula & Logic:**
  Williams separates emotional retail trading from informed professional trading by isolating the overnight gap from the intraday session:
  $$\text{Public Daily Flow}_t = \text{Open}_t - \text{Close}_{t-1}$$
  $$\text{Professional Daily Flow}_t = \text{Close}_t - \text{Open}_t$$
  - $\text{Public Flow}$ is positive if $\text{Open}_t > \text{Close}_{t-1}$ (opening gap up driven by overnight news/retail emotion); negative if $\text{Open}_t < \text{Close}_{t-1}$.
  - $\text{Professional Flow}$ is positive if $\text{Close}_t > \text{Open}_t$ (professionals bid the market up into the close, even if the day finishes down net); negative if $\text{Close}_t < \text{Open}_t$.
  - 14-day smoothing:
    $$\text{Public Index}_t = \text{SMA}_{14}(\text{Open}_t - \text{Close}_{t-1})$$
    $$\text{ProGo}_t = \text{SMA}_{14}(\text{Close}_t - \text{Open}_t)$$
- **Inputs & Parameters:**
  - Daily OHLC price bars. Lookback period = 14 trading days.
- **Exact Divergence Rules:**
  - **Bearish Distribution Divergence:** Price rallies to new multi-week highs while ProGo fails to confirm, forming lower highs or sloping downward. Indicates professionals are liquidating/distributing into retail morning buying.
  - **Bullish Accumulation Divergence:** Price declines to new multi-week lows while ProGo trends steadily upward. Indicates professionals are absorbing inventory from retail selling into the close.

---

### Feature 10.4: Unified Multi-Condition Trading System (Williams Master Setup)
- **Signal Category:** Systematic Multi-Timeframe Entry Architecture
- **Rules & Combination Logic:**
  1. **Condition 1 (Macro COT Setup):** $\text{WILLCO} \ge 75 - 80\%$ for Longs, or $\text{WILLCO} \le 20 - 25\%$ for Shorts.
  2. **Condition 2 (Seasonal Alignment):** Market enters historical seasonal window (e.g. Japanese Yen seasonal low due late February / March; Gold summer rally window).
  3. **Condition 3 (Smart Money Flow Confirmation):** Daily ProGo displays clear divergence against price (Price falling + ProGo rising for Longs; Price rising + ProGo falling for Shorts).
  4. **Condition 4 (Execution Trigger):** Price Trendline Violation. Draw a short-term declining trendline across recent daily swing highs (for Longs) or an inclining trendline across recent swing lows (for Shorts). Enter on the first day price breaks across the trendline.

---

### Feature 10.5: Consecutive Down Closes Mean-Reversion Pattern
- **Signal Category:** Short-Term Daily Price Action Pattern
- **Formula & Logic:**
  - Count consecutive daily closing price declines: $\text{Close}_t < \text{Close}_{t-1}$.
  - Buy on Market Open on day $t+1$ following $N$ consecutive down closes; exit on Market Close of day $t+1$ (1-day holding period).
- **Empirical Edge (S&P 500 Daily Bars 1982–2004, Table 10.2):**
  - $N=1$ Down Close: 2,507 trades $\rightarrow$ Loss $-\$165,990$
  - $N=2$ Down Closes: 1,119 trades $\rightarrow$ Loss $-\$46,618$
  - $N=3$ Down Closes: 480 trades $\rightarrow$ Profit $+\$38,563$
  - $N=4$ Down Closes: 182 trades $\rightarrow$ Profit $+\$93,210$
  - $N=5$ Down Closes: 59 trades $\rightarrow$ Profit $+\$56,783$
- **Systematic Rule:** Significant positive expectancy exists only when buying after **$\ge 3$ consecutive down closes**.

---

### Feature 10.6: 6-Month COT Commercial Index as a Short-Term Trading Filter
- **Signal Category:** Directional Regime Filter for Short-Term Systems
- **Formula & Logic:**
  - Compute 6-Month ($N=26 \text{ weeks}$) Stochastic COT Commercials Index.
  - Long Filter Rule: Only execute long trading systems when $\text{COT Index}_{26} > 50\%$.
  - Short Filter Rule: Only execute short trading systems when $\text{COT Index}_{26} < 50\%$.
- **Empirical Backtest Validation (Book pp. 120–121):**
  - **S&P 500:** Buying on daily open with $\text{COT Index} < 50\%$ produced $-\$170$ profit with a staggering $-\$209,755$ drawdown over 846 days. Buying with $\text{COT Index} > 50\%$ produced $+\$76,605$ profit with only $-\$39,145$ drawdown over 371 trades.
  - **U.S. Treasury Bonds:** Buying with $\text{COT Index} < 50\%$ produced $-\$13,985$ loss with $-\$25,733$ drawdown over 633 trades. Buying with $\text{COT Index} > 50\%$ produced $+\$28,418$ profit with $-\$17,695$ drawdown over 606 trades.
- **Rule:** Never buy when commercials are net sellers ($\text{COT Index} < 50\%$); restrict long strategies exclusively to commercial net buying environments ($\text{COT Index} > 50\%$).

---

## 2. Chapter 11: "A New Twist on the Commercials: Using Them for Stocks"

### Feature 11.1: Aggregated Dollar-Weighted Index Commercial Position
- **Signal Category:** Macro Equity Index Reversal Warning
- **Formula & Logic:**
  In equity index futures, institutional hedgers distribute commitments across full-sized and E-mini contracts on the S&P 500, Dow Jones Industrial Average, and Nasdaq 100.
  $$\text{Aggregated Dollar Commercial Position}_t = \sum_{i \in \{\text{ES, SP, NQ, ND, YM, DJ}\}} \left( \text{Net Commercial Contracts}_{i,t} \times \text{Multiplier}_i \times \text{Price}_{i,t} \right)$$
- **Empirical Threshold (Jason Goepfert & Steve Briese Benchmark):**
  - When commercial traders reach $-\$38 \text{ Billion}$ net short across index contracts while non-reportable small specs reach $+\$30 \text{ Billion}$ net long, an extreme historical imbalance exists.
  - **Historical Outcome:** Only two occurrences reached this magnitude (week ended 03/06/2001 and 03/12/2002); led to immediate crashes (S&P dropped 14%, Nasdaq 100 dropped 32% in subsequent weeks).

---

### Feature 11.2: Synthetic Commercial Proxy Indicator (for Equities & Non-Reporting Markets)
- **Signal Category:** Equity Insider Proxy / Real-Time COT Lead Indicator
- **Concept & Foundation:**
  - Individual equities (e.g. MSFT, MO, FAST, SBUX, C, INTC) and non-reporting commodities (Pork Bellies, CRB Index, foreign exchanges) do not report CFTC COT data.
  - Commercial hedgers consistently exhibit a distinct market signature: they "buy on scale down and sell on scale up" (accumulate during price weakness, distribute into price strength).
- **Algorithmic Mechanics:**
  - Captures daily downside price bars that occur in an accumulation structure (e.g., downward price moves where close finishes strong in the upper half of the range, or where intraday volume reflects institutional absorption) versus upward price moves occurring in distribution.
  - Normalizes the accumulated net synthetic flow using a 26-week (6-month) Stochastic $\%K$ oscillator.
- **Thresholds & Rules:**
  - **Buy Setup Zone:** Synthetic Proxy Index $\ge 80\%$.
  - **Sell Setup Zone:** Synthetic Proxy Index $\le 20\%$.
- **Edge:** Previews/leads actual weekly COT releases in real-time, eliminating the 3-day CFTC reporting lag. Recommended strictly for high-quality, liquid equities rather than low-priced speculative penny stocks.

---

## 3. Chapter 12: "Pointers and Thoughts on Trading"

### Feature 12.1: Dollar-Based Protective Stop Loss (The Trapeze Net Principle)
- **Signal Category:** Risk Management / Capital Preservation
- **Core Principle:** Stops do not alter the statistical expectancy of a system and cannot turn a losing strategy into a winning one. Their sole objective is capital survival ("the trapeze net").
- **Exact Placement Rule:**
  - Stops must NEVER be placed based on chart formations, moving averages, or trendlines. Stops protect dollars, not lines.
  - Stop loss must be set strictly at the maximum acceptable dollar risk per contract based on account size:
    $$\text{Stop Price}_{\text{Long}} = \text{Entry Price} - \frac{\text{Max Dollar Risk}}{\text{Contract Point Value}}$$
    $$\text{Stop Price}_{\text{Short}} = \text{Entry Price} + \frac{\text{Max Dollar Risk}}{\text{Contract Point Value}}$$

---

### Feature 12.2: The 17-Day Trend Trailing Stop
- **Signal Category:** Trade Management / Trend Trailing Stop
- **Exact Rules & Conditions:**
  - Once a trade is established in a confirmed macro trend:
    - **Long Trailing Stop:** Set protective stop at the lowest low of the last 17 trading days.
    - **Short Trailing Stop:** Set protective stop at the highest high of the last 17 trading days.
  - **Crucial Filtering Rule:** **Inside Days are strictly EXCLUDED from the 17-day count**. An inside day ($\text{High}_t \le \text{High}_{t-1}$ and $\text{Low}_t \ge \text{Low}_{t-1}$) is ignored, extending the lookback window until 17 true directional trading days have elapsed.

---

### Feature 12.3: The 13-Day Trend Re-entry Rule
- **Signal Category:** Trend Re-entry / Anti-Whipsaw Filter
- **Exact Rules & Conditions:**
  - If a position is stopped out by the 17-day trailing stop, but the broader macro trend remains intact:
    - **Long Re-entry Trigger:** Buy on a breakout above the highest high of the last 13 trading days (excluding inside days).
    - **Short Re-entry Trigger:** Sell short on a breakdown below the lowest low of the last 13 trading days (excluding inside days).

---

### Feature 12.4: Time-Based Trend Exhaustion & Parabolic Windfall Exits
- **Signal Category:** Exit / Profit Taking
- **Rules:**
  1. **Time-Based Exit:** Liquidate trend positions after **10 to 15 weeks** of duration. "Few markets go up or down for more than 15 weeks without an important correction."
  2. **Parabolic Windfall Exit:** Liquidate immediately upon sudden, massive, explosive vertical price surges rather than waiting for trailing stops.

---

### Feature 12.5: Anti-Plunging & Position Sizing Discipline
- **Signal Category:** Money Management / Position Sizing
- **Rule:** Plunging (overleveraging) is the primary driver of trader ruin. Sizing must remain strictly scaled so that stepping up contract size (e.g., from 1 to 2, or 10 to 50) causes zero alteration in execution rules or psychological detachment. "The market does not know or care how many contracts you hold."

---

## 4. Chapter 13: "The One-Minute Commodity Trader"

### Feature 13.1: The 60-Second Weekly Screening Strategy
- **Signal Category:** Systematic Multi-Timeframe Trend-COT Filter
- **Nomenclature Clarification:** Williams uses "The One-Minute Commodity Trader" to denote a rapid **60-second weekly screening process per chart**, NOT 1-minute intraday chart bars.
- **Rules & Logic:**
  1. **Trend Filter:** 52-Week Simple Moving Average of weekly closes.
     - Uptrend: $\text{SMA}_{52}(t) > \text{SMA}_{52}(t-1)$ (or $\text{SMA}_{52}(t) > \text{SMA}_{52}(t-2)$).
     - Downtrend: $\text{SMA}_{52}(t) < \text{SMA}_{52}(t-1)$.
  2. **COT Setup Filter:** 26-Week (6-Month) COT Index.
     - Long Setup: Uptrend Active AND $\text{COT Index}_{26} > 80\%$.
     - Short Setup: Downtrend Active AND $\text{COT Index}_{26} < 20\%$.
  3. **Asymmetric Filter:** Strictly discard all COT buys when the 52-week SMA is declining (e.g. Wheat bear market 1998–2002); strictly discard all COT sells when the 52-week SMA is rising (e.g. T-Bonds bull run 2000–2004).
- **Empirical Backtest Validation (Table 13.1):**
  - **British Pound:** $+\$51,311$ profit, 21 trades, 71% win rate
  - **T-Bonds:** $+\$51,736$ profit, 26 trades, 69% win rate
  - **Swiss Franc:** $+\$19,093$ profit, 16 trades, 56% win rate
  - **S&P 500:** $+\$82,683$ profit, 24 trades, 54% win rate
  - **Gold:** $+\$13,285$ profit, 13 trades, 69% win rate
- **Derivatives Implementation:** Buy long call options during confirmed uptrend setups; buy long put options during confirmed downtrend setups.

---

### Feature 13.2: Gold / U.S. Dollar Spread Valuation Index
- **Signal Category:** Intermarket Valuation Setup
- **Formula & Logic:**
  $$\text{Spread}_t = \text{Gold Weekly Close}_t - \text{U.S. Dollar Index Weekly Close}_t$$
  $$\text{Spread Valuation Ratio}_t = \frac{\text{SMA}_3(\text{Spread}_t) - \text{SMA}_{21}(\text{Spread}_t)}{100}$$
- **Thresholds & Rules:**
  - **Overbought / Short Setup:** When $\text{Spread Valuation Ratio}_t > 30\%$, Gold has reached extreme overvaluation relative to the Dollar $\rightarrow$ Professional capital enters to sell Gold; triggers a major 6-month+ decline setup.
  - **Undervalued / Long Setup:** When the ratio compresses to deeply negative/low levels, Gold is undervalued relative to the Dollar $\rightarrow$ Preconditions for major multi-month rallies.

---

### Feature 13.3: Gold Seasonal Cycle
- **Signal Category:** Seasonal Timing Filter
- **Rules:**
  - **Sowing Window (Buy):** Seasonal accumulation launches in mid-year (July).
  - **Reaping Window (Sell):** Seasonal distribution/peaks occur at year-end (December) into early Q1 (January–February). Williams refutes the claim that gold rallies during recessions or stock crashes; gold rallies only when inflation accelerates.

---

## 5. Chapter 14: "Charts: What They Are, What They Mean"

### Feature 14.1: Systematic Evaluation of Charting Types
- **Signal Category:** Technical Charting Evaluation
- **Analysis:**
  - **OHLC Bar Charts:** Primary foundation for assessing range expansion and bar structures.
  - **Candlestick Charts:** Williams tests the Caginalp & Laurent academic study (which claimed 1% returns over 2-day holds) and proves candlesticks fail to produce robust, mechanically tradable edges on futures.
  - **Market Profile:** 30-minute letter brackets (A, B, C...) plotting horizontal volume-at-price histograms; useful for identifying intraday support/resistance and bracket stops.
  - **Kagi & Renko Charts:** Trend-following without regular time intervals; prone to whipsaws during consolidations.
  - **Moving Average Crossovers:** Standalone price/moving average crossovers are non-viable due to severe whipsaw losses; moving averages must function strictly as trend slope filters.

---

### Feature 14.2: Empirical Refutation of Fibonacci Retracements
- **Signal Category:** Anti-Pattern / Risk Filter
- **Empirical Evidence (Tables 14.1–14.4):**
  - Jeff Parent large-scale study (8,000 data points on S&P 500 stocks with 20% zigzag filter): Retracements do NOT cluster at 38.2%, 50%, or 61.8%. Peak clustering occurred at 87.5%.
  - Williams hand tests on S&P 500:
    - 5% Rallies (1992–1994, Table 14.1): Average retracement was **84%**.
    - 5% Rallies (2002–2003, Table 14.2): Average retracement was **118%**.
    - 2% Rallies (2003, Table 14.3): Average retracement was **75%**.
    - 60-Minute Bars (Dec 2004 – Jan 2005, Table 14.4): Average retracement was **92%**.
  - **Algorithmic Takeaway:** Markets do not reliably reverse at Fibonacci levels. Never place automated limit buy orders at 38.2%, 50%, or 61.8%. Fibonacci levels provide at best weak resistance during dead-cat bounces.

---

### Feature 14.3: Classical Chart Patterns as Systematic Triggers (Track 'n Trade Pro Catalog)
- **Signal Category:** Price Action Breakout Triggers
- **Catalog of Implementable Formations (Figure 14.11):**
  1. **Trendlines:** Declining trendline (connects swing highs; breakout triggers Long); Inclining trendline (connects swing lows; breakdown triggers Short).
  2. **Channels:** Parallel boundaries; breakout of channel boundary triggers directional entry.
  3. **Triangles (Symmetrical, Ascending, Descending) & Pennants:** Converging swing highs and lows. Buy on break up and out; sell on break down and out.
  4. **Bull & Bear Flags:** Small counter-trend consolidations.
     - **Execution Rule:** Place bracket stop orders above/below flag boundaries. **Leave the unfilled opposite order as the initial protective stop loss!**
  5. **1-2-3 (A-B-C) Reversals:**
     - 1-2-3 Top: Sell short on price breaking below Point 2.
     - 1-2-3 Bottom: Buy on price breaking above Point 2.
  6. **Head and Shoulders:** Enter on breakout/breakdown through the Neckline.
  7. **Double & Triple Tops/Bottoms:** Enter on violation of the intervening swing high/low.

---

### Feature 14.4: Clarification on Prompt Terminology
- **Note on "Thrust Bars" & "Volatility Breakouts":** The coordinator brief referenced thrust bars, volatility breakouts, and false breaks. In Williams' bibliography, those specific formulas ($Open + k \times Range$, Thrust bar expansions) originate from *Long-Term Secrets to Short-Term Trading* (1999). In *Secrets of the COT Report* (2005), Williams replaces them with:
  1. 4-bar daily price configurations (Up Range, Down Range, Inside, Outside).
  2. The Consecutive Down Closes mean-reversion pattern.
  3. The ProGo intraday momentum index.
  4. Classical chart breakout trigger rules (Figure 14.11).

---

## 6. Chapter 15: "Putting Theory to Work: Practicing What I Preach"

### Feature 15.1: The Tripartite Execution Architecture (Williams Unified Model)
- **Signal Category:** Master Algorithmic Execution System
- **Three-Tier Systematic Hierarchy:**
  1. **Tier 1 — Open Interest (OI) Liquidity Exhaustion:**
     - **Bullish Exhaustion:** Total Open Interest drops to multi-year/multi-month cyclical lows $\rightarrow$ Public participation has collapsed; commercials quietly absorb available float.
     - **Bearish Exhaustion:** Total Open Interest rises to multi-year/multi-month cyclical highs $\rightarrow$ Retail speculative frenzy; commercials absorb buying with heavy short hedges.
  2. **Tier 2 — Commercial Positioning (COT Index / WILLCO):**
     - **Bullish Setup:** Commercials net long or $\text{WILLCO} \ge 75 - 80\%$.
     - **Bearish Setup:** Commercials net short or $\text{WILLCO} \le 20 - 25\%$.
  3. **Tier 3 — Long-Term Trend Filter (39-Week Moving Average):**
     - Compute 39-Week Simple Moving Average of weekly closes: $\text{SMA}_{39}(\text{Close})$.
     - **Buy Execution Allowed IF:**
       - $\text{SMA}_{39}$ is sloping upwards, OR
       - Weekly Close $> \text{SMA}_{39}$, OR
       - $\text{SMA}_{39}$ is flat and prices are depressed into heavy commercial accumulation.
     - **Sell Execution Allowed IF:**
       - $\text{SMA}_{39}$ is sloping downwards, OR
       - Weekly Close $< \text{SMA}_{39}$.

---

### Feature 15.2: Empirical Market Case Studies
- **Case Study 1: U.S. Treasury Bonds (5-Year Bull Market Study)**
  - Rising 39-week SMA filtered out all premature COT commercial sell setups throughout 2000–2004.
  - Allowed execution only on high-conviction buy setups when low OI coincided with extreme COT buying, capturing sequential bull waves with minimal drawdown.
- **Case Study 2: Coffee Futures (5-Year Bear Market Study)**
  - Downward-sloping 39-week SMA eliminated all counter-trend COT buy signals.
  - Identified 6 picture-perfect short setups where high OI converged with low COT readings. Five of six produced immediate large declines, with the late-1999 setup yielding $>\$34,000$ per contract on a $\$1,500$ margin.
