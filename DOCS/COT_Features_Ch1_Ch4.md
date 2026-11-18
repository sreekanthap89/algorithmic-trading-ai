# Algorithmic Trading Features: Chapters 1–4
**Source:** *Trade Stocks and Commodities With the Insiders — Secrets of the COT Report* by Larry Williams (John Wiley & Sons, 2005)  
**Extraction Scope:** Chapters 1–4 (Book pp. 1–40 / PDF pp. 13–56)  
**Author:** `cot-core`  
**Target Reviewer:** `@verifier`

---

## 1. Executive Summary & Core Market Principles

### 1.1 The Behavioral DNA of Commercials
1. **Commercial Identity & Mandate:** Commercial traders (hedgers) are producers, processors, merchants, and corporate treasuries (e.g., Cargill, General Mills, JP Morgan Chase). They transact to offset operational price risk and protect profit margins, **not** to speculate for directional trading profit.
2. **Infinite Capacity & Volume Dominance:** Unlike speculators, commercial hedgers are exempt from federal speculative position limits and typically account for **~60% of daily trading volume**. They represent the best-capitalized "smart money" in the financial system.
3. **Daily Participation vs. Astute Timing:** Commercials are compelled by their businesses to transact in the marketplace daily. In contrast, the algorithmic speculator maintains the strategic advantage of waiting in cash until commercial positioning reaches statistically extreme imbalances.
4. **Counter-Trend Scaling Behavior:**
   - In declining markets, commercial users/processors buy all the way down ("dumb as foxes"), methodically reducing their average raw material acquisition cost.
   - In rallying markets, commercial producers scale into short hedges at progressively higher prices to lock in guaranteed corporate profits.
   - Speculators and commercials are almost invariably on opposite sides of contracts.

### 1.2 Core Analytical Axioms
- **The Principle of Asymmetry:** Commercials are structurally net short in most physical commodities (e.g., Wheat, Coffee, Gold, Silver, Crude Oil) because agricultural and extractive producers hedge future harvest/production. Consequently:
  - A small buying push (even approaching or barely exceeding the zero line) represents immense commercial bullishness and can ignite major rallies.
  - A market top requires a massive, deep dip below the zero line (extreme net short positioning) to create severe downward pressure.
- **The Fallacy of Zero-Line Crossovers:** Buying solely because commercials cross above zero or selling because they cross below zero is fundamentally flawed. In several markets (e.g., Silver, Coffee), commercials have remained net short continuously for decades.
- **Extremes Over Crossovers:** Major market inflections are triggered by **relative historical extremes** in commercial positioning, not absolute directional states.
- **Signal Taxonomy — Pure Setup/Regime Zones:** Commercial positioning indicators define the **Setup Zone (Market Condition)**, **not** the precision entry trigger. Larry Williams explicitly stresses: *"The index is not a timing tool for the most part."* Precise trade execution requires technical/price-based timing triggers once the commercial setup zone is active.
- **Dynamic Unwinding Principle:** Commercials frequently unwind extreme positions during the course of a trending move. Algorithmic models must treat extreme readings as an initiation condition/regime state rather than expecting the indicator to remain at extremes throughout the trade.

---

## 2. CFTC Data Ingestion & ETL Specifications

### 2.1 Cadence and Latency
- **Snapshot Date:** Tuesday close of business.
- **Release Date:** Friday at 3:30 p.m. Eastern Time (3-calendar-day / 3-business-day lag).
- **Point-in-Time Integrity:** Data released on Friday at 3:30 p.m. ET can only be applied to trades executed on the Friday closing auction (or post-market) or Monday market open. Backtesters **must not** apply COT readings to Wednesday–Friday intraday bars prior to 3:30 p.m. ET Friday.

### 2.2 Reporting Formats & Accounting Balance
- **Report Types:**
  - **Futures-Only:** Raw futures contract open interest.
  - **Futures-and-Options-Combined:** Aggregates futures contracts with options positions converted to futures-equivalent contracts using exchange delta factors ($\Delta$):
    $$\text{Combined Long} = \text{Futures Long} + (\text{Call Long} \times \Delta_{\text{call}}) + (\text{Put Short} \times |\Delta_{\text{put}}|)$$
    $$\text{Combined Short} = \text{Futures Short} + (\text{Call Short} \times |\Delta_{\text{call}}|) + (\text{Put Long} \times |\Delta_{\text{put}}|)$$
- **Accounting Identities (Conservation of Open Interest):**
  $$\text{Total Open Interest } (OI) = \text{Total Gross Long} = \text{Total Gross Short}$$
  $$\text{Reportable Long} = \text{Commercial Long } (CL) + \text{Non-Commercial Long } (NCL) + \text{Spreading } (SP)$$
  $$\text{Reportable Short} = \text{Commercial Short } (CS) + \text{Non-Commercial Short } (NCS) + \text{Spreading } (SP)$$
  $$\text{Nonreportable (Small Trader) Long } (NRL) = OI - \text{Reportable Long}$$
  $$\text{Nonreportable (Small Trader) Short } (NRS) = OI - \text{Reportable Short}$$

---

## 3. Catalog of Implementable Features

### Feature 1: Raw Commercial Net Position ($CP$)
- **Name:** Commercial Net Position
- **Mathematical Formula:**
  $$CP_t = CL_t - CS_t$$
  Where:
  - $CL_t$ = Total Commercial Long contracts held at report week $t$.
  - $CS_t$ = Total Commercial Short contracts held at report week $t$.
- **Inputs & Parameters:**
  - Data: Weekly CFTC COT Report (Futures-Only or Combined).
  - Parameters: None.
- **Signal Type:** Directional Setup / Baseline Metric.
- **Interpretation & Thresholds:**
  - $CP_t > 0$: Commercials are Net Long (holding more long than short contracts).
  - $CP_t = 0$: Neutral Balance Line (Zero Line).
  - $CP_t < 0$: Commercials are Net Short (holding more short than long contracts).
  - *Contextual Caveat:* In commodities with structural short biases, absolute negative values are normal; raw crossovers of zero are rare and must be evaluated against multi-year history.

---

### Feature 2: Commercial Net Position as % of Total Open Interest ($CP\%OI$)
- **Name:** Commercial Net Open Interest Percentage
- **Mathematical Formula:**
  $$CP\%OI_t = \frac{CL_t - CS_t}{OI_t} \times 100 = \frac{CP_t}{OI_t} \times 100$$
- **Inputs & Parameters:**
  - Data: $CL_t$, $CS_t$, Total Open Interest ($OI_t$).
  - Parameters: None.
- **Signal Type:** Normalized Macro Regime Setup.
- **Interpretation & Thresholds:**
  - Normalizes commercial position size across different markets and across decades of secular open interest growth.
  - Allows cross-commodity comparison (e.g., comparing commercial commitment in E-mini S&P vs Wheat vs Gold).

---

### Feature 3: Commercial Gross Participation Ratios ($CL\%OI$ & $CS\%OI$)
- **Name:** Commercial Gross Long/Short Share of Open Interest
- **Mathematical Formulas:**
  $$CL\%OI_t = \frac{CL_t}{OI_t} \times 100$$
  $$CS\%OI_t = \frac{CS_t}{OI_t} \times 100$$
- **Inputs & Parameters:**
  - Data: $CL_t$, $CS_t$, $OI_t$.
- **Signal Type:** Market Structure / Concentration Regime.
- **Interpretation & Thresholds:**
  - Distinguishes whether changes in Net Position are driven by aggressive new accumulation ($CL\%OI$ rising) or passive short liquidation ($CS\%OI$ falling).
  - Dominance Threshold: When commercial participation exceeds 50% on either side (e.g., $CL\%OI > 50\%$ as cited in Wheat Ch 2), commercials dictate market liquidity.

---

### Feature 4: Commercial Net Weekly Flow / Velocity ($\Delta CP$ and $\Delta CP\%OI$)
- **Name:** Commercial Net Flow (Weekly Velocity)
- **Mathematical Formulas:**
  $$\Delta CP_t = CP_t - CP_{t-1} = (CL_t - CL_{t-1}) - (CS_t - CS_{t-1})$$
  $$\Delta CP\%OI_t = \frac{\Delta CP_t}{OI_t} \times 100$$
- **Inputs & Parameters:**
  - Data: Current and previous week commercial long and short positions.
- **Signal Type:** Early Momentum / Flow Setup.
- **Interpretation & Thresholds:**
  - $\Delta CP_t > 0$: Commercials engaged in net buying during the reporting week.
  - $\Delta CP_t < 0$: Commercials engaged in net selling during the reporting week.
  - Decoupling Detection: Commercials may remain Net Long ($CP_t > 0$) while liquidating longs on a weekly basis ($\Delta CP_t < 0$). This divergence flags momentum exhaustion in an ongoing rally.

---

### Feature 5: Multi-Year Extreme Rule ("Lesson One in Using Commercial Data")
- **Name:** Commercial Multi-Year Net Position Extremes
- **Mathematical Logic:**
  $$\text{IsMultiYearHigh}(t, N) = \begin{cases} 1 & \text{if } CP_t \ge \max_{1 \le k \le N}(CP_{t-k}) \\ 0 & \text{otherwise} \end{cases}$$
  $$\text{IsMultiYearLow}(t, N) = \begin{cases} 1 & \text{if } CP_t \le \min_{1 \le k \le N}(CP_{t-k}) \\ 0 & \text{otherwise} \end{cases}$$
- **Inputs & Parameters:**
  - Standard Lookback ($N$): **4 Years (208 weeks)** — explicitly stated as Larry Williams' standard *"rule of thumb to demand from our monitoring of the commercials' flow of money"*.
  - Extended Multi-Year Windows Documented in Text:
    - $N = 156$ weeks (3 Years)
    - $N = 260$ weeks (5 Years — cited in S&P 500 1998/1999 buy setups)
    - $N = 312$ weeks (6 Years — cited in Gold 2004 top)
    - $N = 520$ weeks (10 Years — cited in British Pound 2000 short setup)
    - $N = 728$ weeks (14 Years — cited in British Pound 1999 bull setup)
    - All-Time / 18-Year Record (cited in S&P 500 early 2000 historic short peak before multi-year crash)
- **Signal Type:** Major Cyclical / Generational Macro Regime Setup.
- **Interpretation & Rules:**
  - **Bull Market Rule:** Multi-year highs in net commercial buying beget bull markets. A 4-year high in $CP$ activates a high-priority macro Bullish Setup Zone.
  - **Bear Market Rule:** Multi-year lows in net commercial selling beget bear markets. A 4-year low in $CP$ activates a high-priority macro Bearish Setup Zone.
  - **Decay Rule:** The extreme reading need not be maintained; commercials work off positions as the trend matures.

---

### Feature 6: The Williams COT Index ($COT\_Index$)
- **Name:** Williams COT Index (Stochastic %K Oscillator of Commercial Net Position)
- **Mathematical Formula:**
  $$COT\_Index_t = \left( \frac{CP_t - \min_{0 \le k < N}(CP_{t-k})}{\max_{0 \le k < N}(CP_{t-k}) - \min_{0 \le k < N}(CP_{t-k})} \right) \times 100$$
  Where:
  - $CP_t = CL_t - CS_t$ (Commercial Net Position at report week $t$).
  - $\min_{0 \le k < N}(CP_{t-k})$ = Lowest commercial net position over the rolling lookback window $N$.
  - $\max_{0 \le k < N}(CP_{t-k})$ = Highest commercial net position over the rolling lookback window $N$.
  - Output is bounded: $0.0 \le COT\_Index_t \le 100.0$.
- **Inputs & Parameters:**
  - Lookback Window ($N$):
    - **Primary Standard (Williams Baseline):** $N = 156 \text{ weeks}$ (3 Years) — standard industry benchmark established in Chapter 4.
    - **Cyclical Annual Intermediate:** $N = 52 \text{ weeks}$ (1 Year).
    - **Responsive Swing Lookback:** $N = 26 \text{ weeks}$ (6 Months).
- **Signal Type:** Primary Systematic Setup Filter (Directional Bias Zone).
- **Thresholds & Quantitative Logic:**
  - **Bullish Setup Zone:** $COT\_Index_t \ge 80\%$ (Alternative sensitivity threshold: $\ge 75\%$; Peak extreme: $100\%$).
    - *Interpretation:* Commercials are accumulating contracts at a rate near the upper extreme of their rolling multi-year range. Market conditions favor long positions; look for upside technical triggers.
  - **Bearish Setup Zone:** $COT\_Index_t \le 20\%$ (Alternative sensitivity threshold: $\le 25\%$; Trough extreme: $0\%$).
    - *Interpretation:* Commercials are heavily distributing or establishing heavy short hedges near the lower extreme of their rolling range. Market conditions favor short positions; look for downside technical triggers.
  - **Neutral / No-Man's-Land:** $20\% < COT\_Index_t < 80\%$. No new setup is generated.
- **Critical Relative Value Nuance:**
  - A market where commercials are net short in absolute terms can simultaneously register a $COT\_Index \ge 80\%$ (e.g., Gold rising to 69%–100% while net short; Crude Oil hitting 100% while net short). Relative positioning dominates absolute sign.

---

### Feature 7: "The Silver Rule" (Structural Net-Short Regime Breakout)
- **Name:** Perpetual Short Regime Breakout
- **Mathematical Logic:**
  $$\text{RegimeBreakout}_{Long, t} = \mathbb{I}\left( CP_t > 0 \;\land\; \min_{1 \le k \le M}(CP_{t-k}) < 0 \;\land\; \max_{1 \le k \le M}(CP_{t-k}) \le 0 \right)$$
- **Inputs & Parameters:**
  - Asset Universe: Markets characterized by permanent commercial short producer hedging (e.g., Silver, Coffee, Cocoa).
  - Lookback ($M$): Historical span of structural net shorting ($M \ge 260 \text{ weeks}$ / 5+ years, or up to 22 years for Silver).
- **Signal Type:** Rare Secular Mega-Bull Regime Trigger.
- **Interpretation:**
  - *"If you ever see the commercials go net long silver, look for a major bull market."* If commercial hedgers abandon their structural short bias and become net long buyers, physical demand has fundamentally overwhelmed producer supply.

---

### Feature 8: Bill Jiler CRB Deviation Model (Historical Precursor)
- **Name:** Jiler Commercial Net Seasonal Deviation
- **Mathematical Formula:**
  $$\text{Seasonal\_Norm}_m(CP) = \frac{1}{Y} \sum_{y=1}^{Y} CP_{m, y}$$
  $$\text{Dev}_t = CP\%OI_t - \text{Seasonal\_Norm}_{\text{month}(t)}(CP\%OI)$$
- **Inputs & Parameters:**
  - Lookback ($Y$): 6 to 10 years of month-end or weekly seasonal historical data.
  - Parameter Noise Filter: $5\%$
  - Parameter Exhaustion Filter: $40\%$
- **Signal Type:** Secondary Structural Valuation Setup.
- **Interpretation & Rules:**
  - **Noise Filter:** Disregard deviations where $|\text{Dev}_t| < 5\%$.
  - **Bullish Configuration:** Large Hedgers heavily net long above seasonal norm ($Dev > +20\%$), Large Speculators net long, Small Traders net short below seasonal norm.
  - **Bearish Configuration:** Large Hedgers heavily net short below seasonal norm ($Dev < -20\%$), Small Traders heavily net long.
  - **Caution / Climax Flag:** Be extremely wary when deviations exceed $40\%$ from the long-term norm ($|\text{Dev}_t| > 40\%$).

---

### Feature 9: Rapid Commercial Rotation Metric (26-Week Swing Velocity)
- **Name:** 26-Week COT Swing Velocity
- **Mathematical Formula:**
  $$\text{SwingVelocity}_{26, t} = COT\_Index_t(N=156) - COT\_Index_{t-26}(N=156)$$
  $$\text{NetContractSwing}_{26, t} = CP_t - CP_{t-26}$$
- **Inputs & Parameters:**
  - Time Horizon: 26 weeks (~6 months).
- **Signal Type:** Regime Shift / Trend Reversal Setup.
- **Interpretation & Thresholds:**
  - Documented in the British Pound case study: commercials swung from a 14-year high in net buying ($COT\_Index \approx 100\%$) to a 10-year low in net selling ($COT\_Index \approx 0\%$) in 26 weeks.
  - Rapid shift ($|\text{SwingVelocity}_{26}| \ge 75\%$) signals an aggressive institutional about-face; invalidates existing trend setups and initiates reverse setup.

---

## 4. Feature Summary & Execution Matrix

| Feature Code | Name | Mathematical Core | Standard Parameters | Signal Category | Bullish Threshold | Bearish Threshold |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **COT-01** | Commercial Net Position | $CL - CS$ | None | Setup / Level | $CP > 0$ (contextual) | $CP \ll 0$ |
| **COT-02** | Commercial Net % of OI | $\frac{CL - CS}{OI} \times 100$ | None | Setup / Normalized | Asset-relative | Asset-relative |
| **COT-03** | Commercial Gross Long/Short % | $\frac{CL}{OI} \times 100$, $\frac{CS}{OI} \times 100$ | None | Market Concentration | $CL\%OI > 50\%$ | $CS\%OI > 50\%$ |
| **COT-04** | Commercial Weekly Net Flow | $\Delta CP = CP_t - CP_{t-1}$ | 1 week | Flow / Velocity | $\Delta CP > 0$ | $\Delta CP < 0$ |
| **COT-05** | Multi-Year Net Extremes | $CP_t \gtrless \text{Extremes}(N)$ | $N = 208\text{ w (4 yr)}$ | Macro Regime Setup | 4-Year New High | 4-Year New Low |
| **COT-06** | Williams COT Index | $\frac{CP - \min(CP, N)}{\max(CP, N) - \min(CP, N)} \times 100$ | $N = 156\text{ w (3 yr)}$ (alt: 52w, 26w) | Tactical Setup Filter | $COT\_Index \ge 80\%$ (or $75\%$) | $COT\_Index \le 20\%$ (or $25\%$) |
| **COT-07** | Perpetual Short Regime Break | $CP_t > 0 \land CP_{t-k \le M} \le 0$ | $M = 260\text{w to } 1000\text{w}$ | Secular Regime Trigger | Flip to Net Long in Structural Short Market | N/A |
| **COT-08** | Jiler Seasonal Deviation | $CP\%OI - \text{SeasonalNorm}$ | 6–10 year seasonal norm | Structural Valuation Setup | $Dev > +20\%$ | $Dev < -20\%$ (Disregard $<5\%$, Caution $>40\%$) |
| **COT-09** | 26-Week Swing Velocity | $Index_t - Index_{t-26}$ | 26 weeks | Regime Reversal Setup | $\Delta_{26} \ge +75\%$ | $\Delta_{26} \le -75\%$ |

---

## 5. Architectural & Implementation Nuances for Verifier & Engine Authors

1. **Pipeline Execution Guard (Look-Ahead Bias Prevention):**
   - The COT report is released every Friday at 15:30 Eastern Time for Tuesday positions.
   - For a daily backtesting engine: on day $D$, the COT state active for decision-making is the report released on or prior to $D$.
   - Any trade execution based on a Friday report must occur at Friday Close / Post-Market (if processing is sub-minute) or Monday Market Open.
2. **Dynamic Unwinding vs Trade Lifespan:**
   - A common algorithmic failure is enforcing $COT\_Index \ge 80\%$ as a continuous holding condition. Williams proves that commercials sell into the rally they ignited. Once the setup initiates a position (via a price trigger), holding and exits must be governed by trend/trailing stops, not by waiting for COT to stay high.
3. **Data Warm-Up Lengths:**
   - For $N = 156$ weeks (3-year COT Index), minimum warm-up requires 3 full calendar years of weekly COT history before the first valid indicator bar.
   - For $N = 208$ weeks (4-year Multi-Year Extreme), minimum warm-up requires 4 full calendar years of weekly data.
4. **Denominator Zero Protection:**
   - In rare synthetic or newly listed contracts where $\max(CP, N) == \min(CP, N)$, the COT Index formula must include an epsilon guard:
     $$\text{Denominator} = \max(\max(CP, N) - \min(CP, N), \epsilon)$$
