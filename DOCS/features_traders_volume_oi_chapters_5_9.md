# Algorithmic Trading Features: Chapters 5–9 (and Chapter 10 Unified Extensions)
**Source**: *Trade Stocks and Commodities With the Insiders - Secrets of the COT Report* by Larry Williams (Wiley, 2005)  
**Assigned Analyst**: `traders-volume-oi`  
**Scope**: Chapters 5–9 (PDF pp. 53–114) and related Unified Theory features in Chapter 10 (PDF pp. 115–128)

---

## Executive Summary of Feature Architecture

Larry Williams organizes market participants and data flows into a clear hierarchy:
1. **Commercials (Hedgers / Insiders)**: Smart money, informed producers/consumers, access to cash markets, scale buyers on weakness and scale sellers on strength.
2. **Non-Commercials (Large Traders / Funds)**: Systematic trend-following funds, CTAs, managed futures, floor traders. They pyramid into trends via technical breakout channels (specifically 20-day, 40-day, 26-week, and 52-week channels). At trend climaxes, they hold maximum positioning ("car out of gas").
3. **Non-Reportables (Small Traders / Public)**: Retail crowd, emotionally driven, consensus-seeking, buying tops and selling bottoms.
4. **Volume**: Measure of aggregate turnover / transaction friction. Empirical tests reveal that conventional technical analysis ("price up, volume up is bullish") fails, whereas **volume surges on price declines** represent institutional accumulation / selling climaxes and are consistently bullish.
5. **Open Interest (OI)**: Total outstanding speculative interest ("swimming pool participants"). Total OI peaks at major tops and troughs at major bottoms ("pure commercial market").
6. **Inside OI & Volume Integration**: The crucial question is not whether OI or volume is rising, but **who is driving the change** (commercials absorbing vs speculators chasing).

Below is the complete catalog of implementable features, formulas, parameters, signal categories, and trading rules.

---

## Section 1: Non-Commercials (Large Traders / Funds) & Non-Reportables (Small Traders) Features

### Feature 1.1: Small Trader Net Position (Net Small Spec)
- **Name**: `Net_Small_Spec`
- **Mathematical Formula**:
  $$\text{Net\_Small\_Spec}_t = \text{NonReportable\_Long}_t - \text{NonReportable\_Short}_t$$
  Where data is extracted from the weekly CFTC Commitments of Traders report (Non-Reportable / Other positions).
- **Inputs & Parameters**:
  - `NonReportable_Long`, `NonReportable_Short` (Weekly frequency).
- **Signal Type**: Baseline Regime Metric / Raw Level.
- **Interpretation & Thresholds**:
  - Absolute positioning is **not** a direct standalone signal because retail traders can remain net long or net short for multi-year secular cycles (e.g., net long cotton throughout a 6.5-year bear market 1998–2004).
  - Used as the fundamental input to normalize relative sentiment and contrarian extremes.

---

### Feature 1.2: Small Trader COT Index (Public Contrary Sentiment Index)
- **Name**: `COT_Index_Small` (or `Small_Spec_Index`)
- **Mathematical Formula**:
  $$\text{COT\_Index\_Small}_t = 100 \times \frac{\text{Net\_Small\_Spec}_t - \min_{k=0}^{N-1}(\text{Net\_Small\_Spec}_{t-k})}{\max_{k=0}^{N-1}(\text{Net\_Small\_Spec}_{t-k}) - \min_{k=0}^{N-1}(\text{Net\_Small\_Spec}_{t-k})}$$
- **Inputs & Parameters**:
  - Lookback window $N$:
    - Primary legacy lookback: $N = 156$ weeks (3-year window).
    - Intermediate lookback: $N = 52$ weeks (1 year) or $N = 26$ weeks (6 months).
- **Signal Type**: Setup (Contrarian Sentiment Extreme Setup).
- **Interpretation & Thresholds**:
  - **Rule 1 (Sell Setup)**: `COT_Index_Small >= 75%` (or `>= 80%`).
    - *Interpretation*: The public has reached an extreme bullish consensus. Retail traders have fully committed their buying power; "there is no one left to drive prices higher." Look to **SELL SHORT**.
  - **Rule 2 (Buy Setup)**: `COT_Index_Small <= 25%` (or `<= 20%`).
    - *Interpretation*: The public has capitulated into extreme bearishness / net selling. Look to **BUY LONG**.
  - **Market Asymmetry Nuance**:
    - High public bullishness (`> 75%`) produces reliable selloffs across virtually all markets (Lumber, Hogs, Bonds, Gold, Cotton).
    - Low public bullishness (`< 25%`) does not automatically trigger an advance unless **Commercials** are concurrently buying (e.g., in persistent secular bear markets like Lumber). In retail-heavy markets with low margins (e.g., Live Hogs, Treasury Bonds), low public readings yield dramatic rallies.

---

### Feature 1.3: Large Trader Net Position (Net Large Spec)
- **Name**: `Net_Large_Spec`
- **Mathematical Formula**:
  $$\text{Net\_Large\_Spec}_t = \text{NonCommercial\_Long}_t - \text{NonCommercial\_Short}_t$$
- **Inputs & Parameters**:
  - `NonCommercial_Long`, `NonCommercial_Short` from weekly CFTC report.
- **Signal Type**: Baseline Regime Metric / Trend Tracking.
- **Interpretation & Logic**:
  - Large traders (commodity funds, CTAs) are technical trend followers.
  - As an uptrend matures, funds systematically add contracts via sequential breakout systems (20-day, 40-day, 26-week, 52-week).
  - Consequently, at the very peak of an advance, large traders hold their largest net long positions; at the trough of a decline, they hold their largest net short positions.

---

### Feature 1.4: Large Trader COT Index (Trend-Follower Climax / Exhaustion Index)
- **Name**: `COT_Index_Large` (or `Large_Spec_Index`)
- **Mathematical Formula**:
  $$\text{COT\_Index\_Large}_t = 100 \times \frac{\text{Net\_Large\_Spec}_t - \min_{k=0}^{N-1}(\text{Net\_Large\_Spec}_{t-k})}{\max_{k=0}^{N-1}(\text{Net\_Large\_Spec}_{t-k}) - \min_{k=0}^{N-1}(\text{Net\_Large\_Spec}_{t-k})}$$
- **Inputs & Parameters**:
  - Lookback window $N$: 156 weeks (3-year) or 26 weeks / 52 weeks.
- **Signal Type**: Setup (Climax Exhaustion Setup).
- **Interpretation & Thresholds**:
  - **Over-Bullish Climax (`COT_Index_Large >= 80%`)**:
    - *Interpretation*: Systematic trend funds are 100% committed to the long side. "When a car is out of gas it cannot go any further. When all the fund buying power is used up it is very hard for a market rally to continue." Setup to **SELL SHORT / Fade the Funds**.
  - **Over-Bearish Climax (`COT_Index_Large <= 20%`)**:
    - *Interpretation*: Trend funds are heavily net short. Downside selling capacity is exhausted. Setup to **BUY LONG / Fade the Funds**.
  - **Setup vs. Trigger Rule**: Williams explicitly notes: *"It is not a precise 'sell today' index, but rather an excellent setup tool, telling us to get ready for the next major trend move in the market."*

---

### Feature 1.5: Large Trader 26-Week Channel Breakout Tracking ("Sweet Spot" Predictor)
- **Name**: `Fund_Channel_SweetSpot`
- **Mathematical Logic**:
  - Compute 26-week High / Low channel:
    $$\text{UpperChannel}_{26W} = \max_{k=1}^{26}(\text{High}_{t-k}), \quad \text{LowerChannel}_{26W} = \min_{k=1}^{26}(\text{Low}_{t-k})$$
  - Breakout Events:
    - $\text{Price}_t \ge \text{UpperChannel}_{26W} \implies$ Massive fund inflow / net long surge.
    - $\text{Price}_t \le \text{LowerChannel}_{26W} \implies$ Massive fund liquidation & aggressive shorting.
- **Inputs & Parameters**: Weekly High, Low, Close (26-week lookback).
- **Signal Type**: Setup / Flow Predictor.
- **Interpretation**:
  - Larry Williams demonstrated that 26-week channel breakouts represent the primary "sweet spot" where institutional funds execute systematic trend allocations across currencies, livestock, grains, and energies.
  - If a 26-week breakout occurs while Commercials are heavily shorting, the breakout is an institutional bull trap.

---

### Feature 1.6: Combined Speculative Net Index (Total Spec Positioning)
- **Name**: `Net_Spec_Composite` & `COT_Index_Spec`
- **Mathematical Formula**:
  $$\text{Net\_Spec}_t = \text{Net\_Large\_Spec}_t + \text{Net\_Small\_Spec}_t$$
  $$\text{COT\_Index\_Spec}_t = 100 \times \frac{\text{Net\_Spec}_t - \min_{k=0}^{N-1}(\text{Net\_Spec}_{t-k})}{\max_{k=0}^{N-1}(\text{Net\_Spec}_{t-k}) - \min_{k=0}^{N-1}(\text{Net\_Spec}_{t-k})}$$
- **Inputs & Parameters**: Weekly Non-Commercial and Non-Reportable net contracts; Lookback $N = 26$ or $156$ weeks.
- **Signal Type**: Setup (Speculative Consensus Regime).
- **Interpretation & Thresholds**:
  - Combines all non-commercial speculation against the hedging community.
  - `COT_Index_Spec > 80%` $\implies$ Speculative saturation (Bearish Setup).
  - `COT_Index_Spec < 20%` $\implies$ Speculative liquidation (Bullish Setup).

---

## Section 2: Volume Indicators & Empirical Rules

Larry Williams conducted extensive backtests on 15 years of S&P 500 futures, British Pound, and Treasury Bond data to separate empirical reality from technical analysis folklore.

### Feature 2.1: Volume Surge on Price Decline (Selling Climax / Absorption)
- **Name**: `Vol_Surge_Decline`
- **Mathematical Formula / Logic**:
  - Condition 1 (Single-Day / Intraday):
    $$\text{Close}_t < \text{Close}_{t-1} \quad \text{AND} \quad \text{Volume}_t > \text{Volume}_{t-1}$$
  - Condition 2 (Moving Average Expansion):
    $$\text{SMA}(\text{Close}, 8)_t < \text{SMA}(\text{Close}, 8)_{t-1} \quad \text{AND} \quad \text{SMA}(\text{Volume}, 8)_t > \text{SMA}(\text{Volume}, 8)_{t-1}$$
  - Condition 3 (Relative Surge):
    $$\text{Close}_t < \text{Close}_{t-1} \quad \text{AND} \quad \text{Volume}_t \ge 1.5 \times \text{SMA}(\text{Volume}, 20)_t$$
- **Inputs & Parameters**: Daily Close, Daily Volume; short MA lookback = 8 days, baseline MA = 20 days.
- **Signal Type**: Setup / Short-Term Reversal Trigger (BULLISH).
- **Empirical Evidence & Interpretation**:
  - Conventional theory labels price-down / volume-up as "bearish selling pressure."
  - Williams proved the exact opposite: **Price down on increasing volume is the single most profitable bullish pattern** across S&P 500 and currencies.
  - Tested on British Pound (60-day MA): Produced 102 trades, \$42,673 profit, 61% win rate, \$487 average trade, with an exceptionally smooth equity curve.
  - *Mechanism*: Represents panic dumping by weak hands into limit orders of commercial accumulators (institutional liquidity absorption).

---

### Feature 2.2: Advancing Price on Declining Volume (Exhaustion Rally)
- **Name**: `Price_Up_Vol_Down`
- **Mathematical Formula / Logic**:
  $$\text{Close}_t > \text{Close}_{t-1} \quad \text{AND} \quad \text{Volume}_t < \text{Volume}_{t-1}$$
  $$\text{Or: } \text{SMA}(\text{Close}, N)_t > \text{SMA}(\text{Close}, N)_{t-1} \quad \text{AND} \quad \text{SMA}(\text{Volume}, N)_t < \text{SMA}(\text{Volume}, N)_{t-1}$$
- **Inputs & Parameters**: Daily Price & Volume; $N \in [5, 8, 60]$.
- **Signal Type**: Setup (BEARISH across Commodities/Currencies; BULLISH Anomaly in Treasury Bonds).
- **Interpretation & Market Nuance**:
  - General Rule: Rallies on shrinking volume reflect a lack of buying commitment and exhaustion of market depth $\implies$ Bearish Setup.
  - **The Bond Exception**: In Treasury Bond futures, Williams found that `Price Up, Volume Down` was the single best performing bullish strategy (63% win rate, \$529/trade, \$12k drawdown), whereas it failed in other markets.

---

### Feature 2.3: Advancing Price on Advancing Volume (The "Conventional Trap")
- **Name**: `Price_Up_Vol_Up_Trap`
- **Mathematical Formula**:
  $$\text{SMA}(\text{Close}, 5)_t > \text{SMA}(\text{Close}, 5)_{t-1} \quad \text{AND} \quad \text{SMA}(\text{Volume}, 5)_t > \text{SMA}(\text{Volume}, 5)_{t-1}$$
- **Inputs & Parameters**: 5-day MA of price and volume.
- **Signal Type**: Warning / Negative Filter.
- **Interpretation**:
  - Textbook technical analysis asserts that advancing price on advancing volume is the pinnacle of bullishness.
  - Tested across 15 years on S&P 500: Resulted in an overall net loss. Fails as a standalone trend-following trigger because late-stage volume surges coincide with retail chase behavior.

---

### Feature 2.4: Multi-Timeframe Capitulation Setup (Trend + Volume Expansion)
- **Name**: `MTF_Capitulation_Buy`
- **Mathematical Logic**:
  1. Long-term trend is down: $\text{Close}_t < \text{SMA}(\text{Close}, 50)_t$
  2. Short-term price is declining: $\text{SMA}(\text{Close}, 8)_t < \text{SMA}(\text{Close}, 8)_{t-1}$
  3. Short-term volume is expanding: $\text{SMA}(\text{Volume}, 8)_t > \text{SMA}(\text{Volume}, 8)_{t-1}$
- **Inputs & Parameters**: Daily Close, Volume; Long-term $L = 50$, Short-term $S = 8$.
- **Signal Type**: Setup (High-Conviction Reversal Buy Setup).
- **Interpretation**: Identifies multi-week capitulation selloffs where volume surges into established long-term support. Produced superior, smoother equity performance than any other price-volume configuration tested by Williams.

---

### Feature 2.5: Larry Williams' ProGo (Professional vs. Public Intraday Accumulation/Distribution)
- **Name**: `ProGo` (Professional Buying vs. Public Buying Index)
- **Mathematical Formula**:
  - **Public Buying Component** (Overnight / Gap Reaction):
    $$\text{Public\_Bar}_t = \text{Open}_t - \text{Close}_{t-1}$$
    $$\text{Public\_Index}_t = \text{SMA}(\text{Public\_Bar}, 14)_t$$
  - **Professional Buying Component** (Regular Trading Hours Action):
    $$\text{Pro\_Bar}_t = \text{Close}_t - \text{Open}_t$$
    $$\text{ProGo}_t = \text{SMA}(\text{Pro\_Bar}, 14)_t$$
- **Inputs & Parameters**: Daily Open, High, Low, Close; Smoothing window = 14 days.
- **Signal Type**: Setup / Divergence Indicator.
- **Interpretation & Rules**:
  - **Premise**: Public orders dominate the open (reacting emotionally to overnight news, media, foreign markets). Professionals dominate intraday trading from open to close.
  - **Bearish Divergence**: Price rallies to new recovery or multi-week highs, but `ProGo` makes lower highs or trends downward $\implies$ Professionals are distributing into public open-driven rallies. Look to **SELL SHORT**. (Used by Williams to nail the Dec 7, 2004 crash in Silver).
  - **Bullish Divergence**: Price breaks to new lows, but `ProGo` forms higher lows or trends upward $\implies$ Professionals are accumulating intraday into public panic selling. Look to **BUY LONG**. (Used by Williams in Gold May 2004).

---

## Section 3: Open Interest (OI) Indicators & Rules

### Feature 3.1: Larry Williams' 1-Year Open Interest Stochastic
- **Name**: `OI_Stochastic_52W`
- **Mathematical Formula**:
  $$\text{OI\_Stochastic}_t = 100 \times \frac{\text{OI}_t - \min_{k=0}^{N-1}(\text{OI}_{t-k})}{\max_{k=0}^{N-1}(\text{OI}_{t-k}) - \min_{k=0}^{N-1}(\text{OI}_{t-k})}$$
- **Inputs & Parameters**:
  - Total Market Open Interest (daily or weekly).
  - Lookback window $N$: 52 weeks (12 months / 250 trading days).
- **Signal Type**: Setup (Speculative Saturation / Abandonment Setup).
- **Interpretation & Thresholds**:
  - **High OI Extreme (`OI_Stochastic >= 80%`)**:
    - *Interpretation*: The "swimming pool" is overflowing. Massive speculative interest and crowd participation. Market is near an unsustainable peak. **Bearish Setup**.
    - *"High levels of OI, as measured by a one-year stochastic indicator, are bearish and are usually seen shortly before a significant top."* (Documented across 20+ years of Silver, Soybeans, Gold, Lumber).
  - **Low OI Extreme (`OI_Stochastic <= 20%`)**:
    - *Interpretation*: Total speculative abandonment. The public and funds have exited the market; it has become a "pure commercial market." **Bullish Setup**.
    - *"More market bottoms and excellent buy points have been formed when OI is low than at any other time."*
  - **Asset Class Limitation**: Does **not** apply directly to equity index futures (e.g. S&P 500, Dow) due to quarterly expiration roll-over liquidations and synthetic arbitrage distortions.

---

### Feature 3.2: Open Interest Stochastic Momentum & Trendline Breakout Trigger
- **Name**: `OI_Trendline_Trigger`
- **Mathematical / Geometric Logic**:
  - Track swing highs and lows of `OI_Stochastic_52W`.
  - Draw a trendline connecting the descending swing highs when `OI_Stochastic` is low ($< 30\%$).
  - Draw a trendline connecting ascending swing lows when `OI_Stochastic` is high ($> 70\%$).
- **Inputs & Parameters**: `OI_Stochastic_52W` swing points.
- **Signal Type**: Timing Trigger.
- **Execution Rules**:
  - **Buy Trigger**: When `OI_Stochastic` is at historically depressed levels and breaks upward through its descending trendline $\implies$ Fresh commercial accumulation is entering the market.
  - **Sell Trigger**: When `OI_Stochastic` is at elevated levels and breaks downward through its ascending trendline $\implies$ Speculators are beginning to liquidate longs.
  - *Philosophical principle*: Avoids using price to predict price ($A \ne A$); uses the momentum of total participant interest.

---

### Feature 3.3: Larry Williams' "Inside Open Interest" Diagnostic Matrix
- **Name**: `Inside_OI_Matrix`
- **Mathematical / Conditional Logic**:
  Cross-references weekly changes in Price ($\Delta P$), Total Open Interest ($\Delta \text{OI}$), and Commercial Net Position ($\Delta \text{Comm\_Net}$):
  $$\Delta P = P_t - P_{t-1}, \quad \Delta \text{OI} = \text{OI}_t - \text{OI}_{t-1}, \quad \Delta \text{Comm\_Net} = \text{Comm\_Net}_t - \text{Comm\_Net}_{t-1}$$

| Setup State | Price Change ($\Delta P$) | Total OI Change ($\Delta \text{OI}$) | Commercial Net ($\Delta \text{Comm\_Net}$) | Underlying Market Dynamics | Quantitative Signal |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **State 1: Speculative Climax Top** | **UP** | **UP** | **DOWN (Selling)** | Public and funds are aggressively chasing the rally; Commercials are aggressively selling short into them. | **Strong Sell Setup** |
| **State 2: Commercial Capitulation Bottom** | **DOWN** | **UP** | **UP (Buying)** | Price drops sharply while OI increases, driven entirely by Commercials absorbing long contracts from panicking speculators. | **Strong Buy Setup** |
| **State 3: Public Despair Blast-off** | **DOWN** | **DOWN** | **UP (Buying)** | Public liquidating longs in despair (dwindling OI); Commercials quietly accumulating the remaining float. | **High-Conviction Buy Setup** |
| **State 4: Hollow Short-Covering** | **UP** | **DOWN** | **DOWN (Selling)** | Rally caused purely by speculative short-covering without commercial sponsorship. Commercials continue reducing longs/selling. | **Bearish Fade / Sell Setup** |

- **Empirical Cases Documented in Text**:
  - *Soybeans (April 2001)*: Price dropped heavily, OI increased sharply, but Commercials were heavily buying $\implies$ Market bottomed and exploded higher.
  - *Soybeans (October 2003)*: Price rallied, OI expanded, Commercials sold heavily $\implies$ Major market top.
  - *Live Cattle (Mad Cow Disease Jan 2004)*: News caused panic selloff, but Commercials bought massive contract volumes $\implies$ Massive rally (turned \$400 margin into \$11,000 profit).
  - *Lumber (Late 2004)*: Prices rallied with increasing OI while Commercials accounted for nearly 100% of the short OI $\implies$ Severe market collapse.

---

### Feature 3.4: Commercial Share of Open Interest (CL/OI and CS/OI Ratios)
- **Name**: `CL_OI_Ratio` and `CS_OI_Ratio`
- **Mathematical Formulas**:
  $$\text{CL\_OI\_Ratio}_t = \frac{\text{Commercial\_Long}_t}{\text{Total\_OI}_t}$$
  $$\text{CS\_OI\_Ratio}_t = \frac{\text{Commercial\_Short}_t}{\text{Total\_OI}_t}$$
- **Inputs & Parameters**: Weekly CFTC Commercial Longs, Commercial Shorts, Total Open Interest.
- **Signal Type**: Regime / Macro Setup.
- **Interpretation & Thresholds**:
  - Measures the exact percentage of the entire market open interest controlled by smart money.
  - **Commercial Short Saturation (`CS_OI_Ratio >= 55%` to `65%`)**:
    - When commercial short positions account for over 55%–65% of all open contracts in the market, smart money has taken overwhelming control of the short side.
    - **Signal**: Major Market Top / High-Probability Short Sale Setup (e.g., Japanese Yen peaks in 1988–1991, late 2004 at 64%).
  - **Commercial Short Depletion (`CS_OI_Ratio < 30%`)**:
    - Commercials refuse to short and hold negligible short inventory relative to total OI $\implies$ Commercials are net buyers.
    - **Signal**: Major Market Low / Long Buy Setup (especially when coinciding with seasonal lows, e.g. Japanese Yen late February/March).

---

### Feature 3.5: WILLCO (Williams Commercial Index - Normalized Commercial Net / Total OI)
- **Name**: `WILLCO`
- **Mathematical Formula**:
  $$\text{Raw\_WILLCO}_t = \frac{\text{Commercial\_Long}_t - \text{Commercial\_Short}_t}{\text{Total\_OI}_t} = \frac{\text{Net\_Commercial}_t}{\text{Total\_OI}_t}$$
  $$\text{WILLCO}_t = 100 \times \frac{\text{Raw\_WILLCO}_t - \min_{k=0}^{N-1}(\text{Raw\_WILLCO}_{t-k})}{\max_{k=0}^{N-1}(\text{Raw\_WILLCO}_{t-k}) - \min_{k=0}^{N-1}(\text{Raw\_WILLCO}_{t-k})}$$
- **Inputs & Parameters**:
  - `Commercial_Long`, `Commercial_Short`, `Total_OI`.
  - Lookback window $N$: **26 weeks** (Larry Williams' default and preferred modern standard; replaces legacy 3-year lookback for faster, more responsive regime detection in volatile markets).
- **Signal Type**: Regime & Setup Indicator.
- **Interpretation & Thresholds**:
  - **Bullish Regime / Buy Setup**: `WILLCO >= 75 - 80` (or `> 90`).
    - *Meaning*: Commercial net long position as a share of total open interest is at a 26-week extreme high. Commercials own the market. Look for trend triggers to go **LONG**.
  - **Bearish Regime / Sell Setup**: `WILLCO <= 20 - 25`.
    - *Meaning*: Commercial net short position as a share of total open interest is at a 26-week extreme high. Look for trend triggers to **SELL SHORT**.
  - **Early Warning Characteristic**: Commercials are "trend enders" and often accumulate early. Traders must wait for price confirmation rather than trading WILLCO in isolation.

---

## Section 4: Composite Speculator vs. Hedger Spreads

### Feature 4.1: Net Speculator vs. Hedger Spread (Single Market)
- **Name**: `Spec_Hedger_Spread`
- **Mathematical Formula**:
  $$\text{Spec\_Hedger\_Spread}_t = \text{Net\_Spec}_t - \text{Net\_Commercial}_t$$
  Where:
  $$\text{Net\_Spec}_t = \text{Net\_Large\_Spec}_t + \text{Net\_Small\_Spec}_t$$
  $$\text{Net\_Commercial}_t = \text{Commercial\_Long}_t - \text{Commercial\_Short}_t$$
  *(Note: Since $\text{Net\_Commercial} + \text{Net\_Large\_Spec} + \text{Net\_Small\_Spec} \approx 0$ by market clearing identity, $\text{Spec\_Hedger\_Spread}_t \approx -2 \times \text{Net\_Commercial}_t$.)*
- **Normalized Spread Index**:
  $$\text{Spread\_Index}_t = 100 \times \frac{\text{Spread}_t - \min_{k=0}^{N-1}(\text{Spread}_{t-k})}{\max_{k=0}^{N-1}(\text{Spread}_{t-k}) - \min_{k=0}^{N-1}(\text{Spread}_{t-k})}$$
- **Inputs & Parameters**: Weekly CFTC categories; $N = 26$ or $156$ weeks.
- **Signal Type**: Macro Regime / Setup.
- **Interpretation**: Measures the absolute tug-of-war between speculative momentum and commercial hedging capacity.
  - High Spread Index ($> 80$): Speculators completely dominate longs; Hedgers hold maximum short inventory $\implies$ High vulnerability to violent liquidation crashes.
  - Low Spread Index ($< 20$): Speculators completely liquidated; Commercials heavily net long $\implies$ Powerful structural floor.

---

### Feature 4.2: Multi-Market Dollar-Weighted Composite Disparity (Goepfert / Briese Index)
- **Name**: `Dollar_Weighted_COT_Disparity`
- **Mathematical Formula**:
  For an asset complex (e.g., US Equity Index Futures: S&P 500 big, E-Mini S&P, Dow Jones, Nasdaq 100):
  $$\text{Dollar\_Net\_Commercial}_t = \sum_{m \in M} \left( \text{Net\_Commercial}_{m,t} \times \text{Contract\_Multiplier}_m \times \text{Futures\_Price}_{m,t} \right)$$
  $$\text{Dollar\_Net\_SmallSpec}_t = \sum_{m \in M} \left( \text{Net\_Small\_Spec}_{m,t} \times \text{Contract\_Multiplier}_m \times \text{Futures\_Price}_{m,t} \right)$$
  $$\text{Dollar\_Disparity}_t = \text{Dollar\_Net\_SmallSpec}_t - \text{Dollar\_Net\_Commercial}_t$$
- **Inputs & Parameters**:
  - Full and E-Mini contracts across S&P, Dow, and NDX.
  - Contract multipliers and index settlement prices.
- **Signal Type**: Macro Crisis / Crash Setup.
- **Empirical Thresholds Documented by Jason Goepfert & Larry Williams**:
  - Historic extreme discrepancy: Commercials $\le -\$38\text{B}$ net short while Small Speculators $\ge +\$30\text{B}$ net long.
  - Only occurred three times in modern financial history:
    1. *Week ending 03/06/2001*: S&P plunged $-14\%$, NDX plunged $-32\%$ over the next few weeks.
    2. *Week ending 03/12/2002*: Led to the catastrophic multi-month summer equity liquidation of 2002.
    3. *December 2004*: Preceded immediate broad-market equity and precious metal meltdowns.
  - **Rule**: Whenever nominal dollar disparity reaches historic extremes ($>\$30\text{B}$ delta), initiate aggressive downside portfolio hedges and macro short positions.

---

## Section 5: Larry Williams' Execution Trigger Architecture

Throughout Chapters 5–10, Larry Williams repeatedly warns that **COT and Sentiment indicators are SETUP tools, never instantaneous execution triggers**, because Commercials scale in early.

### Feature 5.1: Short-Term Swing Trendline Violation Trigger
- **Name**: `LW_Trendline_Break_Trigger`
- **Execution Mechanism**:
  1. **Pre-condition (Regime/Setup)**:
     - Short Setup: `WILLCO <= 25` (or `COT_Index_Small > 75%` / `COT_Index_Large > 80%`) AND `ProGo` showing bearish divergence.
     - Long Setup: `WILLCO >= 75` (or `COT_Index_Small < 25%` / `COT_Index_Large < 20%`) AND `ProGo` showing bullish divergence.
  2. **Short Trigger Identification**:
     - Connect the two most recent swing reaction lows with a trendline.
     - **Execution**: Sell short on the first bar that violates this trendline (either by an open below the trendline or an intraday break below the line).
  3. **Long Trigger Identification**:
     - Connect the two most recent swing reaction highs with a downward trendline.
     - **Execution**: Buy long on the first bar that opens or penetrates above the trendline.
  4. **Risk Management / Stop-Loss Order**:
     - Stop placed immediately beyond the recent swing high (for shorts) or recent swing low (for longs).
     - Williams' Core Risk Rule: *"With a stop-loss order in place your risk is the same regardless of how good or bad the market looks. Your risk is your stop. You have a much better chance of not getting stopped out if you are buying when the public is selling."*

---

## Summary Matrix of Extracted Features

| ID | Feature Name | Source | Primary Parameters | Signal Role | Trading Action / Extreme Threshold |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **F-01** | `Net_Small_Spec` | Ch. 5 | CFTC Non-Reportables | Baseline | Input for normalization |
| **F-02** | `COT_Index_Small` | Ch. 5 | Lookback $N = 156$ or $26$ W | Setup | $>75\%$ Sell Short; $<25\%$ Buy Long |
| **F-03** | `Net_Large_Spec` | Ch. 6 | CFTC Non-Commercials | Baseline | Track fund accumulation/pyramiding |
| **F-04** | `COT_Index_Large` | Ch. 6 | Lookback $N = 156$ or $26$ W | Setup | $>80\%$ Fade Funds (Sell); $<20\%$ Fade Funds (Buy) |
| **F-05** | `Fund_Channel_SweetSpot` | Ch. 6 | 26-Week High/Low Channel | Flow Setup | Anticipate massive fund breakout flows |
| **F-06** | `COT_Index_Spec` | Ch. 6 | Composite Spec Net, $N=26$ W | Setup | $>80\%$ Spec Euphoria; $<20\%$ Capitulation |
| **F-07** | `Vol_Surge_Decline` | Ch. 7 | Price Close, Volume, 8d/20d MA | Setup/Trigger | Price Down + Vol Up = **BULLISH** Reversal |
| **F-08** | `Price_Up_Vol_Down` | Ch. 7 | Price Close, Volume, 5d/8d MA | Setup | Price Up + Vol Down = **BEARISH** (Except Bonds) |
| **F-09** | `Price_Up_Vol_Up_Trap` | Ch. 7 | 5-day Price/Volume MA | Filter | Long-term net loser; avoid chasing rallies |
| **F-10** | `MTF_Capitulation_Buy` | Ch. 7 | 50-day Price MA, 8-day Vol MA | Setup | Price < 50MA, Price 8MA Down, Vol 8MA Up = Buy |
| **F-11** | `ProGo` (Professional vs Public) | Ch. 10 | 14-day MA of (C - O) vs (O - C[1]) | Divergence | ProGo divergence vs Price peaks/troughs |
| **F-12** | `OI_Stochastic_52W` | Ch. 8 | Total OI, 52-week lookback | Setup | $>80\%$ Overcrowded Top; $<20\%$ Commercial Bottom |
| **F-13** | `OI_Trendline_Trigger` | Ch. 8 | Swing breaks on `OI_Stochastic` | Trigger | Downward break from $>70\%$ Sell; Upward break Buy |
| **F-14** | `Inside_OI_Matrix` | Ch. 9 | $\Delta P, \Delta\text{OI}, \Delta\text{Comm}$ | Setup | Matrix diagnostic of who controls the OI |
| **F-15** | `CS_OI_Ratio` / `CL_OI_Ratio` | Ch. 10 | Commercial Positions / Total OI | Macro Regime | CS/OI $\ge 55\%-65\%$ Top; CS/OI $<30\%$ Bottom |
| **F-16** | `WILLCO` | Ch. 10 | Net Comm / Total OI, $N=26$ W | Setup | $\ge 75-80$ Bullish Setup; $\le 20-25$ Bearish Setup |
| **F-17** | `Spec_Hedger_Spread` | Ch. 6/10 | Spec Net - Comm Net, $N=26$ W | Setup | Relative spread extremes between camps |
| **F-18** | `Dollar_Weighted_COT_Disparity`| Ch. 10 | Index complex dollar nominal | Macro Crisis | Comm $\le -\$38\text{B}$ vs Spec $\ge +\$30\text{B}$ = Crash Warning |
| **F-19** | `LW_Trendline_Break_Trigger` | Ch. 10 | 2-point swing trendlines | Trigger | Intraday / open penetration of swing trendline |

---
*Documentation prepared by `traders-volume-oi` for verification by `@verifier`.*
