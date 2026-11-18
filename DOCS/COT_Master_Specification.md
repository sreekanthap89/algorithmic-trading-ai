# COT Master Specification — Larry Williams, *Trade Stocks and Commodities With the Insiders* (Wiley, 2005)

**Status:** Verified synthesis of `COT_Features_Ch1_Ch4.md`, `features_traders_volume_oi_chapters_5_9.md`, `COT_Features_Ch10_Ch15.md`
**Verified against:** book text (PDF `Secrets of the COT Report 3.pdf`, 225 pp.; book page = PDF page − 16).
**Author:** `verifier`

Legend — **[B]** explicit in book · **[B-ex]** book states it only as an example/illustration · **[X]** extrapolation / implementation default NOT in the book (tunable, must be validated by own backtest) · **[?]** book is ambiguous or does not disclose the rule.

---

## 0. Verification Summary (read first)

### 0.1 Corrections to worker findings
| # | Worker claim | Book reality (page) | Resolution |
|---|---|---|---|
| C1 | WILLCO = stochastic of **Net Comm / OI** (all 3 workers) | Ch10 p.105: the formula is `Stochastic Custom(COT Commercials/Open Interest, N)` = net/OI, **but** "in all the remaining charts in this chapter we are now using the total commercial **short** position divided by OI and a 26-week measure". Gold example (p.111): "WILLCO poking above 90 … commercials own most of the OI … look for a rally". | Implement **both** variants: `WILLCO_net` (default) and `WILLCO_short` (stoch of CS/OI, **inverted**: use `100 − stoch` so high = bullish). Polarity of Williams' charts is not explicit → [?]; validate on yen/gold/silver sample charts. |
| C2 | Book COT-Index example gives 33% | Ch4 p.34 example: current 350, low −150, high 750 → book prints "(200÷600)×100 = 33%". Arithmetic is internally wrong (should be 500/900 = 55.6%). | Formula is standard stochastic; **ignore the book's numeric example** (typo). Use unit test with 350/−150/750 → 55.56. |
| C3 | `Vol_Surge_Decline` Condition 3 (`Vol ≥ 1.5×SMA20`) | Not in book. | Mark [X]. Book tests only: (a) single-day price↓ & vol↑ vs prior day (S&P, p.68-69), (b) MA(n) of price vs MA(n) of volume direction, **same n** for both (BP: n=60; S&P: 8-day MAs + 50-day trend filter). |
| C4 | "Price-up/Vol-down = **bearish short setup**" | All volume tests are **long-only** (S&P: buy next open, exit first profitable close or **$3,500 stop**; BP: buy, **exit after 20 days, no stop**, MA(t) > MA(t-2)). "Bearish" = *worse long results*, never tested as a short. In S&P Up/Down was still a mild winner (better than Up/Up); in BP Up/Up made $20,123/127 trades, Up/Down $9,028/133, Down/Down $6,983/99, **Down/Up $42,673/102 (61%, $487/trade)**. Bonds: Up/Down best (63%, $529/trade, ~$12k DD). | Use as **long-entry rankers/filters**, not short signals, unless own testing proves otherwise. Per-market polarity map required (bonds invert). |
| C5 | `Price_Up_Vol_Up_Trap` "fails everywhere" | Net loser on **S&P 1990-2004 only**; BP Up/Up was profitable. | Scope to S&P-type; not universal. |
| C6 | Dollar disparity "occurred 3 times; threshold −$38B/+$30B" | Ch10 p.107 (Goepfert quote, Dec 11 2004): **current** readings were −$38B comm / +$30B nonreportable; "only one other occurrence comes close … 03/06/01 (S&P −14%, NDX −32% over next few weeks); next-closest probably 03/12/02". | Thresholds are *descriptive of Dec-2004 level*, not a derived rule. n≈2-3. Implement as **percentile-of-history alert**, not a hard $38B constant (OI/price scale drifts [X]). |
| C7 | Gold/USD ratio = `(SMA3 − SMA21)/100` | Ch13 p.154-155: "subtract the **3-week MA from the 21-week MA**, divide by 100 … when the 3-week to 21-week spread is greater than 30 percent, decline in gold". Literal text = SMA21−SMA3, but meaning (overbought when short MA ≫ long MA) implies SMA3−SMA21. | Sign ambiguity [?]. Implement `(SMA3−SMA21)/100 > 0.30` as overbought; "spread" definition (Gold − USD index, raw price points) is as worker stated. Weekly only; horizon ≥6 months. |
| C8 | Ch.11 "Synthetic Commercial Proxy" has algorithmic mechanics (close in upper half, volume absorption…) | Ch11 p.131: Williams explicitly **"not yet willing to fully reveal the index"**; only says: "buy on scale down, sell on scale up", 6-month basis, 80/20 zones, works best on quality stocks, leads the real COT. | **Formula undisclosed → [?]**. Worker's mechanics are invention. Spec only a *plug-in interface* (`ProxyIndex(series) -> 0..100`, 26-wk stochastic, 80/20) and a clearly-labelled placeholder (e.g., accumulation/distribution-type flow). Do **not** present as Williams' formula. |
| C9 | Jiler 20%/40%/5% | p.9-10: only "**be wary of positions more than 40% from long-term average and disregard deviations of less than 5%**"; "net-net" = actual net OI position minus seasonal norm (month-end avg over several yrs; examples 6-yr, 10-yr). ±20% is **not** a rule; it comes from examples (e.g., hedgers 20% above 6-yr avg). | Keep 5% noise / 40% caution as [B]; ±20% as [X]. Net position is as **% of OI** (examples quote %-net-long) [B-ex]. |
| C10 | cot-core "|Swing26| ≥ 75% = rule", "Silver M ≥ 260wk", "OI-share 50% dominance", "COT-03 CL%OI > 50%" | Not rules. BP 26-wk swing is a **narrative example** (p.30: 14-yr-high buying → 10-yr-low selling in 26 wks). Silver: "22 years commercials never net long" (p.~1785). | Mark as [X]/[B-ex]. |
| C11 | `Fund_Channel_SweetSpot` "breakout with commercial selling = bull trap" | Ch6 p.60-62: funds' net position jumps when price makes a **new 26-week high/low** (cotton, CAD, hogs) — descriptive study of fund behaviour. Not a trade rule. "Trap" interpretation is an extrapolation. | Keep as *explanatory feature/diagnostic* (fund-flow predictor); the fade rule is Large-Trader Index >80/<20 [B]. |
| C12 | OI-stochastic thresholds 80/20, trendline 70/30 | Ch8 p.83-89: "12-month stochastic of OI"; high = bearish, low = bullish; trigger = trendline break on the OI-stochastic indicator (wheat Fig 8.11). **No numeric thresholds stated.** Does not work cleanly on financials (expiry spikes; p.85-86). | Thresholds [X] (default 80/20 for setup; trendline logic per book). Exclude equity-index/financial futures or use roll-adjusted OI. |
| C13 | Consecutive-down-closes rows are "exactly N" | Ch10 Table 10.2: N=1 row has **2,507 trades = all down closes**, so rows are cumulative "≥ N"/streak-length-reached counts; trade = buy next open, exit that day's close; **S&P only, 1982-2004, no costs/stops**. | Define as "after ≥N consecutive down closes". Treat 1982-2004 P&L as non-tradeable evidence; re-test with costs. |
| C14 | Inside-day = `H≤H₁ & L≥L₁` | Book (Ch10 p.118): inside bar = **lower high and higher low** (strict). | Use strict inequality as default; non-strict as option. |
| C15 | Chapter 15 uses WILLCO ≥75-80 | Ch15 p.197-199: uses **"OI tool" (low/high OI) + COT index (relative level, "we use the index approach")**; not explicitly WILLCO. OI "low" is subjective "or use the OI tool I described earlier". | Tier-2 = `COT_Index` (or WILLCO) – configurable. |
| C16 | Ch13 "short setup" rules & the S&P/BP table | Table 13.1 (p.151) is **buy side only**: `52-wk MA(t) > MA(t-2)` AND `6-mo COT idx > 80`; no exit rule stated, no costs; profits $: BP 51,311/21/71%, Bonds 51,736/26/69%, SF 19,093/16/56%, S&P 82,683/24/54%, Gold 13,285/13/69%. Trend definition in prose uses **MA(t) > MA(t-1)**. Short mirror (idx<20 & MA falling) is described in prose (Bonds, Wheat) but not tabulated. | Parameter `trend_lag ∈ {1,2}` weeks (default 2 for table replication). |
| C17 | "Position-sizing discipline" (12.5) listed as a quantitative rule | Ch12 p.146: psychological "don't plunge"; **no sizing formula in book**. Stop must be dollar-based (p.138). | Sizing is [X]: fixed-fractional risk model is implementer's choice. |
| C18 | cot-core: Combined long/short delta formula | Book p.15: "converted to futures-equivalent using delta factors supplied by the exchanges"; no formula. CFTC publishes combined positions already netted. | ETL ingests CFTC-reported combined figures; do not recompute deltas. |

### 0.2 Confirmed verbatim (high confidence)
COT-Index formula & 3-year default, 80/20 (p.34) · "index is not a timing tool" · multi-year (4-year rule of thumb) highs/lows (Lesson One, p.29-30) · Silver "never net long 22 yrs" · Public index >75 sell, <25 weak buy (Ch5) · Large-trader index >80 sell/<20 buy, "not a precise sell today index" (Ch6) · OI 12-month stochastic (Ch8) · CS/OI ≥55% top (yen; 64% Nov-2004) (Ch10 p.102-103) · WILLCO 26-wk default, 75-80 buy zone, 90 in gold example · ProGo (14-day avg of Close−Open) · 17-day lowest-low/highest-high trailing stop excluding inside days (p.145) · re-entry at 13-day extreme · exit after 10+ weeks / few markets trend >15 weeks · 6-month COT idx >50 long filter (S&P 371 trades +$76,605, DD $39,145 vs 846 trades −$170, DD $209,755; Bonds 606 +$28,418 DD $17,695 vs 633 −$13,985 DD $25,733) · 52-week MA trend filter · 39-week MA filter (Ch15) · Fibonacci refutation (Ch14, Tables 14.1-14.4).

---

## 1. Data Ingestion & CFTC ETL Specification

### 1.1 Source & cadence
| Item | Spec |
|---|---|
| Snapshot | Positions as of **Tuesday close** (`report_date`). |
| Release | **Friday 15:30 ET** [B p.14]. Holiday weeks → release slides (typically Monday or next business day) — `available_at` must come from the **actual publication timestamp**, never `report_date + 3d` (federal-holiday and 2013/2018-19/2025 shutdown delays) [X]. |
| History (book era) | Futures-only since 1986; futures+options combined since 1995 [B]. |
| Formats | **Short** (reportable vs non-reportable, commercial / non-commercial / spreading, change, %OI, trader counts) and **Long** (adds crop-year Old/Other, concentration top-4/top-8 gross & net) [B]. |
| Report types | `futures_only` (default for book replication) and `combined` (futures + delta-adjusted options). Never mix in one series. |

### 1.2 Required fields (Legacy format — the format the book uses)
`report_date, available_at, market_code, contract_name, exchange, open_interest, comm_long, comm_short, noncomm_long, noncomm_short, noncomm_spread, rep_long, rep_short, nonrep_long, nonrep_short, traders_total, traders_comm_long/short, traders_noncomm_long/short, (long form) conc_gross_4/8, conc_net_4/8`.

### 1.3 Identities / validation (assert at load)
- `rep_long = comm_long + noncomm_long + noncomm_spread`; same for shorts [B]
- `nonrep_long = OI − rep_long`; `nonrep_short = OI − rep_short` [B p.16]
- `Σ longs = Σ shorts = OI`. Net positions across 3 groups ≈ 0 **only because spreading is symmetric**: `Net_comm + Net_noncomm + Net_nonrep = 0`. ⇒ `Net_Spec = Net_noncomm + Net_nonrep = −Net_comm` exactly (so `Spec_Hedger_Spread = −2·Net_comm`, redundant — do not build a separate signal; see 3.7).
- `% of OI` columns re-derived must match CFTC's within 0.1 rounding.
- Reject rows with `OI ≤ 0`, missing group, or identity violations > 1 contract.

### 1.4 Point-in-time (look-ahead) rules
1. Every weekly COT row is joined to price bars with `available_at ≤ bar_decision_time` (as-of join, backward).
2. **Daily bars:** a row released Friday 15:30 ET is **not** known during Friday's regular session (most futures settle before 15:30 ET) ⇒ first usable decision point = **Friday evening session / Monday open**. Default: signal computed at weekend, **execute Monday open** (matches book: "every weekend scan through the list", p.153).
3. **Weekly bars** (book's indicator charts): align the weekly price bar (ending Friday) with the COT row whose `available_at` falls in/before that bar close; do **not** stamp the COT value on the Tuesday-dated position — it would leak Wed–Fri data into the indicator on a Tuesday bar.
4. **Intraday bars:** forward-fill last released COT state; state changes only at `available_at`. Between releases the state is constant for 5 trading days (≈ stale up to 10 calendar days at next release).
5. Lag awareness: at release, positions are 3 days stale; Williams: commercials are "trend enders… usually a little early" — do not treat COT change as same-day information.

### 1.5 Legacy vs Disaggregated vs TFF (not in book — implementer decision)
| Format | Availability | Mapping to Williams categories |
|---|---|---|
| **Legacy** (Commercial / Non-commercial / Non-reportable) | 1986→ | **Native to book. Use for all replication.** |
| **Disaggregated** (Producer/Merchant/Processor/User; Swap Dealers; Managed Money; Other Reportables) — physical commodities | 2006→ | Commercial ≈ PMPU (+ optionally Swap Dealers; **not identical**); Non-comm large ≈ Managed Money + Other Reportables; reclassification breaks continuity [X]. |
| **TFF** (Dealer; Asset Manager; Leveraged Funds; Other) — financial futures | 2006/2009→ | Commercial ≈ Dealer/Intermediary (+Asset Mgr debated); "Large Specs" ≈ Leveraged Funds [X]. |
Rules: store the raw group columns; build a **mapping table** per (market, format); never splice formats inside one index lookback; if switching, require re-warm-up of N weeks.

### 1.6 Structural breaks & hygiene
- **Reporting-level changes** (e.g., 20-Jan-2005 per Table 2.5: S&P E-mini 300→1,000; T-Bond 1,000→1,500) shift the reportable / non-reportable split ⇒ discontinuities in `Net_Small`. CFTC aims 70-90% OI coverage [B]. Maintain a `reporting_level_changes` table and either flag/mask windows or use the level-invariant commercial series for long lookbacks.
- Contract code changes/merges (E-mini vs full S&P, Nasdaq-100 NQ/ND, Dow YM/DJ) — map to a **canonical market id** and aggregate where the book aggregates (Ch11).
- Spreading/Old-Other crop-year: use "All" columns; "Old+Other ≠ All" (p.17-18).
- Dedupe revised releases (keep latest `available_at`, but record first-seen for PIT backtests).
- Warm-up: `N_weeks` of valid history before first signal (156 for 3-yr, 208 for 4-yr, 52, 26). Mask outputs inside warm-up.
- Epsilon guard on `max−min`; if range < ε → index = 50 (neutral) [X].
- Price/volume/OI data: **daily or weekly OI is published by exchanges daily**; COT's OI is Tuesday-only. For `OI_Stochastic` use exchange OI (weekly resample on Friday), never mix with COT OI timestamps.
- Continuous futures: back-adjusted prices for trend/stop calculations; **unadjusted OI/volume summed across contracts** (book notes financial-futures OI/volume spikes at expiry, p.85-86).

---

## 2. Core COT Oscillators & Mathematical Indicators

Notation: `CL, CS` commercial long/short; `NCL, NCS` non-commercial; `NRL, NRS` non-reportable; `OI` total open interest; weekly index `t`.

| ID | Name | Formula | Params | Role | Thresholds | Prov. |
|---|---|---|---|---|---|---|
| **F2.1** | Commercial Net | `CP = CL − CS` | — | Input | Absolute sign **not** a signal (structural shorts: silver 22 yrs, coffee…) | B |
| **F2.2** | Net % OI | `CP%OI = 100·CP/OI` | — | Normalised level; cross-market | — | B-ex (Jiler/WILLCO) |
| **F2.3** | Commercial share of OI | `CL%OI = 100·CL/OI`; `CS%OI = 100·CS/OI` | — | Regime / Setup | `CS%OI ≥ 55%` → top zone (yen; 64% Nov-04); `CS%OI < 30%` → bottom [30% from workers—[X]/book gives "decreasing" language only]; lumber late-2004: commercials ≈ 100% of short OI | B (55%), X (30%) |
| **F2.4** | Weekly net flow | `ΔCP_t = CP_t − CP_{t−1}` (also ΔCL, ΔCS separately) | 1 wk | Diagnostic | — | X (derived from "who is driving") |
| **F2.5** | **COT Index (Williams)** | `100·(CP_t − min_N CP)/(max_N CP − min_N CP)`, window **includes current week** | **N=156 wks** (legacy default, p.33); N=26 (preferred modern, Ch10/13); N=52 optional | **Setup** (not timing) | `>80` bullish setup; `<20` bearish; book also cites 75 ("more than 75 → 75% more buying… found at lows", p.36); extremes 0/100; 3 neutral | B |
| **F2.6** | **Multi-year extreme** (Lesson One) | bull if `CP_t ≥ max_{N}(CP)`; bear if `CP_t ≤ min_N(CP)`; equivalently COT-Index(N)=100/0 | **N=208 wks (4 yrs)** "rule of thumb"; examples 3/5/6/7/10/14/18 yrs | **Regime** (macro setup) | Extremes need *not persist* — commercials unwind into trend (p.29) | B |
| **F2.7** | **WILLCO** | `r_t = (CL−CS)/OI`; `100·(r_t − min_26 r)/(max_26 r − min_26 r)` | **N=26** (default), 156 legacy; variant on `CS/OI` (see C1) | Setup | `≥75-80` buy zone (`≥90` strong, gold May-04); `≤20-25` sell | B (N, 75-80, 90), [?] polarity of short variant |
| **F2.8** | "Silver rule" (structural-short break) | `CP_t > 0 ∧ max_{1..M}(CP_{t−k}) ≤ 0` | M = lookback of persistent shorting (silver 22 yrs≈1,144 wks; workers' 260 is [X]) | Rare regime trigger | "If commercials ever go net long silver, look for a major bull market" | B |
| **F2.9** | Jiler seasonal deviation | `Dev_t = Net%OI_t − mean_{y=1..Y}(Net%OI at same calendar month)` | Y=6-10 yrs; noise 5%; caution 40% | Secondary setup | Bullish: hedgers above norm + large specs net long + small traders net short below norm; bearish mirror | B (5%/40%), X (20%) |
| **F2.10** | COT-Index rotation | `COTIdx_t − COTIdx_{t−26}`; or `CP_t − CP_{t−26}` | 26 wks | Diagnostic / regime-change flag | BP example only; no threshold in book | B-ex; threshold X |

Implementation notes
- All stochastics are **rolling, inclusive of current week, on weekly COT series**, then forward-filled to bars (§1.4).
- Provide `lookback_weeks` as a per-market parameter; the book itself says "you can play with this variable … see what number of weeks works best" (p.105).
- Two readings of "relative" vs "absolute": gold May-2004 index 69% **while net short** (p.35) → never gate on `CP>0`.

---

## 3. Sentiment & Speculator Positioning Indicators

| ID | Name | Formula | Params | Role | Rule | Prov. |
|---|---|---|---|---|---|---|
| **F3.1** | Net Small Spec | `NRL − NRS` | — | Input | Level alone not a signal (cotton net long through 6.5-yr bear) | B |
| **F3.2** | **Public (Small-Trader) Index** | stochastic of F3.1 | N=156 (book; "exact same formula"); 26/52 [X] | Setup (contrarian) | `>75` → market slide / **short setup** (reliable: lumber, hogs, bonds, gold, cotton); `<25` → buy **only with commercials aligned** (lumber low did not always rally; bonds/hogs did) | B |
| **F3.3** | Net Large Spec | `NCL − NCS` | — | Input | Funds are trend followers, add on breakouts; max position at trend end | B |
| **F3.4** | **Large-Trader Index** | stochastic of F3.3 | N=156 [B]; 26 [X] | Setup (fade funds) | `>80` → look for sells; `<20` → look for buys; "out of gas" logic; "**not a precise sell-today index**" | B |
| **F3.5** | Fund 26-week channel footprint | `UpCh=max(High,26wk)`, `LoCh=min(Low,26wk)`; fund net jumps on new 26-wk high/low | 26 wks | Diagnostic / flow predictor | Funds' "sweet spot" = 26-week breakout (cotton, CAD, hogs). Anticipates fund pyramiding; **not** a standalone trade | B (behaviour), X (use) |
| **F3.6** | Composite "who is where" panel | COT-Index (comm), Large Idx, Public Idx plotted together | same N | Confirmation | Best bull: Comm idx high + Large idx low + Public idx low; mirror for bear [B-ex Ch6 end, Jiler config] | B-ex |
| **F3.7** | Spec-vs-Hedger spread | `Net_Spec − Net_Comm` | — | — | **Mathematically `= −2·CP`** → identical to COT Index; **do not ship as independent feature** (worker F-17/F-06 duplicates; `COT_Index_Spec` ≡ 100−`COT_Index_comm`) | derived |
| **F3.8** | Index-complex dollar disparity (Goepfert/Briese) | `Σ_m Net_m·Mult_m·Price_m` over ES, SP, NQ, ND, YM, DJ for Commercials and Non-reportables | contract multipliers (ES 50, SP 250, NQ 20, ND 100, YM 5, DJ 10 [X, external]) | Macro crash warning | Dec-2004: comm −$38B, nonrep +$30B; 03/06/01 and 03/12/02 comparable → S&P −14%, NDX −32% in following weeks. Use percentile-of-history, n≈3 ⇒ low statistical confidence | B-ex |
| **F3.9** | Synthetic commercial proxy (equities / non-reporting commodities) | **Undisclosed** (C8) | 26-wk stochastic, 80/20 | Plug-in setup | Williams: buy on scale-down, sell on scale-up; leads real COT; quality stocks only | [?] |

---

## 4. Open Interest & Volume Mechanics

### 4.1 Volume (all tests: long-only; see C3-C5)
| ID | Name | Logic | Params | Evidence | Use |
|---|---|---|---|---|---|
| **F4.1** | Price↓ / Volume↑ (single day) | `Close_t<Close_{t−1} ∧ Vol_t>Vol_{t−1}` → **buy next open** | S&P test exit = first profitable close, stop $3,500 | S&P: best/most tradable curve; BP (MA form) +$42,673/102/61% | Bullish setup/ranker |
| **F4.2** | Price↓/Vol↑ MA form | `MA_n(Close)↓ ∧ MA_n(Vol)↑` with same n; "↓" = `MA_t < MA_{t−2}` (BP) | n=60 (BP, exit after 20 days, no stop); n=5 (S&P) | as above | Bullish |
| **F4.3** | Capitulation buy (best curve) | `Close<MA50(Close)` ∧ `MA8(Close)↓` ∧ `MA8(Vol)↑` | 50/8 | S&P: smoothest curve of all tests (p.70-71) | Bullish setup |
| **F4.4** | Price↑/Vol↓ | `MA↑ ∧ MA(Vol)↓` | n=5/8/60 | S&P mild +; BP +$9,028; **Bonds best bullish (63%, $529)** | **Market-specific polarity**; "bearish" only in relative sense |
| **F4.5** | Price↑/Vol↑ ("textbook bullish") | MA price↑ ∧ MA vol↑ | 5-day | S&P net loser; BP $20,123/127 | Don't treat as bullish confirmation |
| **F4.6** | ProGo / Public Index | `Pro_t = Close_t − Open_t`; `ProGo = SMA14(Pro)`; `Public_t = Open_t − Close_{t−1}`; `PublicIdx = SMA14` | 14 days | Divergence: price new high with ProGo failing to make new high = distribution (silver Oct-Dec 2004); price new low with ProGo rising = accumulation (gold May-04). Also usable as trend/entry tool [B p.109] | Confirmation | B |
| **F4.7** | Consecutive down closes | buy next **open** after ≥N down closes, exit same close | N≥3 | S&P 1982-2004 Table 10.2: N=3 +$38,563/480; N=4 +$93,210/182; N=5 +$56,783/59; N=1,2 negative; **no costs** | Short-term long add-on [X for modern use] |
| **F4.8** | Bar-type stats | up-range/down-range/inside/outside counts (2,156/1,890/565/592) | — | Context only | — |

### 4.2 Open Interest
| ID | Name | Logic | Params | Rule | Prov. |
|---|---|---|---|---|---|
| **F4.9** | **OI 12-month stochastic** | `100·(OI−min_52)/(max_52−min_52)` weekly | N=52 wks | High = crowd in the pool = **bearish**, "usually seen shortly before a significant top"; low = "pure commercial market", "more bottoms formed at low OI". Not for financial futures w/o roll handling. Thresholds: 80/20 default [X] | B |
| **F4.10** | OI trendline trigger | Trendline on F4.9: when indicator is **high** → take trend-line **breaks down** as sell entry; when **low** → breaks **up** as buy entry. Rising-from-low OI ⇒ trend turns up (p.89) | swing detection on indicator | Entry trigger | B (concept), X (swing/thresholds) |
| **F4.11** | Conventional OI/price table (rejected) | OI↑/P↑ bull, OI↑/P↓ bear, OI↓/P↑ bear, OI↓/P↓ bull — "correct only ≈33% of time" | — | Anti-pattern | B |
| **F4.12** | **Inside-OI diagnostic** ("who drives OI") | classify week by sign(ΔP), sign(ΔOI), sign(ΔCP): see table | weekly | Setup | B for rows 1-3; row 4 [X] |

| State | ΔP | ΔOI | ΔCP | Read | Evidence |
|---|---|---|---|---|---|
| Speculative top | ↑ | ↑ | ↓ | Crowd chasing, commercials selling → **sell setup** | Soy Oct-03, lumber late-04 [B] |
| Commercial capitulation bottom | ↓ | ↑ | ↑ | Commercials absorbing panic → **buy** | Soy Apr-01, cattle Jan-04 [B] |
| Despair blast-off | ↓ | ↓ | ↑ | Public liquidating, commercials accumulate → **buy** | Fig 9.3, 9.5 lumber [B] |
| Hollow rally | ↑ | ↓ | ↓ | Short-covering → fade [**X**: not explicit] |

### 4.3 Combined Tier model (Ch15)
Setup = **OI low (bull) / OI high (bear)** AND **commercial index buying (bull) / selling (bear)**; then Tier-3 trend permission (§5.3). Coffee 1995-2000: 6 sell setups (OI high + COT low), 5 on target, late-1999 premature but >$34,000 per contract on $1,500 margin; all buys rejected by falling 39-wk MA [B].

---

## 5. Systematic Setup & Entry / Exit Trading Rules

### 5.0 Role taxonomy (book's hierarchy)
- **Regime/Setup:** COT Index, WILLCO, multi-year extreme, CS%OI, Public/Large idx, OI stochastic, Inside-OI, Jiler.
- **Permission filters:** 6-mo COT >50 / <50; trend MA (52-wk or 39-wk); seasonal window.
- **Confirmation:** ProGo divergence.
- **Trigger:** price trendline violation (open-through or intrabar), breakout of 13-day extreme, classical chart-pattern breaks.
- Book stance: "The index is not a timing tool"; "commercials are trend enders… a little early… look for a price trend change" (p.127).

### 5.1 Rule A — "One-Minute Weekly Scan" (Ch13; best-documented, replicable)
```
every weekend (after Friday 15:30 ET release, weekly bar closed):
  trend_up   = SMA52(weekly close)[t] >  SMA52[t-lag]     # lag=2 for Table 13.1, lag=1 prose
  trend_down = SMA52[t] < SMA52[t-lag]
  LONG  setup if trend_up   and COT_Index_26wk > 80
  SHORT setup if trend_down and COT_Index_26wk < 20        # tested in prose only (bonds, wheat)
  discard COT buys in downtrend; discard COT sells in uptrend
entry: Monday open (or options: buy calls in up-setup / puts in down-setup)
exit: NOT specified in book for Table 13.1 → use §6 exit stack
```
Replication targets (Table 13.1): BP $51,311/21/71%; Bonds $51,736/26/69%; SF $19,093/16/56%; S&P $82,683/24/54%; Gold $13,285/13/69%. Orange juice 2000-04: 7 setups; wheat 1998-2002 sells only.

### 5.2 Rule B — Unified COT/OI/ProGo/Trendline system (Ch10)
```
SETUP : WILLCO_26 ≥ 75-80 (long) or ≤ 20-25 (short)         [B]
OPT.  : seasonal window agrees (yen low late Feb/Mar; gold rally ~July, top Dec/Jan) [B-ex]
CONFIRM: ProGo divergence vs price (price↓ & ProGo↑ for long; price↑ & ProGo↓ for short) [B]
TRIGGER: short-term trendline through 2 most recent swing lows (short) / highs (long);
         enter at first bar that OPENS beyond or trades through the line;
         if missed, enter next open (silver 7-Dec-2004 short on opening gap) [B]
```
Williams: "I would not use a trend line like this at all times … only because there was a very bearish setup."

### 5.3 Rule C — Tripartite model (Ch15)
1. OI stochastic low (bull) / high (bear) [or visually low/high].
2. Commercial COT index high (bull) / low (bear).
3. **39-week MA** of weekly close: long allowed if MA rising, or price > MA, or (MA flat and price depressed); short allowed if MA falling or price < MA.

### 5.4 Rule D — Directional filter for short-term systems (Ch10)
Only run **long** systems when 6-month COT Index > 50; run short only < 50. Evidence S&P/Bonds above. Williams: "never, ever would I trade just because the index was >50, but it's a great place to start."

### 5.5 Rule E — Contrarian-crowd setups
- Public Index >75 → short setup; Large-trader index >80 → short setup; Both + commercial index <20 = highest-conviction short (and mirror for long) [B-ex].
- Public <25 buys only with commercial confirmation [B].

### 5.6 Rule F — Macro/slow filters
- Multi-year extreme (4-yr) → regime flag; "silver rule".
- Gold/USD spread ratio >0.30 → ≥6-month bearish bias for gold (weekly; not short-term) [B]; low ratio → favourable for long.
- Gold seasonality: buy-window ~July, sell/avoid year-end → modifier only.

### 5.7 Chart-pattern triggers (Ch14, optional library)
Trendline, channel, triangle/pennant, flag (bracket orders; **leave the unfilled opposite bracket as initial stop**), 1-2-3 reversal (enter on break of point 2), H&S neckline, double/triple tops-bottoms (enter on break of intervening swing). Moving-average **crossovers rejected**; MAs only as slope filters. Fibonacci levels rejected (avg retracement 84% for 5% rallies '92-94, 118% '02-03, 75% for 2% rallies, 92% on 60-min bars; Parent: clusters at 87.5%).

### 5.8 Terminology note
Book contains **no** thrust-bar / volatility-breakout / false-break formulas (those are in *Long-Term Secrets*). Do not attribute them to this source.

---

## 6. Risk Management, Sizing & Execution Filters

| ID | Rule | Spec | Prov. |
|---|---|---|---|
| **R1** | Dollar stop (Ch12 p.138) | `stop = entry ∓ MaxDollarRisk / PointValue`; stops protect **dollars**, not chart lines; never based on patterns/MAs; always use one ("trapeze net"). Max $ risk chosen from account size | B (principle), X (risk %) |
| **R2** | 17-day trend trailing stop (p.145) | **Only if in a major trend**: long stop = lowest low of last **17 trading days**; short = highest high of 17 days; **inside days excluded** from count; trading days not calendar days | B |
| **R3** | 13-day re-entry | After stop-out, re-enter at **highest high of last 13 days** (long) / lowest low (short) "if it looks like the trend is again exerting itself" | B (trend judgement = [?]; use §5.1/5.3 filter as proxy) |
| **R4** | Time exit | Exit "target" after in trade **10+ weeks**; "few markets go up/down >15 weeks without an important correction" → time-stop 10-15 weeks | B |
| **R5** | Windfall exit | "massive, quick, explosive windfall moves should be exited quickly" (p.158) — define as e.g. >k·ATR gain in ≤5 bars [X] | B/X |
| **R6** | Flag bracket stop | opposite bracket order = stop | B |
| **R7** | Anti-plunging | Constant rules independent of size; no formula (C17) | B |
| **R8** | Cost/slippage / liquidity | Coffee: "not highly liquid, lots of slippage"; trade liquid markets (Bonds). Apply per-market slippage in backtests; book P&Ls are gross | B-ex/X |
| **R9** | Execution timing | Signals at weekend scan → Monday open; limit use of COT inside Fri 15:30–Mon open (§1.4) | X |
| **R10** | Options alternative | Use long calls/puts for small accounts at Rule-A setups (limited risk) | B |
| **R11** | Setup invalidation | Don't require COT extreme to persist (dynamic unwinding): exit via R2/R4, not "index fell below 80" | B |
| **R12** | Margin examples | Bonds $3,000; lumber/coffee $1,500 — illustrative only | B-ex |

Execution filter stack (recommended order): data PIT validity → market has ≥ warm-up → **trend permission** → **COT/WILLCO setup** → **OI state** → optional ProGo/seasonal → trigger → R1 stop → R2/R4/R5 exits.

---

## 7. Edge-Case Test Matrix (for implementers)
1. **Friday-release leakage:** daily backtest must not use Friday COT on Mon–Fri bars before `available_at`. Test: shift row by one bar and assert PnL changes.
2. **Holiday week:** release Monday; first usable bar = Tuesday open (not Monday) if release after cash close.
3. **Warm-up:** first 155 weeks NaN for N=156.
4. **Flat range:** max==min → index 50.
5. **Structural-short market:** index 80+ while `CP<0` still bullish (gold May-2004 69%, crude 100%).
6. **Format switch / reclassification:** break index series; recompute.
7. **Reporting-level change (Jan-2005 Table 2.5):** mask `Net_Small` stochastic ±N weeks.
8. **Combined vs futures-only:** keep separate series.
9. **OI roll spikes:** financial futures → OI stochastic unreliable (use roll-adjusted/`total across months`).
10. **Book arithmetic:** Ch4 example (33%) is wrong; correct = 55.56%.
11. **Weekly↔daily alignment of MA trend filters:** SMA52/39 on **weekly closes**; evaluate on the closed weekly bar only.
12. **Seasonal norm leakage:** Jiler norm must use only prior years (expanding, PIT).

---

## 8. Open / Undisclosed Items (cannot be implemented from the book)
- Synthetic commercial proxy formula (Ch11) — withheld by author.
- Exact polarity of WILLCO on commercial **short**/OI (C1).
- OI-stochastic numeric thresholds, trendline swing definition.
- Entry/exit for Table 13.1 trades; position sizing formula; "windfall" definition.
- Per-market optimal N (book: tune 26-156).

## 9. Source Traceability (feature → book page)
COT identities/format p.14-18 · reporting levels Table 2.5 p.21 · Jiler p.9-10 · multi-year Lesson One p.29-30 · COT Index p.33-38 · Silver p.38-40 & p.~1785 · Public idx Ch5 p.41-54 · Large traders Ch6 p.55-64 (26-wk p.60-62) · Volume Ch7 p.65-78 · OI Ch8 p.79-89 · Inside OI Ch9 p.91-98 · CS/OI & WILLCO Ch10 p.101-107 · ProGo p.108-111 · coin-flip/consecutive closes p.117-121 · proxy Ch11 p.129-135 · stops/17-13 day Ch12 p.138,145 · weekly scan & gold Ch13 p.147-160 · charts/Fibonacci Ch14 p.165-195 · tripartite Ch15 p.197-204.
