# Trading App Improvement Roadmap

## 🔴 High Priority
- [ ] **Task 1: Signal Structure Unification**
  - Align `app.py` trade plan with `execution_dashboard.py` expectations.
  - Map `bias` -> `action`, `stop_loss` -> `sl`, etc.
- [ ] **Task 2: Dashboard Orchestration**
  - Import and invoke `render_execution_dashboard` inside `app.py`.

## 🟡 Medium Priority
- [ ] **Task 3: Risk/Position Sizing Engine**
  - Implement `position_size_usd` calculation based on account equity and risk per trade.
- [ ] **Task 4: Stability & Error Boundaries**
  - Harden Monte Carlo and pattern detection against `NaN` values.

## 🟢 Low Priority
- [ ] **Task 5: Metric Parity**
  - Synchronize `app.py` "Deep Dive" with the full `MetricsDashboard`.
