# GUIDE_walkforward

## Part 1: Conceptual Explanation

`src/volatility_regimes/walkforward/` is the out-of-sample forecasting layer.
Its role is to test predictive claims under strict temporal causality.

Given trade dates $d_1, d_2, \ldots, d_T$, each split trains on past dates and
forecasts the next block. The target for date $t$ and horizon $h$ is:

$$
\mathrm{RV}_t(h) = \mathrm{std}(r_{t+1}, \ldots, r_{t+h}) \times \sqrt{252}
$$

where $r_t = \log(P_t / P_{t-1})$ is the daily log return.

The critical leakage control is the forward-target embargo. If the first test
date is $s$, a training row dated $t$ is removed whenever its forward target
uses returns that overlap the test window.

Default forecast panel models:

- `atm_iv`
- `trailing_realized_vol`
- `linear_features`
- `gmm_regime_mean`

Optional robustness rows can be enabled with fixed `K` and HMM settings in
`walkforward.toml`.

## Part 2: Code Reference

- `walkforward.toml` (repository root): walk-forward sample, evaluation, regime,
  and output settings.
- `engine.py`: orchestration, embargo logic, and output writing.
- `models.py`: benchmark and regime-mean forecast helpers.
- `splits.py`: expanding and rolling split builders.
- `targets.py`: forward-target construction aligned to feature dates.
- `reporting.py`: CSV aggregation and markdown summary writing.

Where to start:

1. Read `engine.py` for end-to-end control flow.
2. Read `models.py` for each forecast row contract.
3. Read `tests/integration/test_walkforward_cli_integration.py` for invariants.

## Part 3: Short Journal

- 2026-04-19: Moved walk-forward code under `volatility_regimes` namespace,
  removed `sys.path` surgery, and standardized outputs to `outputs/walkforward/`.
- 2026-05-20: Moved experiment config to root `walkforward.toml` and split
  walk-forward artifacts into `outputs/reports/walkforward/` and
  `outputs/figures/walkforward/`.
