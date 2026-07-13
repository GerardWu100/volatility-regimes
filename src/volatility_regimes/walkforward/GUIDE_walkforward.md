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

Let $i(d)$ be the integer position of date $d$ in the sorted price history. A
training label at $t$ is safe for a test window beginning at $s$ only when

$$
i(t)+h<i(s).
$$

`min_train_size` is checked after applying this inequality. The default value
is 2,520 safe labels, approximately ten trading years. Early candidate splits
are skipped until this count is available. Before fitting any model, the engine
also computes the largest safe training set that could leave one test row. If
that maximum is smaller than the requested minimum, the run raises a detailed
configuration error rather than producing empty CSV files.

Default forecast panel models:

- `atm_iv`
- `historical_mean`
- `trailing_realized_vol`
- `linear_features`
- `gmm_regime_mean`

The historical mean forecast is the expanding mean forward realized
volatility in the embargoed training labels. It answers a necessary question:
does a feature or regime model add information beyond the unconditional target
level observed so far?

All rows within one test block share one training window. Linear, GMM, and
optional HMM models therefore fit once per block. GMM assigns the whole block
under that train-only fit. For HMM forecasts, each date is still decoded as the
last observation of the training sequence plus that single test row, so later
test rows never enter an earlier date's state assignment.

The metric summary reports root mean squared error, mean absolute error, and
out-of-sample $R^2$ against two benchmarks. For model $m$ and benchmark $b$,

$$
R^2_{\mathrm{OOS},m\mid b}
=1-\frac{\mathrm{MSE}_m}{\mathrm{MSE}_b},
$$

where $\mathrm{MSE}$ is mean squared error on identical forecast dates. A
positive value improves on the benchmark. A negative value is worse.

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
- 2026-07-13: Corrected the minimum-window contract to count post-embargo
  labels, added capacity validation and the historical-mean benchmark, and
  batched fits across each fixed-training test block.
