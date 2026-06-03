# GUIDE_ROOT

## Part 1: Conceptual Explanation

This repository studies whether option-implied volatility-surface structure
contains predictive information that survives out-of-sample testing. The code
is organized as an offline-first package in `src/volatility_regimes/` and uses
tracked raw Parquet files under `data/raw/` as the normal runtime path.

There are two linked layers.

The first layer is the **full-sample descriptive pipeline** in
`volatility_regimes.pipelines.descriptive_pipeline`. It reads raw options and
price data, builds daily surface features, fits latent regimes, and writes CSV
tables plus PNG charts under `outputs/reports/descriptive/` and
`outputs/figures/descriptive/`.

The second layer is the **walk-forward forecasting pipeline** in
`volatility_regimes.walkforward.engine`. For each trade date, it trains on past
data only, predicts forward realized volatility, and records out-of-sample
results under `outputs/reports/walkforward/`. The key leakage control is the
forward-target embargo. If horizon is $h$ days, a training row at date $t$ is
dropped whenever its target window touches the test window.

The main forecasting target is forward realized volatility:

$$
\mathrm{RV}_t(h) = \mathrm{std}(r_{t+1}, \ldots, r_{t+h}) \times \sqrt{A},
$$

where $r_t = \log(P_t / P_{t-1})$ is the daily log return and $A = 252$ is the annualization factor.

The walk-forward layer compares four baseline ideas:

- current ATM implied volatility
- trailing realized volatility
- a linear regression on selected surface features
- a Gaussian Mixture Model regime-mean forecast

Optional robustness sweeps can add fixed-`K` regime models when enabled in
`walkforward.toml`.

The root of the repository owns shared data and feature configuration in
`config.toml`. The walk-forward pipeline reads experiment settings from
`walkforward.toml` and merges sample-window settings into the root config.

## Part 2: Code Reference

- `README.md`: user-facing overview and offline-first quickstart.
- `config.toml`: shared data, cache, and feature configuration.
- `walkforward.toml`: walk-forward experiment settings.
- `scripts/run_descriptive.sh`, `scripts/run_walkforward.sh`: thin CLI wrappers.
- `src/GUIDE_src.md`: map of the package layout under `src/`.
- `src/volatility_regimes/data_access/loader.py`: offline-first raw data loader.
- `src/volatility_regimes/features/surface.py`: surface-feature extraction.
- `src/volatility_regimes/regimes/latent_models.py`: GMM/HMM fitting helpers.
- `src/volatility_regimes/descriptive/analytics.py`: descriptive metrics.
- `src/volatility_regimes/descriptive/plotting.py`: descriptive charts.
- `src/volatility_regimes/pipelines/descriptive_pipeline.py`: descriptive orchestration.
- `src/volatility_regimes/walkforward/engine.py`: walk-forward orchestration.
- `src/volatility_regimes/cli/descriptive.py`: descriptive CLI wrapper.
- `src/volatility_regimes/cli/walkforward.py`: walk-forward CLI wrapper.
- `notebooks/project_demo_walkthrough.ipynb`: main teaching artifact.
- `docs/user/offline_cache_policy.md`: offline raw-data policy contract.
- `tests/unit/`, `tests/integration/`: automated verification for data loading, modeling, and pipelines.

Where to start:

1. Read `README.md` for the research framing.
2. Read `src/volatility_regimes/GUIDE_volatility_regimes.md` for package map.
3. Read `src/volatility_regimes/walkforward/GUIDE_walkforward.md` for predictive workflow.
4. Read `tests/GUIDE_tests.md` for protected invariants.
5. Open `notebooks/project_demo_walkthrough.ipynb` for the full offline walkthrough.

## Part 3: Short Journal

- 2026-05-20: Aligned repository layout with standard `src/`, `scripts/`,
  `tests/unit`, `tests/integration`, `docs/user`, and split output dirs.
