# GUIDE_src

## Part 1: Conceptual Explanation

The `src/` folder now has one coherent package namespace: `volatility_regimes`.
The goal of this structure is to keep the code interview-defensible by
separating responsibilities clearly.

The package is organized by workflow boundaries:

- `data_access`: reads and validates offline raw Parquet inputs and only falls
  back to ClickHouse when cache files are missing or invalid.
- `features`: builds daily volatility-surface features from options rows.
- `regimes`: fits latent-state models and ordered-regime labels.
- `descriptive`: computes full-sample diagnostics and descriptive charts.
- `walkforward`: runs leakage-safe out-of-sample forecasting experiments.
- `pipelines`: importable orchestration functions used by notebook and CLI.
- `cli`: thin wrappers that call package functions without business logic.

This structure keeps the notebook and command-line entrypoints on the same
implementation path, which reduces demo drift and makes behavior easier to
trace.

## Part 2: Code Reference

- `volatility_regimes/data_access/loader.py`: offline-first raw data loading.
- `volatility_regimes/features/surface.py`: feature engineering.
- `volatility_regimes/regimes/latent_models.py`: GMM and HMM fitting helpers.
- `volatility_regimes/descriptive/analytics.py`: descriptive metrics.
- `volatility_regimes/descriptive/plotting.py`: descriptive chart generation.
- `volatility_regimes/pipelines/descriptive_pipeline.py`: descriptive pipeline.
- `volatility_regimes/walkforward/engine.py`: walk-forward research engine.
- `volatility_regimes/cli/descriptive.py`: descriptive CLI wrapper.
- `volatility_regimes/cli/walkforward.py`: walk-forward CLI wrapper.

Where to start:

1. Read `volatility_regimes/GUIDE_volatility_regimes.md` for package map.
2. Read `volatility_regimes/walkforward/GUIDE_walkforward.md` for forecasting flow.
3. Read `tests/GUIDE_tests.md` for invariants protected by tests.

## Part 3: Short Journal

- 2026-05-20: Aligned package docs with split output dirs and root
  `walkforward.toml`.
