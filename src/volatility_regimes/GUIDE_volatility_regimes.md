# GUIDE_volatility_regimes

## Part 1: Conceptual Explanation

`volatility_regimes` is the reusable research backbone for the repository.
It keeps the full workflow modular while preserving one canonical execution
path for both notebook and CLI usage.

Subpackages and their roles:

- `data_access`: load and validate raw offline Parquet data.
- `features`: transform options chains into daily feature matrices.
- `regimes`: fit and reorder latent volatility regimes.
- `descriptive`: compute full-sample summaries and descriptive charts.
- `walkforward`: build targets and run leakage-safe forecast experiments.
- `pipelines`: orchestrate higher-level workflow steps.
- `cli`: lightweight wrappers for command-line execution.
- `project_paths.py`: shared repository-root and output-path constants.

The dependency direction is intentionally simple:

`data_access -> features -> regimes -> descriptive/walkforward -> pipelines -> cli`

## Part 2: Code Reference

- `project_paths.py`: repository root, config paths, and standard output dirs.
- `data_access/loader.py`: offline-first contract and ClickHouse fallback path.
- `features/surface.py`: interpolation and feature registry logic.
- `regimes/latent_models.py`: GMM/HMM model fitting and regime ordering.
- `descriptive/analytics.py`: realized-vol, VRP, transition, and regression outputs.
- `descriptive/plotting.py`: PNG chart writers under `outputs/figures/descriptive/`.
- `walkforward/engine.py`: out-of-sample runner and CSV/Markdown exports.
- `walkforward/models.py`: benchmark and regime-mean forecast helpers.
- `walkforward/splits.py`: split constructors.
- `walkforward/targets.py`: forward target construction.
- `walkforward/reporting.py`: metric aggregation and summary writer.
- `pipelines/descriptive_pipeline.py`: descriptive orchestration entrypoint.
- `cli/descriptive.py`, `cli/walkforward.py`: thin wrappers.

Where to start:

1. Read `walkforward/GUIDE_walkforward.md` for predictive workflow details.
2. Read `tests/integration/test_pipeline_cache_integration.py` for end-to-end offline behavior.
3. Read `tests/integration/test_walkforward_cli_integration.py` for walk-forward invariants.

## Part 3: Short Journal

- 2026-04-19: Introduced unified `volatility_regimes` namespace and moved
  orchestration code into importable pipelines with thin CLI adapters.
- 2026-05-20: Centralized path constants in `project_paths.py` and aligned
  outputs with `outputs/reports/` and `outputs/figures/`.
