# GUIDE_tests

## Part 1: Conceptual Explanation

The `tests/` folder protects the two research claims that matter most in this
repository:

1. the feature and regime code does what the methodology says it does
2. the walk-forward layer is leakage-safe and reproducible

Tests are split by scope:

- `tests/unit/`: fast checks of formulas, feature construction, regime fitting,
  and walk-forward helpers.
- `tests/integration/`: end-to-end offline cache behavior, walk-forward CLI
  runs, and notebook teaching contracts.
- `tests/fixtures/`: reserved for shared static payloads when needed.

For the walk-forward package, the core invariant is temporal causality. The split builders must keep the training window strictly before the test window, and the CLI runner must additionally embargo any training label whose forward return window would overlap the test horizon. Without that second rule, the forecast target would leak future returns even if the split itself looks correct.

Model tests focus on forecast behavior instead of implementation internals.
For example, HMM tests verify that forecast decoding uses the full
train-plus-test sequence.

When adding new research behavior, prefer tests that protect a real methodological claim:

- no lookahead leakage
- correct target alignment
- correct benchmark naming in exported panels
- correct handling of selected versus fixed regime counts
- correct offline loading from `data/raw/`

## Part 2: Code Reference

### Unit tests

- `unit/test_feature_extraction.py`: checks surface-feature construction and data shape.
- `unit/test_feature_sets.py`: verifies named feature subsets used by walk-forward.
- `unit/test_regime_model.py`: covers GMM/HMM fitting and regime ordering.
- `unit/test_cache_policy.py`: checks offline-first `data/raw/` contract.
- `unit/test_walkforward_splits.py`: checks split construction.
- `unit/test_walkforward_targets.py`: checks target alignment and realized-vol formulas.
- `unit/test_walkforward_models.py`: tests benchmark and regime forecast helpers.

### Integration tests

- `integration/test_pipeline_cache_integration.py`: verifies end-to-end offline cache behavior.
- `integration/test_walkforward_cli_integration.py`: verifies walk-forward outputs, embargo,
  and config defaults.
- `integration/test_project_demo_notebook.py`: enforces notebook teaching and structure
  contract.

Where to start:

1. Read `integration/test_pipeline_cache_integration.py` for offline path guarantees.
2. Read `integration/test_walkforward_cli_integration.py` for walk-forward invariants.
3. Read `integration/test_project_demo_notebook.py` if notebook behavior changes.

## Part 3: Short Journal

- 2026-04-19: Updated imports and contracts for the new `volatility_regimes`
  package and `data/raw/` offline data path.
- 2026-05-20: Split tests into `unit/` and `integration/` to match the standard
  repository layout.
