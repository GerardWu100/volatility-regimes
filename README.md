# Volatility Surface Regime Research

Offline-first quantitative research repository for one central question:

> Do latent volatility-surface regimes improve out-of-sample forecasts of future realized volatility relative to simple benchmarks?

The project is intentionally notebook-first for teaching and interview use, with
reusable package code under `src/volatility_regimes/`.

## Research Framing

Core definitions:

- **ATM IV** (at-the-money implied volatility): implied volatility near delta
  magnitude `0.50`.
- **Forward realized volatility**: future standard deviation of log returns over
  a fixed horizon.
- **Regime-mean forecast**: infer the current latent regime, then forecast with
  the historical mean target in that regime.
- **Target embargo**: remove training labels whose future return window reaches
  the first test date.

For horizon $h$ trading days:

- $P_t$: close price at day $t$
- $r_t = \log(P_t / P_{t-1})$: daily log return
- $A$: annualization factor (default `252`)

$$
\mathrm{RV}_t(h) = \mathrm{std}(r_{t+1}, \ldots, r_{t+h}) \times \sqrt{A}
$$

The walk-forward layer enforces an embargo so training labels never overlap
returns used by the test window. In `walkforward.toml`, `min_train_size` is the
minimum number of labelled training rows that must remain after this embargo.
The default is `2520`, approximately ten trading years at 252 days per year.
If the aligned sample cannot leave that many safe labels plus one test row, the
CLI raises a configuration error instead of writing empty result files.

The default panel compares five forecasts:

- current ATM implied volatility
- expanding historical mean realized volatility
- trailing realized volatility
- linear regression on the selected surface features
- Gaussian Mixture Model (GMM) regime mean

The historical mean is the minimum test for incremental model information. A
regime forecast can beat ATM implied volatility while adding nothing beyond the
unconditional realized-volatility level.

## Offline-First Quickstart

1. Sync environment:

```bash
uv sync --group dev
```

2. Run tests:

```bash
uv run python -m pytest -v
```

3. Run descriptive pipeline:

```bash
uv run python -m volatility_regimes.cli.descriptive
# or: ./scripts/run_descriptive.sh
```

4. Run walk-forward pipeline:

```bash
uv run python -m volatility_regimes.cli.walkforward
# or: ./scripts/run_walkforward.sh
```

5. Execute notebook top-to-bottom:

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/project_demo_walkthrough.ipynb --output project_demo_walkthrough.executed.ipynb
```

## Notebook-First Learning Path

Use `notebooks/project_demo_walkthrough.ipynb` as the primary walkthrough.
It demonstrates the full offline workflow from `data/raw/` through:

1. raw file validation
2. price and options loading
3. target construction
4. feature engineering
5. latent regime modeling
6. walk-forward forecast evaluation
7. interpretation and limitations

## Offline Portability Contract

- Normal runs read only from `data/raw/` when those files are valid.
- ClickHouse is attempted only when required raw files are missing or invalid.
- A fresh clone with tracked raw files can run tests, CLIs, and notebook
  without `.env`.

Detailed policy: `docs/user/offline_cache_policy.md`

## Portable Demo Data

Tracked raw files live in `data/raw/`:

- `options_spx.parquet` and `options_spx.metadata.json`
- `options_ndx.parquet` and `options_ndx.metadata.json`
- `prices_spx.parquet` and `prices_spx.metadata.json`
- `prices_ndx.parquet` and `prices_ndx.metadata.json`

Total payload is approximately `1.9 MB`, so the repository stays lightweight.

## One-Time ClickHouse Cache Refresh

ClickHouse is optional and used only for one-time cache population/refresh when
raw files are missing or fail validation. Day-to-day research runs should not
require database access.

## Repository Layout

```text
volatility-regimes/
├── config.toml
├── walkforward.toml
├── src/volatility_regimes/
├── scripts/
├── tests/unit/
├── tests/integration/
├── data/raw/
├── outputs/reports/
├── outputs/figures/
├── notebooks/
└── docs/user/
```

## Package Layout

```text
src/volatility_regimes/
├── data_access/
├── features/
├── regimes/
├── descriptive/
├── walkforward/
├── pipelines/
└── cli/
```

## Outputs

- `outputs/reports/descriptive/`: CSV and text tables from full-sample analysis.
- `outputs/figures/descriptive/`: PNG charts from full-sample analysis.
- `outputs/reports/walkforward/`: forecast panel CSV, metric summary CSV, markdown
  summary. The metric table reports out-of-sample $R^2$ versus both ATM implied
  volatility and the expanding historical mean.
- `outputs/figures/walkforward/`: walk-forward diagnostic plots when generated.
