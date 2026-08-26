# Volatility Regimes

Offline-first research repo asking one question: do latent volatility-surface
regimes improve out-of-sample forecasts of future realized volatility, versus
simple benchmarks? The project is notebook-first for teaching and interview
use, with reusable package code under `src/volatility_regimes/`.

## What it does

Uses SPX and NDX options and price data to build daily volatility-surface
features (ATM implied volatility, skew, term structure), fit latent regimes
with a Gaussian Mixture Model (GMM) and a Gaussian Hidden Markov Model (HMM),
and forecast forward realized volatility over a fixed horizon.

Two pipelines:

- **Descriptive** (`volatility_regimes.pipelines.descriptive_pipeline`): full-sample
  feature engineering, regime fitting, and descriptive tables/charts.
- **Walk-forward** (`volatility_regimes.walkforward.engine`): expanding-window
  out-of-sample forecasting with a forward-target embargo (training labels
  whose return window reaches the test date are dropped). Each test block
  compares five forecasts: current ATM implied volatility, expanding
  historical mean realized volatility, trailing realized volatility, an OLS
  linear regression on surface features, and the GMM regime-mean forecast.

Forward realized volatility for horizon $h$ trading days, annualization
factor $A$, and daily log return $r_t = \log(P_t / P_{t-1})$:

$$
\mathrm{RV}_t(h) = \mathrm{std}(r_{t+1}, \ldots, r_{t+h}) \times \sqrt{A}
$$

Full methodology, including the embargo definition and metrics: see
`docs/reference/walkforward_methodology.md` and `GUIDE_ROOT.md`.

## Requirements

- Python 3.13
- ClickHouse: only needed for a one-time cache refresh when the tracked raw
  files under `data/raw/` are missing or fail validation. Normal runs, tests,
  and the notebook work offline against `data/raw/`.
- If ClickHouse is used, these environment variables (read from `.env`):
  `CLICKHOUSE_HOST`, `CLICKHOUSE_PORT`, `CLICKHOUSE_USER`,
  `CLICKHOUSE_PASSWORD`, `CLICKHOUSE_SECURE`, `CLICKHOUSE_VERIFY`.

## Setup

```bash
uv sync --group dev
```

## Usage

```bash
uv run python -m pytest -v                          # run tests
uv run python -m volatility_regimes.cli.descriptive  # descriptive pipeline
uv run python -m volatility_regimes.cli.walkforward  # walk-forward pipeline
```

`scripts/run_descriptive.sh` and `scripts/run_walkforward.sh` are thin
wrappers around the two CLI commands above.

To run the teaching notebook top to bottom:

```bash
uv run jupyter nbconvert --to notebook --execute \
  notebooks/project_demo_walkthrough.ipynb \
  --output project_demo_walkthrough.executed.ipynb
```

## Configuration

- `config.toml`: shared data window (`symbols`, `start_date`, `end_date`),
  offline-cache policy (`[cache]`), surface-feature parameters (`[features]`),
  and regime model settings (`[regime]`, GMM `min_k`/`max_k`, HMM iteration
  and restart counts).
- `walkforward.toml`: walk-forward experiment settings — `horizons`,
  `feature_sets`, `min_train_size` (default `2520`, about ten trading years,
  the minimum leakage-safe training rows after the embargo), `step_size`, and
  regime model settings for the walk-forward run.

## Layout

```text
volatility-regimes/
├── config.toml               # shared data/feature/regime config
├── walkforward.toml          # walk-forward experiment config
├── src/volatility_regimes/   # package: data_access, features, regimes, descriptive, walkforward, pipelines, cli
├── scripts/                  # thin CLI wrapper scripts
├── tests/unit/, tests/integration/
├── data/raw/                 # tracked offline parquet inputs (~1.9 MB)
├── notebooks/                # main teaching walkthrough
├── docs/                     # methodology and offline-cache policy reference
└── outputs/                  # reports/ and figures/, descriptive and walkforward
```

## Output

- `outputs/reports/descriptive/`, `outputs/figures/descriptive/`: CSV tables
  and PNG charts from the full-sample analysis.
- `outputs/reports/walkforward/`: forecast panel CSV, metric summary CSV
  (out-of-sample $R^2$ versus both ATM implied volatility and the expanding
  historical mean), and a markdown summary.
- `outputs/figures/walkforward/`: walk-forward diagnostic plots, when
  generated.
