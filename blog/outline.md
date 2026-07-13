# Outline proposal

## Project scan summary

- Project archetype candidate: mixed `risk-model` and forecast-comparison research.
- Supporting evidence from files: `features/surface.py` builds seven daily surface features; `regimes/latent_models.py` fits Gaussian mixture and hidden Markov models; `walkforward/engine.py` compares four forecast rules under an expanding window and a forward-target embargo; `walkforward.toml` defines the tracked demo run.

## Blueprint selection

- Selected blueprint: mixed, with the risk-model sequence followed by a forecast-design audit.
- Why this blueprint fits this project: the descriptive layer asks whether volatility surfaces form distinct states, while the predictive layer asks whether a state-conditioned mean forecasts future realized volatility better than direct benchmarks.
- Planned section order:
  1. The forecast question, stated narrowly.
  2. Turning an option chain into level, skew, curvature, and term features.
  3. Building ordered latent regimes with a Gaussian mixture model.
  4. Defining forward realized volatility and the regime-mean forecast.
  5. Preventing overlap leakage with the target embargo.
  6. Descriptive evidence from the tracked SPX and NDX demo data.
  7. Why the configured walk-forward result is empty and what can still be concluded.
  8. A defensible next experiment.

## Planned equations

1. Daily log return and forward realized volatility:
   - Purpose: define the forecast target.
   - Symbols: close price $P_t$, log return $r_t$, horizon $h$, annualization factor $A$.
   - Delimiter: display.
2. Delta-space skew, butterfly, and term slope:
   - Purpose: connect option quotes to the regime feature vector.
   - Symbols: interpolated implied volatility $\sigma_t(\Delta,\tau)$, delta $\Delta$, maturity $\tau$.
   - Delimiter: display.
3. Gaussian mixture density and Bayesian information criterion:
   - Purpose: explain latent-state assignment and state-count selection.
   - Symbols: feature vector $x_t$, component weight $\pi_k$, mean $\mu_k$, covariance $\Sigma_k$, parameter count $p_K$, sample size $n$.
   - Delimiter: display.
4. Regime-mean forecast:
   - Purpose: state the model's predictive rule.
   - Symbols: ordered regime $z_t$ and training set $\mathcal{T}_t$.
   - Delimiter: display.
5. Forward-target embargo:
   - Purpose: show why ordinary chronological splitting is insufficient.
   - Symbols: training date position $i(t)$, test start position $i(s)$, horizon $h$.
   - Delimiter: display.
6. Out-of-sample $R^2$ versus ATM implied volatility:
   - Purpose: define the relative score produced by the reporting layer.
   - Symbols: model mean squared error and benchmark mean squared error.
   - Delimiter: display.

## Planned code excerpts

1. File: `src/volatility_regimes/features/surface.py`
   - Function/block: delta interpolation and the three surface-shape formulas.
   - Why include this excerpt: it is the bridge from raw option rows to the daily state vector.
2. File: `src/volatility_regimes/walkforward/engine.py`
   - Function/block: `_apply_forward_target_embargo`.
   - Why include this excerpt: it protects the central methodological claim.
3. File: `src/volatility_regimes/walkforward/reporting.py`
   - Function/block: relative out-of-sample score.
   - Why include this excerpt: it makes the benchmark comparison unambiguous.

## Planned technical graphs

1. Graph type: SPX near-term ATM implied volatility through time with full-sample Gaussian-mixture regime shading.
   - Source: generate from tracked Parquet data with `blog/generate_charts.py`.
   - Expected takeaway: ordered states mostly separate volatility level, while transitions cluster around sharp changes.
2. Graph type: regime profile chart for SPX and NDX.
   - Source: generate from frozen blog summary data.
   - Expected takeaway: higher ordered regimes have higher implied and forward realized volatility in the demo sample.
3. Graph type: ATM implied volatility versus 20-day forward realized volatility, coloured by descriptive regime.
   - Source: generate from tracked Parquet data.
   - Expected takeaway: the relationship is positive but dispersed; descriptive separation alone does not establish forecast improvement.

## Risks, gaps, and assumptions

- Data gaps: the tracked Parquet files are portable demo inputs. They cover 2010-01-04 through 2024-12-31 but should not be presented as production vendor history.
- Assumptions: descriptive regimes use the complete sample and are explicitly labelled in-sample. Implied volatilities and realized volatilities are decimals and annualized realized volatility uses 252 trading days.
- Known configured-result gap: after feature completeness, the 20-day forward target, and the 3,880-row minimum training requirement are combined, the default walk-forward run produces no forecast observations. The article will not report forecast superiority.
- Validation checks to run before final draft: regenerate every chart; freeze summary CSV files under `blog/data/`; verify regime ordering; trace the forward-volatility shift and embargo inequality on a small date sequence; run the project tests and the blog validator; verify both language files reference the same images.
- Deployment note: the canonical workspace is `volatility-regimes/blog/`. The usual publish bundle would be under `~/projects/website/content/post/volatility-surface-regimes/`, but the user explicitly deferred publication. Nothing will be copied to, built in, committed in, or pushed from the website repository during this task.

## Outline review

The outline passes the four coverage checks: it states the research question, derives the methodology, limits evidence to reproducible descriptive outputs, and separates practical interpretation from claims the empty configured forecast panel cannot support. The main drafting risk is overstating what full-sample clusters prove; the evidence section and conclusion will keep prediction and description separate.
