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
  7. The original empty-run diagnosis and corrected safe-window contract.
  8. Out-of-sample results against ATM IV and the historical mean.
  9. What the failed incremental comparison means in finance terms.

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
   - Purpose: define the relative score against either ATM IV or the historical mean.
   - Symbols: model mean squared error $\mathrm{MSE}_m$ and benchmark mean squared error $\mathrm{MSE}_b$.
   - Delimiter: display.
7. GMM parameter count:
   - Purpose: derive the BIC penalty for full-covariance mixtures.
   - Symbols: component count $K$ and feature dimension $d$.
   - Delimiter: display.

## Planned code excerpts

1. File: `src/volatility_regimes/features/surface.py`
   - Function/block: delta interpolation and the three surface-shape formulas.
   - Why include this excerpt: it is the bridge from raw option rows to the daily state vector.
2. File: `src/volatility_regimes/walkforward/engine.py`
   - Function/block: `_apply_forward_target_embargo`.
   - Why include this excerpt: it protects the central methodological claim.
3. File: `src/volatility_regimes/walkforward/reporting.py`
   - Function/block: relative out-of-sample scores against ATM IV and the historical mean.
   - Why include this excerpt: it separates beating implied volatility from adding information beyond an unconditional forecast.

## Planned technical graphs

1. Graph type: SPX near-term ATM implied volatility through time with full-sample Gaussian-mixture regime shading.
   - Source: generate from tracked Parquet data with `blog/generate_charts.py`.
   - Expected takeaway: ordered states mostly separate volatility level, while transitions cluster around sharp changes.
2. Graph type: out-of-sample RMSE bars for all five forecast models and both symbols.
   - Source: generate from the frozen corrected walk-forward metric table.
   - Expected takeaway: the historical mean narrowly beats the GMM and linear models, while ATM IV and trailing realized volatility are weaker.
3. Graph type: cumulative GMM squared-error loss minus historical-mean loss.
   - Source: generate from the frozen row-level forecast panel.
   - Expected takeaway: any regime advantage is time-dependent and disappears by the end of the sample.

## Risks, gaps, and assumptions

- Data gaps: the tracked Parquet files are portable demo inputs. They cover 2010-01-04 through 2024-12-31 but should not be presented as production vendor history.
- Assumptions: descriptive regimes use the complete sample and are explicitly labelled in-sample. Implied volatilities and realized volatilities are decimals and annualized realized volatility uses 252 trading days.
- Corrected configuration: the old 3,880-row threshold exceeded the 3,872-row aligned panel before embargo. `min_train_size` now means post-embargo safe labels and is set to 2,520, approximately ten trading years. Impossible experiments fail before writing outputs.
- Result limitation: adjacent 20-day targets overlap by 19 returns, so daily loss observations are dependent and the report provides point estimates without overlap-adjusted inference.
- Validation checks to run before final draft: regenerate every chart; freeze summary CSV files under `blog/data/`; verify regime ordering; trace the forward-volatility shift and embargo inequality on a small date sequence; run the project tests and the blog validator; verify both language files reference the same images.
- Deployment note: the canonical workspace is `volatility-regimes/blog/`. The usual publish bundle would be under `~/projects/website/content/post/volatility-surface-regimes/`, but the user explicitly deferred publication. Nothing will be copied to, built in, committed in, or pushed from the website repository during this task.

## Outline review

The revised outline passes the four coverage checks: it states the research question, derives the regime and embargo methods, reports reproducible walk-forward evidence, and interprets the failed comparison against the historical mean. The main drafting risk remains overstating a teaching dataset; the article labels the full-sample regime chart as descriptive, treats overlapping losses as dependent, and limits its conclusion to the tracked demo data and one configuration.
