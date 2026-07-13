# Walk-Forward Methodology Reference

This document defines the alignment and evaluation contract implemented by
`src/volatility_regimes/walkforward/`. It is the ground truth for configuration,
tests, and generated reports.

## Units and target alignment

Let $P_t$ be the closing index level on trading date $t$. The daily log return
is

$$
r_t=\log(P_t/P_{t-1}).
$$

For horizon $h\geq2$ and annualization factor $A$, the target attached to date
$t$ is

$$
\mathrm{RV}_t(h)
=\operatorname{std}(r_{t+1},\ldots,r_{t+h})\sqrt{A}.
$$

The standard deviation uses the sample convention. Volatility values are
annualized decimals. With $A=252$, `0.20` means 20 percent annualized
volatility.

Every model is evaluated on dates where all selected surface features, the
forward target, and the trailing realized-volatility benchmark are present.
This common aligned panel prevents models from receiving different forecast
dates.

## Forward-target embargo

Chronological ordering alone does not prevent target overlap. Let $i(d)$ be
the position of date $d$ in the complete sorted price history, and let $s$ be
the first date of a test block. The target at training date $t$ is safe only if

$$
i(t)+h<i(s).
$$

The strict inequality removes every label whose final return reaches the first
test date. Features at those recent training dates are harmless, but their
labels are not yet observable at the forecast origin.

## Minimum training history and capacity

`evaluation.min_train_size` in `walkforward.toml` is the minimum number of
labelled training rows remaining after the embargo. The default is `2520`,
approximately ten trading years at 252 days per year.

The expanding split builder begins at the configured row count. The engine
embargoes each candidate train set and skips early candidates until at least
`min_train_size` safe labels remain. Before model fitting, it also places the
last aligned row in a one-row test set and computes the maximum possible safe
training count. A smaller maximum raises `ValueError` with:

- symbol, horizon, and feature set
- aligned row count
- maximum safe training row count
- requested minimum training size

The engine never writes an empty forecast panel for an impossible
configuration.

## Forecasts

The default output contains five rows per symbol and forecast date:

| Model | Forecast |
|:--|:--|
| `atm_iv` | Current near-term at-the-money implied volatility |
| `historical_mean` | Mean training target after embargo |
| `trailing_realized_vol` | Realized volatility over the preceding $h$ returns |
| `linear_features` | Ordinary least-squares prediction from the selected surface features |
| `gmm_regime_mean` | Mean training target in the predicted ordered GMM regime |

GMM means Gaussian Mixture Model. Optional Hidden Markov Model (HMM) and fixed
state-count rows are enabled through the regime settings.

One test block has one training window, so model fitting occurs once per block.
This gives the same GMM and linear predictions as fitting separately for every
test row. An HMM is also fitted once, but each test date is decoded from the
training sequence plus only that row. This retains its chronological
information set.

## Metrics

For actual target $y_t$, forecast $\hat y_t$, and $N$ forecast dates, mean
squared error (MSE), root mean squared error (RMSE), and mean absolute error
(MAE) are

$$
\begin{aligned}
\mathrm{MSE} &= \frac{1}{N}\sum_{t=1}^{N}(y_t-\hat y_t)^2, \\
\mathrm{RMSE} &= \sqrt{\mathrm{MSE}}, \\
\mathrm{MAE} &= \frac{1}{N}\sum_{t=1}^{N}|y_t-\hat y_t|.
\end{aligned}
$$

For model $m$ and benchmark $b$, relative out-of-sample $R^2$ is

$$
R^2_{\mathrm{OOS},m\mid b}
=1-\frac{\mathrm{MSE}_m}{\mathrm{MSE}_b}.
$$

The report uses both `atm_iv` and `historical_mean` as benchmarks. All compared
errors use the same symbol, horizon, feature set, and dates. A positive score
means lower squared error than the named benchmark. These are point estimates;
the default report does not adjust statistical inference for overlapping
20-day targets.
