"""Generate frozen data and technical charts for the project blog post.

The script reads the tracked portable demo Parquet files, reconstructs the
project's seven delta-space surface features, fits full-sample Gaussian mixture
models for descriptive analysis, freezes the completed production walk-forward
outputs, and writes reproducible CSV and PNG artifacts under ``blog/``. Run the
walk-forward CLI before this script.

Outputs
-------
blog/data/regime_summary.csv
    One row per symbol and ordered descriptive regime.
blog/data/model_selection.csv
    Bayesian information criterion values for candidate regime counts.
blog/data/sample_audit.csv
    Row counts for the corrected default walk-forward configuration.
blog/data/walkforward_metric_summary.csv
    Frozen production metrics for the five default forecast models.
blog/data/walkforward_forecast_panel.csv
    Frozen row-level production forecasts used by the loss-path chart.
blog/images/01_spx_regime_timeline.png
    SPX near-term ATM implied volatility with descriptive regime shading.
blog/images/02_oos_rmse.png
    Out-of-sample root mean squared error by symbol and model.
blog/images/03_cumulative_loss_difference.png
    Cumulative GMM squared-error difference versus the historical mean.

Notes
-----
The fitted regimes use the complete sample and are therefore descriptive, not
out-of-sample predictions. All volatility values are stored as decimals.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


BLOG_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BLOG_ROOT.parent
DATA_DIR = BLOG_ROOT / "data"
IMAGE_DIR = BLOG_ROOT / "images"
WALKFORWARD_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports" / "walkforward"
SYMBOLS = ("SPX", "NDX")
ANNUALIZATION_FACTOR = 252
FORWARD_HORIZON = 20
CANDIDATE_REGIME_COUNTS = tuple(range(2, 7))
RANDOM_SEED = 42
GMM_INITIALIZATIONS = 5
DEFAULT_MIN_TRAIN_SIZE = 2_520
REGIME_COLORS = ("#2a6fbb", "#22a7a7", "#e2b04a", "#dc6b45", "#a33e5c", "#6b3c8f")


def _interpolate_wing(
    frame: pd.DataFrame, option_type: str, target_delta: float
) -> pd.Series:
    """Interpolate implied volatility at a fixed delta for every date and expiry.

    Parameters
    ----------
    frame : pandas.DataFrame
        Option rows for one symbol. Required columns are ``trade_date``,
        ``dte``, ``option_type``, ``delta``, and ``mid_iv``.
    option_type : str
        Option side, either ``"p"`` for put or ``"c"`` for call.
    target_delta : float
        Requested signed delta. The portable data brackets the target at 0.20
        and 0.30 in absolute value.

    Returns
    -------
    pandas.Series
        Interpolated implied volatility indexed by ``(trade_date, dte)``.

    Raises
    ------
    ValueError
        If the target is not bracketed by two observed deltas.
    """
    side = frame.loc[frame["option_type"] == option_type]
    pivot = side.pivot(index=["trade_date", "dte"], columns="delta", values="mid_iv")
    observed_deltas = np.asarray(pivot.columns, dtype=float)
    lower_candidates = observed_deltas[observed_deltas <= target_delta]
    upper_candidates = observed_deltas[observed_deltas >= target_delta]
    if len(lower_candidates) == 0 or len(upper_candidates) == 0:
        raise ValueError(f"Target delta {target_delta} is outside observed support")

    lower_delta = float(lower_candidates.max())
    upper_delta = float(upper_candidates.min())
    if np.isclose(lower_delta, upper_delta):
        return pivot[lower_delta].astype(float)

    weight = (target_delta - lower_delta) / (upper_delta - lower_delta)
    interpolated = pivot[lower_delta] + weight * (
        pivot[upper_delta] - pivot[lower_delta]
    )
    return interpolated.astype(float)


def build_surface_features(options: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the project's seven daily volatility-surface features.

    Parameters
    ----------
    options : pandas.DataFrame
        Portable option-chain rows for one symbol.

    Returns
    -------
    pandas.DataFrame
        Daily rows indexed by trade date. Columns are near- and mid-term ATM
        implied volatility, skew, butterfly, and the ATM term slope.

    Notes
    -----
    The tracked demo data contains one 30-day and one 90-day expiry per date.
    ATM implied volatility uses the -0.50 put delta. Wing values use linear
    interpolation at -0.25 put delta and +0.25 call delta.
    """
    atm_put = _interpolate_wing(options, option_type="p", target_delta=-0.50)
    put_wing = _interpolate_wing(options, option_type="p", target_delta=-0.25)
    call_wing = _interpolate_wing(options, option_type="c", target_delta=0.25)
    skew = put_wing - call_wing
    butterfly = 0.5 * (put_wing + call_wing) - atm_put

    by_expiry = pd.DataFrame(
        {"atm_iv": atm_put, "skew": skew, "butterfly": butterfly}
    ).reset_index()
    near = by_expiry.loc[by_expiry["dte"] == 30].set_index("trade_date")
    mid = by_expiry.loc[by_expiry["dte"] == 90].set_index("trade_date")

    features = pd.DataFrame(index=near.index)
    for column in ("atm_iv", "skew", "butterfly"):
        features[f"{column}_near"] = near[column]
        features[f"{column}_mid"] = mid[column].reindex(features.index)
    features["term_slope"] = features["atm_iv_mid"] - features["atm_iv_near"]
    return features.dropna().sort_index()


def build_forward_realized_volatility(prices: pd.DataFrame) -> pd.Series:
    """Compute annualized realized volatility over the next 20 trading days.

    Parameters
    ----------
    prices : pandas.DataFrame
        Daily prices with ``date`` and ``close`` columns.

    Returns
    -------
    pandas.Series
        Forward annualized realized volatility indexed by the date before the
        first return in each 20-day window.
    """
    close = prices.sort_values("date").set_index("date")["close"].astype(float)
    log_return = np.log(close / close.shift(1))
    trailing_volatility = log_return.rolling(FORWARD_HORIZON).std()
    forward_volatility = trailing_volatility.shift(-FORWARD_HORIZON)
    return forward_volatility * np.sqrt(float(ANNUALIZATION_FACTOR))


def fit_descriptive_regimes(features: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Fit and order a full-sample Gaussian mixture model.

    Parameters
    ----------
    features : pandas.DataFrame
        Complete daily seven-feature matrix.

    Returns
    -------
    tuple[numpy.ndarray, pandas.DataFrame]
        Ordered integer regime labels and one Bayesian information criterion
        row for every candidate component count.

    Notes
    -----
    Models use full covariance matrices, five initializations, and seed 42,
    matching ``src/volatility_regimes/regimes/latent_models.py``. Labels are
    reordered by mean near-term ATM implied volatility from low to high.
    """
    standardized = StandardScaler().fit_transform(features.to_numpy(dtype=float))
    models: dict[int, GaussianMixture] = {}
    rows: list[dict[str, float | int]] = []
    for count in CANDIDATE_REGIME_COUNTS:
        model = GaussianMixture(
            n_components=count,
            covariance_type="full",
            n_init=GMM_INITIALIZATIONS,
            random_state=RANDOM_SEED,
        ).fit(standardized)
        models[count] = model
        rows.append({"candidate_k": count, "bic": float(model.bic(standardized))})

    model_selection = pd.DataFrame(rows)
    selected_k = int(
        model_selection.loc[model_selection["bic"].idxmin(), "candidate_k"]
    )
    raw_labels = models[selected_k].predict(standardized)
    atm_column = int(features.columns.get_loc("atm_iv_near"))
    mean_atm = {
        label: float(standardized[raw_labels == label, atm_column].mean())
        for label in np.unique(raw_labels)
    }
    mapping = {old: new for new, old in enumerate(sorted(mean_atm, key=mean_atm.get))}
    ordered_labels = np.asarray(
        [mapping[int(label)] for label in raw_labels], dtype=int
    )
    model_selection["selected"] = model_selection["candidate_k"] == selected_k
    return ordered_labels, model_selection


def summarize_symbol(
    symbol: str,
    features: pd.DataFrame,
    forward_realized_volatility: pd.Series,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Summarize descriptive volatility levels by ordered regime.

    Parameters
    ----------
    symbol : str
        Underlying identifier.
    features : pandas.DataFrame
        Daily surface features.
    forward_realized_volatility : pandas.Series
        Annualized 20-day forward realized volatility.
    labels : numpy.ndarray
        Full-sample ordered regime label for every feature row.

    Returns
    -------
    pandas.DataFrame
        Regime counts, sample shares, mean ATM implied volatility, mean forward
        realized volatility, and their mean difference.
    """
    panel = features.loc[:, ["atm_iv_near"]].copy()
    panel["forward_realized_vol"] = forward_realized_volatility.reindex(panel.index)
    panel["regime"] = labels
    grouped = panel.groupby("regime", sort=True)
    summary = grouped.agg(
        n_days=("atm_iv_near", "size"),
        mean_atm_iv=("atm_iv_near", "mean"),
        mean_forward_realized_vol=("forward_realized_vol", "mean"),
    ).reset_index()
    summary["pct_days"] = summary["n_days"] / len(panel)
    summary["mean_variance_risk_premium"] = (
        summary["mean_atm_iv"] - summary["mean_forward_realized_vol"]
    )
    summary.insert(0, "symbol", symbol)
    return summary


def plot_spx_timeline(panel: pd.DataFrame, output_path: Path) -> None:
    """Plot near-term SPX ATM implied volatility with regime-coloured points.

    Parameters
    ----------
    panel : pandas.DataFrame
        SPX feature panel with ``atm_iv_near`` and ``regime`` columns.
    output_path : pathlib.Path
        Destination PNG path.

    Returns
    -------
    None
        The function writes one PNG file.
    """
    figure, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    axis.plot(panel.index, 100.0 * panel["atm_iv_near"], color="#79828d", linewidth=0.7)
    for regime in sorted(panel["regime"].unique()):
        mask = panel["regime"] == regime
        axis.scatter(
            panel.index[mask],
            100.0 * panel.loc[mask, "atm_iv_near"],
            s=7,
            alpha=0.75,
            color=REGIME_COLORS[int(regime)],
            label=f"Regime {int(regime)}",
        )
    axis.set_title("SPX near-term ATM implied volatility and descriptive regimes")
    axis.set_xlabel("Trade date")
    axis.set_ylabel("Annualized implied volatility (%)")
    axis.grid(alpha=0.2)
    axis.legend(ncol=3, frameon=False)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_oos_rmse(metric_summary: pd.DataFrame, output_path: Path) -> None:
    """Plot out-of-sample root mean squared error by model and symbol.

    Parameters
    ----------
    metric_summary : pandas.DataFrame
        Production walk-forward metrics with one row per symbol and model.
    output_path : pathlib.Path
        Destination PNG path.

    Returns
    -------
    None
        The function writes one PNG file.
    """
    model_order = (
        "historical_mean",
        "gmm_regime_mean",
        "linear_features",
        "trailing_realized_vol",
        "atm_iv",
    )
    model_labels = (
        "Historical\nmean",
        "GMM regime\nmean",
        "Linear\nfeatures",
        "Trailing\nrealized vol",
        "Current\nATM IV",
    )
    bar_colors = ("#34495e", "#2a6fbb", "#22a7a7", "#e2b04a", "#dc6b45")

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.5),
        constrained_layout=True,
        sharey=True,
    )
    for axis, symbol in zip(axes, SYMBOLS, strict=True):
        symbol_metrics = metric_summary.loc[
            metric_summary["symbol"] == symbol
        ].set_index("model_name")
        rmse_percent = 100.0 * symbol_metrics.loc[list(model_order), "rmse"]
        bars = axis.bar(model_labels, rmse_percent, color=bar_colors)
        for bar, value in zip(bars, rmse_percent, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.04,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axis.set_title(f"{symbol}: 20-day horizon")
        axis.set_xlabel("Forecast model")
        axis.grid(alpha=0.2, axis="y")
    axes[0].set_ylabel("Out-of-sample RMSE (volatility percentage points)")
    figure.suptitle("The historical mean is the lowest-RMSE forecast")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_cumulative_loss_difference(
    forecast_panel: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot cumulative GMM squared-error loss minus historical-mean loss.

    Parameters
    ----------
    forecast_panel : pandas.DataFrame
        Production row-level forecasts for all symbols and models. Volatility
        values are annualized decimals.
    output_path : pathlib.Path
        Destination PNG path.

    Returns
    -------
    None
        The function writes one PNG file.
    """
    figure, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    for symbol, color in zip(SYMBOLS, ("#2a6fbb", "#dc6b45"), strict=True):
        symbol_panel = forecast_panel.loc[
            (forecast_panel["symbol"] == symbol)
            & forecast_panel["model_name"].isin(["gmm_regime_mean", "historical_mean"])
        ]
        predictions = symbol_panel.pivot(
            index="date",
            columns="model_name",
            values="prediction",
        )
        actual = symbol_panel.groupby("date", sort=True)["actual"].first()
        gmm_error_points = 100.0 * (actual - predictions["gmm_regime_mean"])
        mean_error_points = 100.0 * (actual - predictions["historical_mean"])
        cumulative_difference = (
            gmm_error_points.pow(2) - mean_error_points.pow(2)
        ).cumsum()
        axis.plot(
            cumulative_difference.index,
            cumulative_difference,
            color=color,
            linewidth=1.5,
            label=f"{symbol} (end = {cumulative_difference.iloc[-1]:.2f})",
        )
    axis.axhline(0.0, color="#333333", linewidth=1.0)
    axis.set_title("Cumulative GMM loss minus historical-mean loss")
    axis.set_xlabel("Forecast date")
    axis.set_ylabel("Cumulative squared-error difference (percentage points²)")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    """Generate all frozen tables and figures for the bilingual post.

    Returns
    -------
    None
        Artifacts are written beneath the project-local ``blog/`` directory.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    summary_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, int | str]] = []

    for symbol in SYMBOLS:
        options = pd.read_parquet(
            PROJECT_ROOT / "data" / "raw" / f"options_{symbol.lower()}.parquet"
        )
        prices = pd.read_parquet(
            PROJECT_ROOT / "data" / "raw" / f"prices_{symbol.lower()}.parquet"
        )
        features = build_surface_features(options)
        forward_realized_volatility = build_forward_realized_volatility(prices)
        labels, model_selection = fit_descriptive_regimes(features)

        model_selection.insert(0, "symbol", symbol)
        selection_frames.append(model_selection)
        summary_frames.append(
            summarize_symbol(symbol, features, forward_realized_volatility, labels)
        )

        panel = features.loc[:, ["atm_iv_near"]].copy()
        panel["forward_realized_vol"] = forward_realized_volatility.reindex(panel.index)
        panel["regime"] = labels
        complete_target_rows = int(
            panel.dropna(subset=["forward_realized_vol"]).shape[0]
        )
        close = prices.sort_values("date").set_index("date")["close"].astype(float)
        trailing_realized_volatility = np.log(close / close.shift(1)).rolling(
            FORWARD_HORIZON
        ).std() * np.sqrt(float(ANNUALIZATION_FACTOR))
        fully_aligned_rows = int(
            panel.assign(
                trailing_realized_vol=trailing_realized_volatility.reindex(panel.index)
            )
            .dropna(subset=["forward_realized_vol", "trailing_realized_vol"])
            .shape[0]
        )
        expected_embargo_rows = FORWARD_HORIZON
        possible_test_rows = max(
            0,
            fully_aligned_rows - DEFAULT_MIN_TRAIN_SIZE - expected_embargo_rows,
        )
        audit_rows.append(
            {
                "symbol": symbol,
                "feature_rows": int(len(features)),
                "complete_20d_target_rows": complete_target_rows,
                "fully_aligned_rows_with_trailing_rv": fully_aligned_rows,
                "configured_min_train_size": DEFAULT_MIN_TRAIN_SIZE,
                "embargo_rows_before_first_test": expected_embargo_rows,
                "configured_oos_rows": possible_test_rows,
            }
        )

    summary = pd.concat(summary_frames, ignore_index=True)
    selection = pd.concat(selection_frames, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    summary.to_csv(DATA_DIR / "regime_summary.csv", index=False)
    selection.to_csv(DATA_DIR / "model_selection.csv", index=False)
    audit.to_csv(DATA_DIR / "sample_audit.csv", index=False)

    forecast_panel = pd.read_csv(
        WALKFORWARD_OUTPUT_DIR / "forecast_panel.csv",
        parse_dates=["date"],
    )
    metric_summary = pd.read_csv(WALKFORWARD_OUTPUT_DIR / "metric_summary.csv")
    forecast_panel.to_csv(
        DATA_DIR / "walkforward_forecast_panel.csv",
        index=False,
    )
    metric_summary.to_csv(
        DATA_DIR / "walkforward_metric_summary.csv",
        index=False,
    )

    spx_options = pd.read_parquet(PROJECT_ROOT / "data" / "raw" / "options_spx.parquet")
    spx_features = build_surface_features(spx_options)
    spx_labels, _ = fit_descriptive_regimes(spx_features)
    spx_panel = spx_features.loc[:, ["atm_iv_near"]].copy()
    spx_panel["regime"] = spx_labels
    plot_spx_timeline(
        spx_panel,
        IMAGE_DIR / "01_spx_regime_timeline.png",
    )
    plot_oos_rmse(metric_summary, IMAGE_DIR / "02_oos_rmse.png")
    plot_cumulative_loss_difference(
        forecast_panel,
        IMAGE_DIR / "03_cumulative_loss_difference.png",
    )

    print(summary.to_string(index=False))
    print("\nModel selection")
    print(selection.to_string(index=False))
    print("\nSample audit")
    print(audit.to_string(index=False))
    print("\nWalk-forward metrics")
    print(metric_summary.to_string(index=False))


if __name__ == "__main__":
    main()
