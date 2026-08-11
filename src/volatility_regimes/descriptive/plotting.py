"""Plotting utilities for volatility regime analytics outputs.

All figure functions save publication-quality PNG files to
`outputs/figures/descriptive/` and return the saved path.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from volatility_regimes.project_paths import FIGURES_DESCRIPTIVE_DIR

logger = logging.getLogger(__name__)

FIGURES_DESCRIPTIVE_DIR.mkdir(parents=True, exist_ok=True)

REGIME_COLORS = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c", "#8e44ad", "#1abc9c"]
REGIME_NAMES = {
    0: "Low-Vol Complacent",
    1: "Normal",
    2: "High-Skew Crash Fear",
    3: "Elevated Uncertainty",
    4: "Extreme Stress",
}


def _shade_regime_background(
    axis: plt.Axes,
    dates: pd.DatetimeIndex,
    regime_labels: np.ndarray,
) -> None:
    """Paint semi-transparent vertical spans between consecutive trade dates."""
    for date_index in range(len(dates) - 1):
        regime_id = int(regime_labels[date_index])
        axis.axvspan(
            dates[date_index],
            dates[date_index + 1],
            alpha=0.3,
            color=REGIME_COLORS[regime_id],
            linewidth=0,
        )


def plot_bic_curve(bic_scores: dict[int, float], best_k: int) -> Path:
    """Plot BIC across candidate regime counts and highlight selected K."""
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)

    component_counts = sorted(bic_scores)
    bic_values = [bic_scores[k] for k in component_counts]

    axis.plot(
        component_counts, bic_values, "o-", color="#2c3e50", linewidth=2, markersize=8
    )
    axis.axvline(best_k, color="#e74c3c", linestyle="--", label=f"Best K={best_k}")
    axis.set_xlabel("Number of Components (K)")
    axis.set_ylabel("BIC Score")
    axis.set_title("GMM Model Selection: BIC vs Number of Regimes")
    axis.set_xticks(component_counts)
    axis.grid(True, alpha=0.3)
    axis.legend()

    output_path = FIGURES_DESCRIPTIVE_DIR / "bic_curve.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Saved %s", output_path)
    return output_path


def plot_regime_timeseries(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    regime_labels: np.ndarray,
    n_regimes: int,
    symbol: str = "SPX",
) -> Path:
    """Plot price and ATM IV with background shading by inferred regime."""
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(16, 10),
        constrained_layout=True,
        sharex=True,
    )

    dates = features.index

    price_axis = axes[0]
    price_series = prices.set_index("date")["close"].reindex(dates)
    price_axis.plot(dates, price_series, color="#2c3e50", linewidth=0.8)

    _shade_regime_background(price_axis, dates, regime_labels)

    price_axis.set_ylabel(f"{symbol} Close Price")
    price_axis.set_title(f"{symbol} Price with Volatility Regimes")
    price_axis.grid(True, alpha=0.2)

    legend_items = [
        Patch(
            facecolor=REGIME_COLORS[r],
            alpha=0.5,
            label=REGIME_NAMES.get(r, f"Regime {r}"),
        )
        for r in range(n_regimes)
    ]
    price_axis.legend(handles=legend_items, loc="upper left", fontsize=9)

    iv_axis = axes[1]
    iv_axis.plot(
        dates,
        features["atm_iv_near"],
        color="#2c3e50",
        linewidth=0.8,
        label="Near-term ATM IV",
    )
    iv_axis.plot(
        dates,
        features["atm_iv_mid"],
        color="#7f8c8d",
        linewidth=0.8,
        label="Mid-term ATM IV",
    )

    _shade_regime_background(iv_axis, dates, regime_labels)

    iv_axis.set_ylabel("Implied Volatility")
    iv_axis.set_xlabel("Date")
    iv_axis.grid(True, alpha=0.2)
    iv_axis.legend(loc="upper left", fontsize=9)
    iv_axis.xaxis.set_major_locator(mdates.YearLocator())
    iv_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    output_path = FIGURES_DESCRIPTIVE_DIR / f"regime_timeseries_{symbol.lower()}.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    logger.info("Saved %s", output_path)
    return output_path


def plot_transition_matrix(transition_matrix: np.ndarray, n_regimes: int) -> Path:
    """Plot HMM transition matrix as a heatmap with in-cell annotations."""
    fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)

    image = axis.imshow(transition_matrix, cmap="YlOrRd", vmin=0.0, vmax=1.0)

    regime_labels = [REGIME_NAMES.get(i, f"Regime {i}") for i in range(n_regimes)]
    axis.set_xticks(range(n_regimes))
    axis.set_yticks(range(n_regimes))
    axis.set_xticklabels(regime_labels, rotation=45, ha="right", fontsize=9)
    axis.set_yticklabels(regime_labels, fontsize=9)

    for i in range(n_regimes):
        for j in range(n_regimes):
            probability = transition_matrix[i, j]
            text_color = "white" if probability > 0.5 else "black"
            axis.text(
                j,
                i,
                f"{probability:.3f}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=text_color,
            )

    axis.set_xlabel("To Regime")
    axis.set_ylabel("From Regime")
    axis.set_title("HMM Regime Transition Probabilities")
    fig.colorbar(image, ax=axis, label="Probability")

    output_path = FIGURES_DESCRIPTIVE_DIR / "transition_matrix.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Saved %s", output_path)
    return output_path


def plot_feature_distributions(
    features: pd.DataFrame,
    regime_labels: np.ndarray,
    n_regimes: int,
) -> Path:
    """Plot feature-wise boxplots split by regime."""
    feature_names = features.columns.tolist()
    feature_count = len(feature_names)

    n_columns = 2
    n_rows = (feature_count + 1) // n_columns

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(14, 4 * n_rows),
        constrained_layout=True,
    )
    axes_flat = np.array(axes).reshape(-1)

    plotting_df = features.copy()
    plotting_df["regime"] = regime_labels

    for feature_index, feature_name in enumerate(feature_names):
        axis = axes_flat[feature_index]

        values_by_regime = [
            plotting_df.loc[plotting_df["regime"] == regime_id, feature_name].dropna()
            for regime_id in range(n_regimes)
        ]

        regime_tick_labels = [
            REGIME_NAMES.get(regime_id, f"R{regime_id}")
            for regime_id in range(n_regimes)
        ]
        boxplot = axis.boxplot(
            values_by_regime,
            tick_labels=regime_tick_labels,
            patch_artist=True,
        )

        for regime_id, box in enumerate(boxplot["boxes"]):
            box.set_facecolor(REGIME_COLORS[regime_id])
            box.set_alpha(0.6)

        axis.set_title(feature_name, fontsize=10)
        axis.grid(True, alpha=0.2)
        axis.tick_params(axis="x", rotation=30, labelsize=8)

    for unused_axis_index in range(feature_count, len(axes_flat)):
        axes_flat[unused_axis_index].set_visible(False)

    fig.suptitle("Feature Distributions by Regime", fontsize=14, fontweight="bold")

    output_path = FIGURES_DESCRIPTIVE_DIR / "feature_distributions.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Saved %s", output_path)
    return output_path


def plot_vrp_by_regime(vrp_stats: pd.DataFrame) -> Path:
    """Plot mean VRP per regime with 95% confidence interval error bars."""
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)

    regimes = vrp_stats.index.to_numpy()
    means = vrp_stats["mean_vrp"].to_numpy()
    lower_error = means - vrp_stats["ci_lower"].to_numpy()
    upper_error = vrp_stats["ci_upper"].to_numpy() - means
    errors = np.array([lower_error, upper_error])

    bar_colors = [REGIME_COLORS[int(regime)] for regime in regimes]
    bar_labels = [
        REGIME_NAMES.get(int(regime), f"Regime {int(regime)}") for regime in regimes
    ]

    bars = axis.bar(
        bar_labels,
        means,
        yerr=errors,
        capsize=5,
        color=bar_colors,
        alpha=0.7,
        edgecolor="black",
    )

    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("Variance Risk Premium (IV - RV)")
    axis.set_title("Variance Risk Premium by Regime (with 95% CI)")
    axis.grid(True, alpha=0.2, axis="y")

    for bar, mean in zip(bars, means):
        bar_center = bar.get_x() + bar.get_width() / 2
        bar_top = bar.get_height()
        axis.text(
            bar_center,
            bar_top + 0.002,
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    output_path = FIGURES_DESCRIPTIVE_DIR / "vrp_by_regime.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Saved %s", output_path)
    return output_path


def plot_regime_durations(regime_labels: np.ndarray, n_regimes: int) -> Path:
    """Plot histogram of consecutive-run durations by regime."""
    change_points = np.where(np.diff(regime_labels) != 0)[0] + 1
    run_starts = np.concatenate([[0], change_points])
    run_ends = np.concatenate([change_points, [len(regime_labels)]])

    run_lengths = run_ends - run_starts
    run_regimes = regime_labels[run_starts]

    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)

    for regime_id in range(n_regimes):
        regime_mask = run_regimes == regime_id
        if not regime_mask.any():
            continue

        lengths = run_lengths[regime_mask]
        mean_duration = lengths.mean()
        label = f"{REGIME_NAMES.get(regime_id, f'Regime {regime_id}')} (mean={mean_duration:.1f}d)"

        axis.hist(
            lengths,
            bins=30,
            alpha=0.5,
            color=REGIME_COLORS[regime_id],
            label=label,
        )

    axis.set_xlabel("Duration (trading days)")
    axis.set_ylabel("Frequency")
    axis.set_title("Regime Duration Distributions")
    axis.grid(True, alpha=0.2)
    axis.legend()

    output_path = FIGURES_DESCRIPTIVE_DIR / "regime_durations.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Saved %s", output_path)
    return output_path


def plot_vol_surface_snapshots(
    options: pd.DataFrame,
    regime_labels: np.ndarray,
    features: pd.DataFrame,
    n_regimes: int,
    symbol: str = "SPX",
) -> Path:
    """Plot representative near-term delta smiles for each regime."""
    fig, axes = plt.subplots(
        1, n_regimes, figsize=(5 * n_regimes, 5), constrained_layout=True
    )

    if n_regimes == 1:
        axes_list = [axes]
    else:
        axes_list = list(np.array(axes).reshape(-1))

    for regime_id in range(n_regimes):
        axis = axes_list[regime_id]

        regime_mask = regime_labels == regime_id
        regime_dates = features.index[regime_mask]

        if len(regime_dates) == 0:
            axis.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes
            )
            axis.set_title(REGIME_NAMES.get(regime_id, f"Regime {regime_id}"))
            continue

        regime_atm_iv = features.loc[regime_mask, "atm_iv_near"]
        median_atm_iv = regime_atm_iv.median()

        distances = np.abs(regime_atm_iv.to_numpy() - median_atm_iv)
        representative_idx = int(np.argmin(distances))
        representative_date = regime_dates[representative_idx]

        day_options = options.loc[options["trade_date"] == representative_date]

        near_dtes = day_options["dte"].unique()
        near_dtes = near_dtes[(near_dtes >= 15) & (near_dtes <= 45)]

        if len(near_dtes) == 0:
            axis.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes
            )
            axis.set_title(REGIME_NAMES.get(regime_id, f"Regime {regime_id}"))
            continue

        target_near_dte = near_dtes[int(np.argmin(np.abs(near_dtes - 30)))]
        expiry_slice = day_options.loc[day_options["dte"] == target_near_dte]

        puts = expiry_slice.loc[expiry_slice["option_type"] == "p"].sort_values("delta")
        calls = expiry_slice.loc[expiry_slice["option_type"] == "c"].sort_values(
            "delta"
        )

        axis.plot(
            puts["delta"],
            puts["mid_iv"],
            "o-",
            color="#e74c3c",
            markersize=2,
            linewidth=1,
            label="Puts",
        )
        axis.plot(
            calls["delta"],
            calls["mid_iv"],
            "o-",
            color="#2ecc71",
            markersize=2,
            linewidth=1,
            label="Calls",
        )

        regime_name = REGIME_NAMES.get(regime_id, f"Regime {regime_id}")
        axis.set_title(
            f"{regime_name}\n{representative_date.strftime('%Y-%m-%d')}", fontsize=10
        )
        axis.set_xlabel("Delta")
        axis.set_ylabel("Implied Vol")
        axis.grid(True, alpha=0.2)
        axis.legend(fontsize=8)

    fig.suptitle(
        f"{symbol} IV Smile by Regime (Representative Days)",
        fontsize=13,
        fontweight="bold",
    )

    output_path = (
        FIGURES_DESCRIPTIVE_DIR / f"vol_surface_snapshots_{symbol.lower()}.png"
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Saved %s", output_path)
    return output_path
