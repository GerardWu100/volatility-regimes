"""Regime-conditional analysis for volatility surface regime classification.

This module computes downstream economic diagnostics from fitted regime labels,
including:
- forward realized volatility,
- variance risk premium (VRP),
- regime summary statistics,
- transition persistence statistics,
- predictive ordinary least squares (OLS) regression,
- forward-return behavior by regime.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)

# Human-readable names after ordering regimes by volatility level.
REGIME_NAMES = {
    0: "Low-Vol Complacent",
    1: "Normal",
    2: "High-Skew Crash Fear",
    3: "Elevated Uncertainty",
    4: "Extreme Stress",
    5: "Regime 5",
}


def compute_realized_vol(
    prices: pd.DataFrame,
    window: int = 20,
    annualization: int = 252,
) -> pd.Series:
    """Compute forward annualized realized volatility from daily close prices.

    Definitions
    -----------
    Let `close_t` denote close price on day `t`.
    Let `r_t = ln(close_t / close_{t-1})` denote log return.
    Forward realized volatility at day `t` is:

    `rv_t = std(r_{t+1}, ..., r_{t+window}) * sqrt(annualization)`

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with columns:
        - `date`: datetime64[ns]
        - `close`: float
    window : int, default=20
        Number of trading days in the forward realized-volatility window.
    annualization : int, default=252
        Trading days per year used to annualize volatility.

    Returns
    -------
    pd.Series
        Series named `realized_vol`, indexed by date.
        Last `window` rows are NaN because forward returns are unavailable.
    """
    sorted_prices = prices.sort_values("date").set_index("date")

    close_series = sorted_prices["close"]
    log_returns = np.log(close_series / close_series.shift(1))

    rolling_std = log_returns.rolling(window=window).std()
    annualized_vol = rolling_std * np.sqrt(annualization)

    forward_realized_vol = annualized_vol.shift(-window)
    forward_realized_vol.name = "realized_vol"
    return forward_realized_vol


def compute_vrp(features: pd.DataFrame, realized_vol: pd.Series) -> pd.DataFrame:
    """Compute daily variance risk premium (VRP).

    Formula
    -------
    `vrp_t = atm_iv_near_t - realized_vol_t`

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix indexed by trade date, containing `atm_iv_near`.
    realized_vol : pd.Series
        Forward realized volatility indexed by date.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by trade date with columns:
        `atm_iv_near`, `realized_vol`, `vrp`.
    """
    vrp_df = features[["atm_iv_near"]].copy()
    vrp_df["realized_vol"] = realized_vol.reindex(vrp_df.index)
    vrp_df["vrp"] = vrp_df["atm_iv_near"] - vrp_df["realized_vol"]
    return vrp_df


def regime_summary_stats(
    features: pd.DataFrame,
    regime_labels: np.ndarray,
    n_regimes: int,
) -> pd.DataFrame:
    """Aggregate per-regime feature means, standard deviations, and frequencies.

    Parameters
    ----------
    features : pd.DataFrame
        Daily feature matrix indexed by date.
    regime_labels : np.ndarray
        Regime label array of shape `(n_days,)` aligned with `features` rows.
    n_regimes : int
        Number of modeled regimes.

    Returns
    -------
    pd.DataFrame
        One row per regime index with flattened summary columns and:
        `n_days`, `pct_days`.
    """
    summary_df = features.copy()
    summary_df["regime"] = regime_labels

    grouped = summary_df.groupby("regime").agg(["mean", "std"])
    grouped.columns = [f"{feature}_{stat}" for feature, stat in grouped.columns]

    day_counts = summary_df.groupby("regime").size().rename("n_days")
    day_share = (day_counts / len(summary_df) * 100.0).rename("pct_days")

    result = grouped.join(day_counts).join(day_share)
    return result


def regime_vrp_stats(vrp_df: pd.DataFrame, regime_labels: np.ndarray) -> pd.DataFrame:
    """Compute VRP summary statistics by regime with 95% confidence intervals.

    Parameters
    ----------
    vrp_df : pd.DataFrame
        Output of `compute_vrp`, containing `vrp` column.
    regime_labels : np.ndarray
        Regime labels aligned to full `vrp_df` index.

    Returns
    -------
    pd.DataFrame
        Indexed by regime with columns:
        `mean_vrp`, `std_vrp`, `count`, `ci_lower`, `ci_upper`.
    """
    clean_vrp = vrp_df[["vrp"]].copy()
    valid_mask = clean_vrp["vrp"].notna()

    filtered = clean_vrp.loc[valid_mask].copy()
    filtered_labels = regime_labels[valid_mask.to_numpy()]
    filtered["regime"] = filtered_labels

    stats = filtered.groupby("regime")["vrp"].agg(["mean", "std", "count"])
    stats.columns = ["mean_vrp", "std_vrp", "count"]

    standard_error = stats["std_vrp"] / np.sqrt(stats["count"])
    confidence_radius = 1.96 * standard_error
    stats["ci_lower"] = stats["mean_vrp"] - confidence_radius
    stats["ci_upper"] = stats["mean_vrp"] + confidence_radius

    return stats


def regime_transition_stats(
    regime_labels: np.ndarray, n_regimes: int
) -> dict[str, object]:
    """Compute persistence and transition-frequency statistics for labels.

    Parameters
    ----------
    regime_labels : np.ndarray
        Regime labels indexed in chronological order.
    n_regimes : int
        Total regime count.

    Returns
    -------
    dict[str, object]
        Dictionary with keys:
        - `avg_duration`: dict[int, float]
        - `transition_count`: int
        - `transition_rate`: float
    """
    change_indices = np.where(np.diff(regime_labels) != 0)[0] + 1
    run_starts = np.concatenate([[0], change_indices])
    run_ends = np.concatenate([change_indices, [len(regime_labels)]])

    run_lengths = run_ends - run_starts
    run_regimes = regime_labels[run_starts]

    avg_duration: dict[int, float] = {}
    for regime_id in range(n_regimes):
        regime_mask = run_regimes == regime_id
        if regime_mask.any():
            avg_duration[regime_id] = float(run_lengths[regime_mask].mean())
        else:
            avg_duration[regime_id] = 0.0

    transition_count = int(len(change_indices))
    denominator = max(len(regime_labels) - 1, 1)
    transition_rate = transition_count / denominator

    return {
        "avg_duration": avg_duration,
        "transition_count": transition_count,
        "transition_rate": transition_rate,
    }


def predictive_regression(
    features: pd.DataFrame,
    realized_vol: pd.Series,
    regime_labels: np.ndarray,
    n_regimes: int,
) -> dict[str, object]:
    """Fit predictive OLS: forward realized vol ~ ATM IV + regime dummies.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix indexed by trade date with `atm_iv_near` column.
    realized_vol : pd.Series
        Forward realized volatility aligned by date index.
    regime_labels : np.ndarray
        Regime labels aligned with `features` rows.
    n_regimes : int
        Number of regimes (used for interpretation and validation context).

    Returns
    -------
    dict[str, object]
        Dictionary with keys:
        `r_squared`, `adj_r_squared`, `coefficients`, `pvalues`, `summary`.
    """
    regression_df = features[["atm_iv_near"]].copy()
    regression_df["realized_vol"] = realized_vol.reindex(regression_df.index)
    regression_df["regime"] = regime_labels
    regression_df = regression_df.dropna()

    regime_dummies = pd.get_dummies(
        regression_df["regime"], prefix="regime", dtype=float
    )
    regime_dummies = regime_dummies.iloc[:, 1:]

    design_matrix = pd.concat([regression_df[["atm_iv_near"]], regime_dummies], axis=1)
    design_matrix = sm.add_constant(design_matrix)
    target = regression_df["realized_vol"]

    model = sm.OLS(target, design_matrix).fit()
    logger.info(
        "Predictive regression R^2=%.4f adjR^2=%.4f", model.rsquared, model.rsquared_adj
    )

    return {
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "coefficients": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "summary": model.summary().as_text(),
    }


def forward_returns_by_regime(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    regime_labels: np.ndarray,
    horizon: int = 20,
) -> pd.DataFrame:
    """Compute forward return distribution summary by regime.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily close series with columns `date`, `close`.
    features : pd.DataFrame
        Feature matrix indexed by trade date.
    regime_labels : np.ndarray
        Regime labels aligned to `features` index.
    horizon : int, default=20
        Forward-return horizon in trading days.

    Returns
    -------
    pd.DataFrame
        Regime-indexed summary with columns:
        `mean_return`, `std_return`, `count`, `sharpe`.
    """
    price_series = prices.set_index("date")["close"].sort_index()
    forward_returns = price_series.shift(-horizon) / price_series - 1.0

    aligned = features.iloc[:, 0:0].copy()
    aligned["fwd_return"] = forward_returns.reindex(aligned.index)
    aligned["regime"] = regime_labels
    aligned = aligned.dropna()

    grouped = aligned.groupby("regime")["fwd_return"].agg(["mean", "std", "count"])
    grouped.columns = ["mean_return", "std_return", "count"]

    annualization_ratio = np.sqrt(252 / horizon)
    grouped["sharpe"] = (
        grouped["mean_return"] / grouped["std_return"] * annualization_ratio
    )

    return grouped
