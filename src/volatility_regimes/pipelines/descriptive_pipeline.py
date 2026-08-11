"""Offline-first descriptive volatility-regime pipeline.

This module orchestrates one full-sample descriptive workflow:
1. Load options and underlying index prices.
2. Extract daily volatility-surface features.
3. Fit Gaussian Mixture Model and Hidden Markov Model regimes.
4. Compute regime-conditional analytics and economic diagnostics.
5. Generate charts and save output tables under `outputs/reports/descriptive/`
   and `outputs/figures/descriptive/`.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
import tomllib

from volatility_regimes.data_access.loader import load_daily_prices, load_options
from volatility_regimes.descriptive.analytics import (
    REGIME_NAMES,
    compute_realized_vol,
    compute_vrp,
    forward_returns_by_regime,
    predictive_regression,
    regime_summary_stats,
    regime_transition_stats,
    regime_vrp_stats,
)
from volatility_regimes.descriptive.plotting import (
    plot_bic_curve,
    plot_feature_distributions,
    plot_regime_durations,
    plot_regime_timeseries,
    plot_transition_matrix,
    plot_vol_surface_snapshots,
    plot_vrp_by_regime,
)
from volatility_regimes.features.surface import extract_features
from volatility_regimes.project_paths import PROJECT_ROOT, REPORTS_DESCRIPTIVE_DIR
from volatility_regimes.regimes.latent_models import (
    fit_gmm,
    fit_hmm,
    order_regimes_by_volatility,
    standardize_features,
)

REPORTS_DESCRIPTIVE_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    """Configure application logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def load_config() -> dict[str, object]:
    """Load `config.toml` from project root."""
    config_path = PROJECT_ROOT / "config.toml"
    with config_path.open("rb") as file_handle:
        config: dict[str, object] = tomllib.load(file_handle)
    return config


def _recompute_transition_matrix_from_labels(
    labels: np.ndarray, n_states: int
) -> np.ndarray:
    """Recompute transition probabilities directly from a relabeled state path.

    Parameters
    ----------
    labels : np.ndarray
        Integer regime labels ordered in time.
    n_states : int
        Number of unique states.

    Returns
    -------
    np.ndarray
        Transition matrix where row `i` sums to one when state `i` occurs.
    """
    if len(labels) < 2:
        return np.zeros((n_states, n_states), dtype=float)

    # Count one-step transitions, then normalize each origin state row.
    from_states = labels[:-1].astype(int)
    to_states = labels[1:].astype(int)
    transition_counts = pd.crosstab(from_states, to_states)
    transition_counts = transition_counts.reindex(
        index=range(n_states),
        columns=range(n_states),
        fill_value=0,
    )
    row_totals = transition_counts.sum(axis=1)
    transition_probabilities = transition_counts.div(
        row_totals.where(row_totals > 0),
        axis=0,
    ).fillna(0.0)
    return transition_probabilities.to_numpy(dtype=float)


def run_pipeline_for_symbol(
    symbol: str, config: dict[str, object]
) -> dict[str, object]:
    """Execute full regime pipeline for one symbol.

    Parameters
    ----------
    symbol : str
        Underlying symbol, for example `SPX`.
    config : dict[str, object]
        Parsed TOML configuration dictionary.

    Returns
    -------
    dict[str, object]
        In-memory outputs used for per-symbol and cross-asset summaries.
    """
    logger = logging.getLogger(symbol)

    data_config = config["data"]
    cache_config = config["cache"]
    feature_config = config["features"]
    regime_config = config["regime"]
    analysis_config = config["analysis"]

    assert isinstance(data_config, dict)
    assert isinstance(cache_config, dict)
    assert isinstance(feature_config, dict)
    assert isinstance(regime_config, dict)
    assert isinstance(analysis_config, dict)

    logger.info("Loading options data")
    options = load_options(
        symbol=symbol,
        start_date=str(data_config["start_date"]),
        end_date=str(data_config["end_date"]),
        delta_min=float(feature_config["delta_min"]),
        delta_max=float(feature_config["delta_max"]),
        cache_config=cache_config,
    )

    logger.info("Loading daily prices")
    prices = load_daily_prices(
        symbol=symbol,
        start_date=str(data_config["start_date"]),
        end_date=str(data_config["end_date"]),
        cache_config=cache_config,
    )

    logger.info("Extracting volatility surface features")
    features = extract_features(
        options=options,
        near_dte_min=int(feature_config["near_term_dte_min"]),
        near_dte_target=int(feature_config["near_term_dte_target"]),
        near_dte_max=int(feature_config["near_term_dte_max"]),
        mid_dte_min=int(feature_config["mid_term_dte_min"]),
        mid_dte_target=int(feature_config["mid_term_dte_target"]),
        mid_dte_max=int(feature_config["mid_term_dte_max"]),
        atm_delta=float(feature_config["atm_delta"]),
        wing_delta=float(feature_config["wing_delta"]),
        min_strikes=int(feature_config["min_strikes_per_side"]),
    )

    features_clean = features.dropna()
    dropped_count = len(features) - len(features_clean)
    logger.info(
        "Feature rows retained=%s dropped=%s", len(features_clean), dropped_count
    )

    feature_matrix = features_clean.to_numpy()
    standardized_matrix, _ = standardize_features(feature_matrix)

    logger.info("Fitting GMM")
    gmm_labels, bic_scores, best_k, _ = fit_gmm(
        standardized_matrix,
        min_k=int(regime_config["min_k"]),
        max_k=int(regime_config["max_k"]),
    )

    logger.info("Fitting HMM with %s states", best_k)
    hmm_labels_raw, _, _ = fit_hmm(
        standardized_matrix,
        n_states=best_k,
        n_iter=int(regime_config["hmm_n_iter"]),
        n_restarts=int(regime_config["hmm_random_restarts"]),
    )

    # Look the ATM IV column up by name. Ordering regimes by whatever happens to
    # sit in column 0 would silently rank them on the wrong feature if the
    # feature matrix ever changes order.
    atm_iv_column_index = int(features_clean.columns.get_loc("atm_iv_near"))
    hmm_labels = order_regimes_by_volatility(
        hmm_labels_raw, standardized_matrix, atm_iv_col_idx=atm_iv_column_index
    )
    gmm_labels = order_regimes_by_volatility(
        gmm_labels, standardized_matrix, atm_iv_col_idx=atm_iv_column_index
    )
    transition_matrix = _recompute_transition_matrix_from_labels(
        hmm_labels, n_states=best_k
    )

    logger.info("Computing analysis outputs")
    realized_vol = compute_realized_vol(
        prices,
        window=int(analysis_config["realized_vol_window"]),
        annualization=int(analysis_config["annualization_factor"]),
    )

    vrp_df = compute_vrp(features_clean, realized_vol)
    summary_stats = regime_summary_stats(features_clean, hmm_labels, best_k)
    vrp_stats = regime_vrp_stats(vrp_df, hmm_labels)
    transition_stats = regime_transition_stats(hmm_labels, best_k)
    forward_return_stats = forward_returns_by_regime(
        prices,
        features_clean,
        hmm_labels,
        horizon=int(analysis_config["realized_vol_window"]),
        annualization=int(analysis_config["annualization_factor"]),
    )
    regression_results = predictive_regression(
        features_clean, realized_vol, hmm_labels, best_k
    )

    agreement = float((gmm_labels == hmm_labels).mean())
    logger.info("GMM vs HMM label agreement=%.1f%%", agreement * 100.0)

    logger.info("Generating plots")
    plot_bic_curve(bic_scores, best_k)
    plot_regime_timeseries(features_clean, prices, hmm_labels, best_k, symbol=symbol)
    plot_transition_matrix(transition_matrix, best_k)
    plot_feature_distributions(features_clean, hmm_labels, best_k)
    plot_vrp_by_regime(vrp_stats)
    plot_regime_durations(hmm_labels, best_k)
    plot_vol_surface_snapshots(
        options, hmm_labels, features_clean, best_k, symbol=symbol
    )

    summary_stats.to_csv(
        REPORTS_DESCRIPTIVE_DIR / f"regime_summary_{symbol.lower()}.csv"
    )
    vrp_stats.to_csv(REPORTS_DESCRIPTIVE_DIR / f"vrp_stats_{symbol.lower()}.csv")
    forward_return_stats.to_csv(
        REPORTS_DESCRIPTIVE_DIR / f"forward_returns_{symbol.lower()}.csv"
    )

    regression_path = (
        REPORTS_DESCRIPTIVE_DIR / f"regression_summary_{symbol.lower()}.txt"
    )
    regression_path.write_text(str(regression_results["summary"]))
    logger.info("Saved tabular outputs to %s", REPORTS_DESCRIPTIVE_DIR)

    print("\n" + "=" * 60)
    print(f"  {symbol} REGIME ANALYSIS RESULTS")
    print("=" * 60)
    print(f"\nRegimes detected: {best_k} (BIC-selected)")
    print(f"GMM vs HMM agreement: {agreement:.1%}")

    print("\nRegime Summary:")
    for regime_id in range(best_k):
        regime_name = REGIME_NAMES.get(regime_id, f"Regime {regime_id}")
        n_days = int(summary_stats.loc[regime_id, "n_days"])
        pct_days = float(summary_stats.loc[regime_id, "pct_days"])
        avg_duration = float(transition_stats["avg_duration"][regime_id])
        print(
            f"  {regime_name}: {n_days} days ({pct_days:.1f}%), avg duration {avg_duration:.1f} days"
        )

    print("\nVariance Risk Premium by Regime:")
    for regime_id in vrp_stats.index:
        regime_name = REGIME_NAMES.get(int(regime_id), f"Regime {int(regime_id)}")
        mean_vrp = float(vrp_stats.loc[regime_id, "mean_vrp"])
        print(f"  {regime_name}: {mean_vrp:.4f} ({mean_vrp * 100.0:.2f}%)")

    print(f"\nPredictive Regression R^2: {float(regression_results['r_squared']):.4f}")
    print(
        "Transition rate: "
        f"{float(transition_stats['transition_rate']):.3f} "
        f"({int(transition_stats['transition_count'])} transitions)"
    )

    print("\n20-day Forward Returns by Regime:")
    for regime_id in forward_return_stats.index:
        regime_name = REGIME_NAMES.get(int(regime_id), f"Regime {int(regime_id)}")
        mean_return = float(forward_return_stats.loc[regime_id, "mean_return"])
        sharpe_ratio = float(forward_return_stats.loc[regime_id, "sharpe"])
        print(
            f"  {regime_name}: mean={mean_return * 100.0:.2f}%, Sharpe={sharpe_ratio:.2f}"
        )

    return {
        "features": features_clean,
        "options": options,
        "prices": prices,
        "gmm_labels": gmm_labels,
        "hmm_labels": hmm_labels,
        "bic_scores": bic_scores,
        "best_k": best_k,
        "transition_matrix": transition_matrix,
        "vrp_df": vrp_df,
        "summary_stats": summary_stats,
        "vrp_stats": vrp_stats,
        "transition_stats": transition_stats,
        "regression_results": regression_results,
        "forward_returns": forward_return_stats,
    }


def _cross_asset_comparison(results: dict[str, dict[str, object]]) -> None:
    """Print simple cross-asset regime alignment for first two symbols."""
    symbols = list(results)
    if len(symbols) < 2:
        return

    first_symbol = symbols[0]
    second_symbol = symbols[1]

    first_features = results[first_symbol]["features"]
    second_features = results[second_symbol]["features"]
    first_labels = results[first_symbol]["hmm_labels"]
    second_labels = results[second_symbol]["hmm_labels"]

    assert isinstance(first_features, pd.DataFrame)
    assert isinstance(second_features, pd.DataFrame)
    assert isinstance(first_labels, np.ndarray)
    assert isinstance(second_labels, np.ndarray)

    common_dates = first_features.index.intersection(second_features.index)

    first_mask = first_features.index.isin(common_dates)
    second_mask = second_features.index.isin(common_dates)

    aligned_first_labels = first_labels[first_mask]
    aligned_second_labels = second_labels[second_mask]

    alignment = float((aligned_first_labels == aligned_second_labels).mean())

    print("\n" + "=" * 60)
    print(f"  CROSS-ASSET COMPARISON: {first_symbol} vs {second_symbol}")
    print("=" * 60)
    print(f"Common trading days: {len(common_dates)}")
    print(f"Regime agreement: {alignment:.1%}")


def main() -> None:
    """Run pipeline for all configured symbols."""
    setup_logging()
    logger = logging.getLogger("main")

    config = load_config()
    symbols = config["data"]["symbols"]

    assert isinstance(symbols, list)

    all_results: dict[str, dict[str, object]] = {}
    for symbol in symbols:
        logger.info("\n%s", "=" * 60)
        logger.info("Processing %s", symbol)
        logger.info("%s", "=" * 60)
        all_results[str(symbol)] = run_pipeline_for_symbol(str(symbol), config)

    _cross_asset_comparison(all_results)
    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()
