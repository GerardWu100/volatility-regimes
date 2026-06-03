"""CLI helpers for walk-forward volatility-regime research.

This module runs the Task 4 experiment grid: benchmark forecasts, linear
feature forecasts, and GMM regime-mean forecasts across symbols, feature
subsets, and forecast horizons.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import tomllib
from typing import Callable

import numpy as np
import pandas as pd

from volatility_regimes.data_access.loader import load_daily_prices, load_options
from volatility_regimes.features.surface import extract_features, select_feature_columns
from volatility_regimes.walkforward.models import (
    forecast_atm_iv,
    forecast_linear_features,
    forecast_regime_mean,
    forecast_trailing_realized_vol,
)
from volatility_regimes.walkforward.reporting import (
    summarize_metrics,
    write_research_summary,
)
from volatility_regimes.walkforward.splits import build_expanding_window_splits
from volatility_regimes.walkforward.targets import build_forward_targets


from volatility_regimes.project_paths import PROJECT_ROOT, REPORTS_WALKFORWARD_DIR

OUTPUT_DIR = REPORTS_WALKFORWARD_DIR


def _load_walkforward_config(project_root: Path) -> dict[str, object]:
    """Load the experiment-owned walk-forward config.

    Parameters
    ----------
    project_root : Path
        Repository root that contains `walkforward.toml`.

    Returns
    -------
    dict[str, object]
        Parsed experiment config dictionary.
    """
    config_path = project_root / "walkforward.toml"
    with config_path.open("rb") as file_handle:
        config = tomllib.load(file_handle)
    return config


def _load_project_config(project_root: Path) -> dict[str, object]:
    """Load the shared root-level project config.

    Parameters
    ----------
    project_root : Path
        Repository root that contains `config.toml`.

    Returns
    -------
    dict[str, object]
        Parsed root project config dictionary.
    """
    config_path = project_root / "config.toml"
    with config_path.open("rb") as file_handle:
        config = tomllib.load(file_handle)
    return config


def _project_config_with_sample_window(
    project_config: dict[str, object],
    walkforward_config: dict[str, object],
) -> dict[str, object]:
    """Override root config dates with the walk-forward sample window.

    Parameters
    ----------
    project_config : dict[str, object]
        Parsed root-level config used for shared data, cache, and features.
    walkforward_config : dict[str, object]
        Parsed walk-forward config that owns the experiment sample window.

    Returns
    -------
    dict[str, object]
        Project config copy with `data.start_date` and `data.end_date`
        replaced by `sample.start_date` and `sample.end_date`.
    """
    sample_config = walkforward_config["sample"]
    merged_project_config = dict(project_config)
    merged_data_config = dict(project_config["data"])
    merged_data_config["start_date"] = str(sample_config["start_date"])
    merged_data_config["end_date"] = str(sample_config["end_date"])
    merged_project_config["data"] = merged_data_config
    return merged_project_config


def _resolve_output_dir(
    project_root: Path,
    walkforward_config: dict[str, object],
) -> Path:
    """Resolve the output directory from the walk-forward config.

    Parameters
    ----------
    project_root : Path
        Repository root for the current run.
    walkforward_config : dict[str, object]
        Parsed walk-forward experiment config.

    Returns
    -------
    Path
        Output directory resolved relative to the repository root.
    """
    output_config = walkforward_config["output"]
    relative_output_dir = Path(str(output_config["output_dir"]))
    output_dir = project_root / relative_output_dir
    return output_dir


def _load_symbol_inputs(
    symbol: str,
    project_config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load features and daily prices for one symbol using root config values.

    Parameters
    ----------
    symbol : str
        Underlying symbol, for example `SPX` or `NDX`.
    project_config : dict[str, object]
        Parsed root-level `config.toml` settings.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple of `(features, prices)` where features are indexed by
        `trade_date` and prices contain `date` and `close`.
    """
    data_config = project_config["data"]
    cache_config = project_config["cache"]
    feature_config = project_config["features"]

    options = load_options(
        symbol=symbol,
        start_date=str(data_config["start_date"]),
        end_date=str(data_config["end_date"]),
        delta_min=float(feature_config["delta_min"]),
        delta_max=float(feature_config["delta_max"]),
        cache_config=cache_config,
    )
    prices = load_daily_prices(
        symbol=symbol,
        start_date=str(data_config["start_date"]),
        end_date=str(data_config["end_date"]),
        cache_config=cache_config,
    )

    # Extract the reusable surface features once so each horizon reuses the
    # same daily research panel.
    features = extract_features(
        options=options,
        near_dte_min=int(feature_config["near_term_dte_min"]),
        near_dte_target=int(feature_config["near_term_dte_target"]),
        near_dte_max=int(feature_config["near_term_dte_max"]),
        mid_dte_min=int(feature_config["mid_term_dte_min"]),
        mid_dte_target=int(feature_config["mid_term_dte_target"]),
        mid_dte_max=int(feature_config["mid_term_dte_max"]),
        wing_delta=float(feature_config["wing_delta"]),
        min_strikes=int(feature_config["min_strikes_per_side"]),
    ).dropna()
    return features, prices


def _compute_trailing_realized_volatility(
    prices: pd.DataFrame,
    annualization: int,
    horizon: int,
    evaluation_index: pd.DatetimeIndex,
) -> pd.Series:
    """Compute a trailing realized-volatility benchmark on the test index.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily price table with columns `date` and `close`.
    annualization : int
        Trading-day annualization factor.
    horizon : int
        Trailing rolling window length in trading days.
    evaluation_index : pd.DatetimeIndex
        Dates that the trailing series should be aligned to.

    Returns
    -------
    pd.Series
        Trailing annualized realized volatility aligned to `evaluation_index`.
    """
    sorted_prices = prices.loc[:, ["date", "close"]].sort_values("date")
    close_series = sorted_prices.set_index("date")["close"].astype(float)

    # trailing_rv_t = std(r_{t-h+1}, ..., r_t) * sqrt(annualization)
    log_returns = np.log(close_series / close_series.shift(1))
    trailing_realized_volatility = log_returns.rolling(window=horizon).std()
    trailing_realized_volatility = trailing_realized_volatility * np.sqrt(
        float(annualization)
    )
    trailing_realized_volatility = trailing_realized_volatility.reindex(
        evaluation_index
    )
    return trailing_realized_volatility


def _build_price_date_positions(prices: pd.DataFrame) -> dict[pd.Timestamp, int]:
    """Map each price date to its chronological position.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily price table with columns `date` and `close`.

    Returns
    -------
    dict[pd.Timestamp, int]
        Mapping from price date to its zero-based position in sorted order.
    """
    sorted_prices = prices.loc[:, ["date"]].sort_values("date")
    unique_dates = pd.DatetimeIndex(sorted_prices["date"].drop_duplicates())
    date_positions = {
        pd.Timestamp(date): position for position, date in enumerate(unique_dates)
    }
    return date_positions


def _apply_forward_target_embargo(
    train_index: pd.DatetimeIndex,
    test_index: pd.DatetimeIndex,
    horizon: int,
    price_date_positions: dict[pd.Timestamp, int],
) -> pd.DatetimeIndex:
    """Drop train rows whose forward target window overlaps the test window.

    Definitions
    -----------
    Let `target_t` use returns from `t+1` through `t+h`, where `h` is the
    forecast horizon. If the test window starts at date `s`, then any train
    row `t` with `position(t) + h >= position(s)` leaks test-window returns
    into the training labels and must be removed.

    Parameters
    ----------
    train_index : pd.DatetimeIndex
        Candidate training dates from the walk-forward split.
    test_index : pd.DatetimeIndex
        Test dates from the walk-forward split.
    horizon : int
        Forward target horizon in trading days.
    price_date_positions : dict[pd.Timestamp, int]
        Position lookup built from the sorted price history.

    Returns
    -------
    pd.DatetimeIndex
        Leakage-safe training dates after the forward-target embargo.
    """
    test_start_date = pd.Timestamp(test_index[0])
    test_start_position = price_date_positions[test_start_date]
    safe_train_dates: list[pd.Timestamp] = []

    for train_date in train_index:
        normalized_train_date = pd.Timestamp(train_date)
        train_position = price_date_positions[normalized_train_date]
        forward_window_end_position = train_position + horizon

        # Keep only labels whose final forward return lands strictly before
        # the first test date.
        if forward_window_end_position < test_start_position:
            safe_train_dates.append(normalized_train_date)

    safe_train_index = pd.DatetimeIndex(safe_train_dates)
    return safe_train_index


def _build_hmm_forecast_kwargs(
    hmm_n_iter: int,
    hmm_random_restarts: int,
) -> dict[str, object]:
    """Build optional HMM kwargs accepted by the forecast helper.

    Parameters
    ----------
    hmm_n_iter : int
        Maximum expectation-maximization iterations for HMM fitting.
    hmm_random_restarts : int
        Number of random HMM restarts.

    Returns
    -------
    dict[str, object]
        Keyword arguments that can be passed to `forecast_regime_mean`
        without breaking tests that monkeypatch a simplified helper.
    """
    forecast_parameters = inspect.signature(forecast_regime_mean).parameters
    hmm_forecast_kwargs: dict[str, object] = {}

    # Some tests monkeypatch `forecast_regime_mean` with a shorter signature,
    # so add HMM kwargs only when the active function supports them.
    if "hmm_n_iter" in forecast_parameters:
        hmm_forecast_kwargs["hmm_n_iter"] = hmm_n_iter
    if "hmm_random_restarts" in forecast_parameters:
        hmm_forecast_kwargs["hmm_random_restarts"] = hmm_random_restarts
    return hmm_forecast_kwargs


def _append_forecast_row(
    forecast_rows: list[dict[str, object]],
    *,
    symbol: str,
    feature_set: str,
    horizon: int,
    forecast_date: pd.Timestamp,
    model_name: str,
    prediction: float,
    actual: float,
    selected_k: float = float(np.nan),
    predicted_regime: float = float(np.nan),
) -> None:
    """Append one standardized forecast row to the in-memory panel."""
    forecast_rows.append(
        _build_forecast_row(
            symbol=symbol,
            feature_set=feature_set,
            horizon=horizon,
            forecast_date=forecast_date,
            model_name=model_name,
            prediction=prediction,
            actual=actual,
            selected_k=selected_k,
            predicted_regime=predicted_regime,
        )
    )


def _append_regime_mean_row(
    forecast_rows: list[dict[str, object]],
    *,
    symbol: str,
    feature_set: str,
    horizon: int,
    forecast_date: pd.Timestamp,
    model_name: str,
    regime_result: dict[str, object],
    actual: float,
) -> None:
    """Append one regime-mean forecast row with model-selection metadata."""
    _append_forecast_row(
        forecast_rows,
        symbol=symbol,
        feature_set=feature_set,
        horizon=horizon,
        forecast_date=forecast_date,
        model_name=model_name,
        prediction=float(regime_result["prediction"]),
        actual=actual,
        selected_k=float(int(regime_result["selected_k"])),
        predicted_regime=float(int(regime_result["predicted_regime"])),
    )


def _build_forecast_row(
    symbol: str,
    feature_set: str,
    horizon: int,
    forecast_date: pd.Timestamp,
    model_name: str,
    prediction: float,
    actual: float,
    selected_k: float = float(np.nan),
    predicted_regime: float = float(np.nan),
) -> dict[str, object]:
    """Build one standardized forecast row for the output panel.

    Parameters
    ----------
    symbol : str
        Symbol for this forecast row.
    feature_set : str
        Feature-set key used for the forecast.
    horizon : int
        Forecast horizon in trading days.
    forecast_date : pd.Timestamp
        Date for the out-of-sample forecast.
    model_name : str
        Name of the forecast model.
    prediction : float
        Model prediction.
    actual : float
        Realized target value.
    selected_k : float, default=np.nan
        Regime count used by model-based forecasts.
    predicted_regime : float, default=np.nan
        Ordered regime label predicted by model-based forecasts.

    Returns
    -------
    dict[str, object]
        One forecast panel row with stable column names and value types.
    """
    return {
        "symbol": symbol,
        "feature_set": feature_set,
        "horizon": int(horizon),
        "date": forecast_date,
        "model_name": model_name,
        "prediction": float(prediction),
        "actual": float(actual),
        "selected_k": selected_k,
        "predicted_regime": predicted_regime,
    }


def _forecast_hmm_safely(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    test_row: pd.Series,
    regime_min_k: int,
    regime_max_k: int,
    hmm_forecast_kwargs: dict[str, object],
) -> dict[str, object] | None:
    """Return HMM regime-mean forecast, or None when fitting fails.

    Small synthetic windows can fail to fit HMM models for some `K` values,
    so the walk-forward run should skip only the failed row and continue.
    """
    try:
        return forecast_regime_mean(
            train_features=train_features,
            train_target=train_target,
            test_row=test_row,
            model_type="hmm",
            min_k=regime_min_k,
            max_k=regime_max_k,
            **hmm_forecast_kwargs,
        )
    except RuntimeError:
        return None


def run_research(
    symbols: list[str],
    feature_sets: list[str],
    horizons: list[int],
    min_train_size: int,
    step_size: int,
    annualization: int,
    regime_min_k: int,
    regime_max_k: int,
    fixed_k_values: list[int] | None = None,
    hmm_n_iter: int = 200,
    hmm_random_restarts: int = 10,
    project_config: dict[str, object] | None = None,
    symbol_input_loader: (
        Callable[[str, dict[str, object]], tuple[pd.DataFrame, pd.DataFrame]] | None
    ) = None,
) -> None:
    """Run the walk-forward research experiment grid and write outputs.

    Parameters
    ----------
    symbols : list[str]
        Symbols included in the research run.
    feature_sets : list[str]
        Feature-set names passed to `select_feature_columns`.
    horizons : list[int]
        Forward forecast horizons in trading days.
    min_train_size : int
        Minimum expanding-window train length before the first forecast.
    step_size : int
        Number of rows forecasted in each walk-forward step.
    annualization : int
        Trading-day annualization factor used in volatility targets.
    regime_min_k : int
        Minimum GMM regime count considered by model selection.
    regime_max_k : int
        Maximum GMM regime count considered by model selection.
    fixed_k_values : list[int] | None, default=None
        Fixed regime counts used for robustness sweeps. Each value adds one
        extra GMM row and one extra HMM row per forecast date.
    hmm_n_iter : int, default=200
        Maximum expectation-maximization iterations for HMM robustness runs.
    hmm_random_restarts : int, default=10
        Number of random HMM restarts for robustness runs.
    project_config : dict[str, object] | None, default=None
        Root project configuration used for data loading. When None, the
        function loads `config.toml` from the repository root.
    symbol_input_loader : Callable[[str, dict[str, object]], tuple[pd.DataFrame, pd.DataFrame]] | None, default=None
        Optional loader override for `(features, prices)` per symbol.
    """
    runtime_project_config = (
        project_config
        if project_config is not None
        else _load_project_config(project_root=PROJECT_ROOT)
    )
    load_symbol_inputs = symbol_input_loader or _load_symbol_inputs

    robustness_k_values = list(fixed_k_values or [])
    include_robustness_models = fixed_k_values is not None
    hmm_forecast_kwargs = _build_hmm_forecast_kwargs(
        hmm_n_iter=hmm_n_iter,
        hmm_random_restarts=hmm_random_restarts,
    )

    forecast_rows: list[dict[str, object]] = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        # Use positional arguments so tests can monkeypatch the loader with a
        # simple lambda without mirroring keyword names.
        features, prices = load_symbol_inputs(symbol, runtime_project_config)
        price_date_positions = _build_price_date_positions(prices=prices)

        for horizon in horizons:
            targets = build_forward_targets(
                prices=prices,
                features=features,
                horizon=horizon,
                annualization=annualization,
            )
            trailing_realized_volatility = _compute_trailing_realized_volatility(
                prices=prices,
                annualization=annualization,
                horizon=horizon,
                evaluation_index=pd.DatetimeIndex(features.index),
            )

            for feature_set in feature_sets:
                feature_subset = select_feature_columns(
                    features=features,
                    feature_set=feature_set,
                )
                combined = feature_subset.join(targets.loc[:, ["realized_vol"]])
                combined["trailing_realized_vol"] = trailing_realized_volatility
                combined = combined.dropna()

                if combined.empty:
                    continue
                if len(combined) <= min_train_size:
                    continue

                splits = build_expanding_window_splits(
                    dates=pd.DatetimeIndex(combined.index),
                    min_train_size=min_train_size,
                    step_size=step_size,
                )

                for split in splits:
                    embargoed_train_index = _apply_forward_target_embargo(
                        train_index=split.train_index,
                        test_index=split.test_index,
                        horizon=horizon,
                        price_date_positions=price_date_positions,
                    )
                    if len(embargoed_train_index) == 0:
                        continue

                    train_features = combined.loc[
                        embargoed_train_index,
                        feature_subset.columns,
                    ]
                    train_target = combined.loc[
                        embargoed_train_index,
                        "realized_vol",
                    ]
                    test_features = combined.loc[
                        split.test_index,
                        feature_subset.columns,
                    ]
                    test_target = combined.loc[split.test_index, "realized_vol"]
                    test_trailing_realized_vol = combined.loc[
                        split.test_index,
                        "trailing_realized_vol",
                    ]

                    for test_date, test_row in test_features.iterrows():
                        actual_value = float(test_target.loc[test_date])
                        atm_prediction = forecast_atm_iv(test_row=test_row)
                        trailing_prediction = forecast_trailing_realized_vol(
                            trailing_realized_vol=test_trailing_realized_vol,
                            test_date=test_date,
                        )
                        linear_prediction = forecast_linear_features(
                            train_features=train_features,
                            train_target=train_target,
                            test_row=test_row,
                        )
                        gmm_result = forecast_regime_mean(
                            train_features=train_features,
                            train_target=train_target,
                            test_row=test_row,
                            model_type="gmm",
                            min_k=regime_min_k,
                            max_k=regime_max_k,
                        )

                        # Always write the original Task 4 baselines first.
                        _append_forecast_row(
                            forecast_rows,
                            symbol=symbol,
                            feature_set=feature_set,
                            horizon=horizon,
                            forecast_date=test_date,
                            model_name="atm_iv",
                            prediction=atm_prediction,
                            actual=actual_value,
                        )
                        _append_forecast_row(
                            forecast_rows,
                            symbol=symbol,
                            feature_set=feature_set,
                            horizon=horizon,
                            forecast_date=test_date,
                            model_name="trailing_realized_vol",
                            prediction=trailing_prediction,
                            actual=actual_value,
                        )
                        _append_forecast_row(
                            forecast_rows,
                            symbol=symbol,
                            feature_set=feature_set,
                            horizon=horizon,
                            forecast_date=test_date,
                            model_name="linear_features",
                            prediction=linear_prediction,
                            actual=actual_value,
                        )
                        _append_regime_mean_row(
                            forecast_rows,
                            symbol=symbol,
                            feature_set=feature_set,
                            horizon=horizon,
                            forecast_date=test_date,
                            model_name="gmm_regime_mean",
                            regime_result=gmm_result,
                            actual=actual_value,
                        )

                        if include_robustness_models:
                            hmm_result = _forecast_hmm_safely(
                                train_features=train_features,
                                train_target=train_target,
                                test_row=test_row,
                                regime_min_k=regime_min_k,
                                regime_max_k=regime_max_k,
                                hmm_forecast_kwargs=hmm_forecast_kwargs,
                            )

                            if hmm_result is not None:
                                _append_regime_mean_row(
                                    forecast_rows,
                                    symbol=symbol,
                                    feature_set=feature_set,
                                    horizon=horizon,
                                    forecast_date=test_date,
                                    model_name="hmm_regime_mean",
                                    regime_result=hmm_result,
                                    actual=actual_value,
                                )

                            # Run fixed-K sweeps after the baseline models so the
                            # original Task 4 row set remains present.
                            for fixed_k in robustness_k_values:
                                fixed_gmm_result = forecast_regime_mean(
                                    train_features=train_features,
                                    train_target=train_target,
                                    test_row=test_row,
                                    model_type="gmm",
                                    min_k=int(fixed_k),
                                    max_k=int(fixed_k),
                                )
                                _append_regime_mean_row(
                                    forecast_rows,
                                    symbol=symbol,
                                    feature_set=feature_set,
                                    horizon=horizon,
                                    forecast_date=test_date,
                                    model_name=f"gmm_regime_mean_k_{int(fixed_k)}",
                                    regime_result=fixed_gmm_result,
                                    actual=actual_value,
                                )

                                fixed_hmm_result = _forecast_hmm_safely(
                                    train_features=train_features,
                                    train_target=train_target,
                                    test_row=test_row,
                                    regime_min_k=int(fixed_k),
                                    regime_max_k=int(fixed_k),
                                    hmm_forecast_kwargs=hmm_forecast_kwargs,
                                )

                                if fixed_hmm_result is None:
                                    continue

                                _append_regime_mean_row(
                                    forecast_rows,
                                    symbol=symbol,
                                    feature_set=feature_set,
                                    horizon=horizon,
                                    forecast_date=test_date,
                                    model_name=f"hmm_regime_mean_k_{int(fixed_k)}",
                                    regime_result=fixed_hmm_result,
                                    actual=actual_value,
                                )

    forecast_panel = pd.DataFrame(forecast_rows)
    metric_summary = summarize_metrics(forecast_panel=forecast_panel)

    forecast_panel.to_csv(OUTPUT_DIR / "forecast_panel.csv", index=False)
    metric_summary.to_csv(OUTPUT_DIR / "metric_summary.csv", index=False)
    write_research_summary(
        metric_summary=metric_summary,
        output_path=OUTPUT_DIR / "research_summary.md",
    )
    print(f"Research outputs written to {OUTPUT_DIR}")


def main() -> None:
    """Run the walk-forward research CLI using the experiment config defaults.

    This entrypoint sources symbols, evaluation settings, and output location
    from `walkforward.toml`, then injects that
    sample window into the shared root config before calling `run_research`.
    """
    global OUTPUT_DIR

    walkforward_config = _load_walkforward_config(project_root=PROJECT_ROOT)
    project_config = _load_project_config(project_root=PROJECT_ROOT)
    effective_project_config = _project_config_with_sample_window(
        project_config=project_config,
        walkforward_config=walkforward_config,
    )
    sample_config = walkforward_config["sample"]
    evaluation_config = walkforward_config["evaluation"]
    regime_config = walkforward_config["regime"]
    configured_fixed_k_values = list(regime_config["fixed_k_values"])
    fixed_k_values: list[int] | None = (
        configured_fixed_k_values if configured_fixed_k_values else None
    )

    OUTPUT_DIR = _resolve_output_dir(
        project_root=PROJECT_ROOT,
        walkforward_config=walkforward_config,
    )

    run_research(
        symbols=list(sample_config["symbols"]),
        feature_sets=list(evaluation_config["feature_sets"]),
        horizons=list(evaluation_config["horizons"]),
        min_train_size=int(evaluation_config["min_train_size"]),
        step_size=int(evaluation_config["step_size"]),
        annualization=int(evaluation_config["annualization"]),
        regime_min_k=int(regime_config["min_k"]),
        regime_max_k=int(regime_config["max_k"]),
        fixed_k_values=fixed_k_values,
        hmm_n_iter=int(regime_config["hmm_n_iter"]),
        hmm_random_restarts=int(regime_config["hmm_random_restarts"]),
        project_config=effective_project_config,
    )


if __name__ == "__main__":
    main()
