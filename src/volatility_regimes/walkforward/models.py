"""Walk-forward benchmark and regime forecast models.

This module provides simple benchmark forecasts and regime-conditional
forecasts for leakage-safe walk-forward research. Every forecast uses only the
provided training window and the current test row.
"""

from __future__ import annotations

from typing import Literal
from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from volatility_regimes.regimes.latent_models import fit_gmm
from volatility_regimes.regimes.latent_models import fit_hmm
from volatility_regimes.regimes.latent_models import order_regimes_by_volatility
from volatility_regimes.regimes.latent_models import standardize_features


class RegimeMeanForecast(TypedDict):
    """Typed output contract for regime-conditional forecast helpers.

    Fields
    ------
    prediction : float
        Forecast equal to the train-window mean target of the inferred
        ordered regime.
    selected_k : int
        Regime count used for the fitted forecast model. This is either the
        BIC-selected value or a fixed `K` imposed by the caller.
    predicted_regime : int
        Ordered regime label assigned to the current test row.
    """

    prediction: float
    selected_k: int
    predicted_regime: int


def forecast_atm_iv(test_row: pd.Series) -> float:
    """Forecast realized volatility with the current near-term ATM IV.

    Parameters
    ----------
    test_row : pd.Series
        Current walk-forward test row. The row must contain `atm_iv_near`.

    Returns
    -------
    float
        The current near-term ATM implied volatility.
    """
    forecast_value = float(test_row["atm_iv_near"])
    return forecast_value


def forecast_historical_mean(train_target: pd.Series) -> float:
    """Forecast with the expanding-window mean realized volatility.

    Parameters
    ----------
    train_target : pd.Series
        Leakage-safe training realized-volatility target in annualized decimal
        units.

    Returns
    -------
    float
        Arithmetic mean of the available training targets.

    Notes
    -----
    This benchmark tests whether surface features or regime assignments add
    information beyond the unconditional volatility level observed so far.
    """
    return float(train_target.mean())


def forecast_linear_features(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    test_row: pd.Series,
) -> float:
    """Fit a linear model on the train window and predict one test row.

    Parameters
    ----------
    train_features : pd.DataFrame
        Train-window feature matrix.
    train_target : pd.Series
        Train-window target aligned to `train_features`.
    test_row : pd.Series
        Current walk-forward test row with the same feature columns.

    Returns
    -------
    float
        Linear regression forecast for the current test row.
    """
    test_feature_frame = test_row.reindex(train_features.columns).to_frame().T
    forecast_series = forecast_linear_features_batch(
        train_features=train_features,
        train_target=train_target,
        test_features=test_feature_frame,
    )
    forecast_value = float(forecast_series.iloc[0])
    return forecast_value


def forecast_linear_features_batch(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    test_features: pd.DataFrame,
) -> pd.Series:
    """Fit one linear model and predict a complete walk-forward test block.

    Parameters
    ----------
    train_features : pd.DataFrame
        Leakage-safe training feature matrix.
    train_target : pd.Series
        Training target aligned to ``train_features``.
    test_features : pd.DataFrame
        Test-block features with the same columns as ``train_features``.

    Returns
    -------
    pd.Series
        One prediction per test row, preserving ``test_features.index``.

    Notes
    -----
    A walk-forward split has one fixed training window. Fitting once per block
    gives the same linear predictions as refitting for every row while avoiding
    redundant work.
    """
    model = LinearRegression()
    model.fit(train_features, train_target)
    ordered_test_features = test_features.reindex(columns=train_features.columns)
    predictions = model.predict(ordered_test_features)
    return pd.Series(predictions, index=test_features.index, dtype=float)


def forecast_trailing_realized_vol(
    trailing_realized_vol: pd.Series,
    test_date: pd.Timestamp,
) -> float:
    """Forecast with the trailing realized volatility observed on test_date.

    Parameters
    ----------
    trailing_realized_vol : pd.Series
        Trailing realized-volatility series indexed by date.
    test_date : pd.Timestamp
        Current walk-forward test date.

    Returns
    -------
    float
        Trailing realized volatility for the current test date.
    """
    forecast_value = float(trailing_realized_vol.loc[test_date])
    return forecast_value


def _ordered_regime_mapping(
    original_labels: np.ndarray,
    ordered_labels: np.ndarray,
) -> dict[int, int]:
    """Build a mapping from original regime labels to ordered labels.

    Parameters
    ----------
    original_labels : np.ndarray
        Regime labels returned directly by the fitted model.
    ordered_labels : np.ndarray
        Regime labels reordered from low-vol to high-vol.

    Returns
    -------
    dict[int, int]
        Mapping from original label to ordered label.
    """
    original_to_ordered: dict[int, int] = {}

    for original_label, ordered_label in zip(
        original_labels,
        ordered_labels,
        strict=True,
    ):
        original_label_int = int(original_label)
        ordered_label_int = int(ordered_label)

        if original_label_int not in original_to_ordered:
            original_to_ordered[original_label_int] = ordered_label_int

    return original_to_ordered


def _selected_hmm_state_count(
    standardized_train_features: np.ndarray,
    min_k: int,
    max_k: int,
) -> int:
    """Select the HMM state count with the existing GMM BIC helper.

    Parameters
    ----------
    standardized_train_features : np.ndarray
        Standardized train-window features.
    min_k : int
        Minimum candidate state count.
    max_k : int
        Maximum candidate state count.

    Returns
    -------
    int
        Selected HMM state count.
    """
    if min_k == max_k:
        return int(min_k)

    # Reuse the existing BIC-based selector so the HMM state count is driven
    # by the same train-window structure as the GMM benchmark.
    _, _, selected_k, _ = fit_gmm(
        feature_matrix=standardized_train_features,
        min_k=min_k,
        max_k=max_k,
    )
    return int(selected_k)


def _atm_iv_column_index(train_features: pd.DataFrame) -> int:
    """Return the ATM IV column index from the actual feature order.

    Parameters
    ----------
    train_features : pd.DataFrame
        Train-window feature matrix.

    Returns
    -------
    int
        Positional index of `atm_iv_near` in the feature matrix.
    """
    atm_iv_column_index = int(train_features.columns.get_loc("atm_iv_near"))
    return atm_iv_column_index


def _forecast_from_ordered_regime(
    predicted_ordered_label: int | None,
    ordered_target_means: pd.Series,
    train_target: pd.Series,
) -> tuple[float, int]:
    """Resolve prediction and reported regime from ordered-regime means.

    Parameters
    ----------
    predicted_ordered_label : int | None
        Ordered label inferred for the test row, if available.
    ordered_target_means : pd.Series
        Mapping from ordered regime label to train-window mean target.
    train_target : pd.Series
        Full train-window target used as the fallback prediction source.

    Returns
    -------
    tuple[float, int]
        Prediction value and regime label to report in output.
    """
    fallback_prediction = float(train_target.mean())

    # If we cannot map the predicted label into the ordered regime space,
    # fall back to the train-window mean and mark regime as unknown.
    if predicted_ordered_label is None:
        return fallback_prediction, -1

    predicted_regime = int(predicted_ordered_label)

    # If the ordered regime exists conceptually but has no mean in this train
    # window, keep the regime label for diagnostics and still fall back.
    if predicted_regime not in ordered_target_means.index:
        return fallback_prediction, predicted_regime

    prediction = float(ordered_target_means.loc[predicted_regime])
    return prediction, predicted_regime


def forecast_regime_mean(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    test_row: pd.Series,
    model_type: Literal["gmm", "hmm"],
    min_k: int,
    max_k: int,
    hmm_n_iter: int = 200,
    hmm_random_restarts: int = 10,
) -> RegimeMeanForecast:
    """Forecast the target as the mean of the inferred ordered regime.

    Parameters
    ----------
    train_features : pd.DataFrame
        Train-window feature matrix.
    train_target : pd.Series
        Train-window target aligned to `train_features`.
    test_row : pd.Series
        Current walk-forward test row with the same feature columns.
    model_type : {"gmm", "hmm"}
        Regime model used for the forecast.
    min_k : int
        Minimum candidate regime count.
    max_k : int
        Maximum candidate regime count.
    hmm_n_iter : int, default=200
        Maximum expectation-maximization iterations for the HMM.
    hmm_random_restarts : int, default=10
        Number of random HMM initializations.

    Returns
    -------
    RegimeMeanForecast
        Dictionary with keys:
        - `prediction`: ordered-regime mean forecast.
        - `selected_k`: selected regime count.
        - `predicted_regime`: ordered regime predicted for the test row.

    Notes
    -----
    The HMM path decodes the test row as the final observation of the
    train-plus-test sequence, not as an isolated row. This preserves the
    transition-aware state assignment verified by the HMM forecast-path tests.
    """
    test_feature_frame = test_row.reindex(train_features.columns).to_frame().T
    batch_result = forecast_regime_mean_batch(
        train_features=train_features,
        train_target=train_target,
        test_features=test_feature_frame,
        model_type=model_type,
        min_k=min_k,
        max_k=max_k,
        hmm_n_iter=hmm_n_iter,
        hmm_random_restarts=hmm_random_restarts,
    )
    first_result = batch_result.iloc[0]
    result: RegimeMeanForecast = {
        "prediction": float(first_result["prediction"]),
        "selected_k": int(first_result["selected_k"]),
        "predicted_regime": int(first_result["predicted_regime"]),
    }
    return result


def forecast_regime_mean_batch(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    test_features: pd.DataFrame,
    model_type: Literal["gmm", "hmm"],
    min_k: int,
    max_k: int,
    hmm_n_iter: int = 200,
    hmm_random_restarts: int = 10,
) -> pd.DataFrame:
    """Fit one latent-state model and forecast a complete test block.

    Parameters
    ----------
    train_features : pd.DataFrame
        Leakage-safe training feature matrix.
    train_target : pd.Series
        Training realized-volatility target aligned to ``train_features``.
    test_features : pd.DataFrame
        Test-block feature rows in chronological order.
    model_type : {"gmm", "hmm"}
        Gaussian Mixture Model (GMM) or Hidden Markov Model (HMM).
    min_k : int
        Minimum candidate regime count.
    max_k : int
        Maximum candidate regime count.
    hmm_n_iter : int, default=200
        Maximum expectation-maximization iterations for an HMM fit.
    hmm_random_restarts : int, default=10
        Number of HMM random initializations.

    Returns
    -------
    pd.DataFrame
        Index matches ``test_features``. Columns are ``prediction``,
        ``selected_k``, and ``predicted_regime``.

    Notes
    -----
    Scaling and model fitting use training rows only. GMM test states are
    independent conditional assignments. Each HMM test state is decoded as
    the last observation of the training sequence plus that one test row,
    preserving the single-row forecast semantics without refitting the HMM.
    """
    if test_features.empty:
        return pd.DataFrame(
            columns=["prediction", "selected_k", "predicted_regime"],
            index=test_features.index,
        )

    # Standardize from train data only, then transform the full test block with
    # the same scaler. Test information never enters the fitted scale.
    train_feature_matrix = train_features.to_numpy(dtype=float)
    standardized_train_features, fitted_scaler = standardize_features(
        feature_matrix=train_feature_matrix
    )
    ordered_test_features = test_features.reindex(columns=train_features.columns)
    standardized_test_features = fitted_scaler.transform(
        ordered_test_features.to_numpy(dtype=float)
    )
    atm_iv_column_index = _atm_iv_column_index(train_features=train_features)

    if model_type == "gmm":
        # GMM assigns every test row independently under one train-only fit.
        train_labels, _, selected_k, fitted_model = fit_gmm(
            feature_matrix=standardized_train_features,
            min_k=min_k,
            max_k=max_k,
        )
        predicted_test_labels = fitted_model.predict(standardized_test_features)
    elif model_type == "hmm":
        # HMM reuses the same K-selection rule, then fits with sequence
        # dynamics so state decoding can use transition persistence.
        selected_k = _selected_hmm_state_count(
            standardized_train_features=standardized_train_features,
            min_k=min_k,
            max_k=max_k,
        )
        train_labels, _, fitted_model = fit_hmm(
            feature_matrix=standardized_train_features,
            n_states=selected_k,
            n_iter=hmm_n_iter,
            n_restarts=hmm_random_restarts,
        )

        # Preserve the established one-date forecast contract: each state is
        # decoded from the train sequence plus that row, with no future test
        # rows included in its information set.
        predicted_test_labels = np.asarray(
            [
                int(
                    fitted_model.predict(
                        np.vstack([standardized_train_features, test_feature_row])
                    )[-1]
                )
                for test_feature_row in standardized_test_features
            ],
            dtype=int,
        )
    else:
        raise ValueError("model_type must be either 'gmm' or 'hmm'")

    ordered_train_labels = order_regimes_by_volatility(
        labels=train_labels,
        standardized_features=standardized_train_features,
        atm_iv_col_idx=atm_iv_column_index,
    )
    original_to_ordered = _ordered_regime_mapping(
        original_labels=train_labels,
        ordered_labels=ordered_train_labels,
    )

    # Compute one target mean per ordered regime label for the forecast lookup.
    ordered_target_means = (
        pd.DataFrame(
            {
                "ordered_regime": ordered_train_labels,
                "target": train_target.to_numpy(dtype=float),
            }
        )
        .groupby("ordered_regime")["target"]
        .mean()
    )

    result_rows: list[dict[str, float | int]] = []
    for predicted_test_label in predicted_test_labels:
        predicted_ordered_label = original_to_ordered.get(int(predicted_test_label))
        forecast_value, predicted_regime = _forecast_from_ordered_regime(
            predicted_ordered_label=predicted_ordered_label,
            ordered_target_means=ordered_target_means,
            train_target=train_target,
        )
        result_rows.append(
            {
                "prediction": forecast_value,
                "selected_k": int(selected_k),
                "predicted_regime": predicted_regime,
            }
        )

    return pd.DataFrame(result_rows, index=test_features.index)
