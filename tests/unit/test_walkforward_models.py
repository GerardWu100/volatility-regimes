"""Tests for walk-forward benchmark and regime forecast models.

These tests cover simple benchmark forecasts and regime-conditional forecasts
used in leakage-safe walk-forward research.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_forecast_atm_iv_returns_current_atm_iv() -> None:
    """Return the current near-term ATM implied volatility unchanged."""
    from volatility_regimes.walkforward.models import forecast_atm_iv

    test_row = pd.Series(
        {
            "atm_iv_near": 0.24,
            "skew_near": -0.03,
        }
    )

    forecast_value = forecast_atm_iv(test_row=test_row)

    assert forecast_value == 0.24


def test_forecast_historical_mean_uses_only_training_targets() -> None:
    """Return the arithmetic mean of the supplied leakage-safe targets."""
    from volatility_regimes.walkforward.models import forecast_historical_mean

    train_target = pd.Series([0.10, 0.20, 0.30], dtype=float)

    assert forecast_historical_mean(train_target=train_target) == pytest.approx(0.20)


def test_forecast_linear_features_returns_finite_value() -> None:
    """Fit a linear forecast on the train window and score one test row."""
    from volatility_regimes.walkforward.models import forecast_linear_features

    train_features = pd.DataFrame(
        {
            "atm_iv_near": [0.15, 0.18, 0.21, 0.24, 0.27],
            "skew_near": [-0.01, -0.02, -0.03, -0.04, -0.05],
        }
    )
    train_target = pd.Series([0.14, 0.17, 0.20, 0.23, 0.26], dtype=float)
    test_row = pd.Series(
        {
            "atm_iv_near": 0.23,
            "skew_near": -0.035,
        }
    )

    forecast_value = forecast_linear_features(
        train_features=train_features,
        train_target=train_target,
        test_row=test_row,
    )

    assert np.isfinite(forecast_value)


def test_batch_forecasts_match_single_row_gmm_and_linear_results() -> None:
    """Batch fitting should preserve the established row-level predictions."""
    from volatility_regimes.walkforward.models import (
        forecast_linear_features,
        forecast_linear_features_batch,
        forecast_regime_mean,
        forecast_regime_mean_batch,
    )

    rng = np.random.default_rng(123)
    train_features = pd.DataFrame(
        {
            "atm_iv_near": np.concatenate(
                [
                    rng.normal(0.15, 0.01, 60),
                    rng.normal(0.30, 0.01, 60),
                ]
            ),
            "term_slope": np.concatenate(
                [
                    rng.normal(0.02, 0.005, 60),
                    rng.normal(-0.01, 0.005, 60),
                ]
            ),
        }
    )
    train_target = pd.Series(
        np.concatenate(
            [
                rng.normal(0.14, 0.01, 60),
                rng.normal(0.28, 0.01, 60),
            ]
        ),
        dtype=float,
    )
    test_features = pd.DataFrame(
        {
            "atm_iv_near": [0.16, 0.31],
            "term_slope": [0.018, -0.012],
        },
        index=pd.bdate_range("2025-01-02", periods=2),
    )

    linear_batch = forecast_linear_features_batch(
        train_features=train_features,
        train_target=train_target,
        test_features=test_features,
    )
    gmm_batch = forecast_regime_mean_batch(
        train_features=train_features,
        train_target=train_target,
        test_features=test_features,
        model_type="gmm",
        min_k=2,
        max_k=2,
    )

    for test_date, test_row in test_features.iterrows():
        single_linear = forecast_linear_features(
            train_features=train_features,
            train_target=train_target,
            test_row=test_row,
        )
        single_gmm = forecast_regime_mean(
            train_features=train_features,
            train_target=train_target,
            test_row=test_row,
            model_type="gmm",
            min_k=2,
            max_k=2,
        )
        assert linear_batch.loc[test_date] == pytest.approx(single_linear)
        assert gmm_batch.loc[test_date, "prediction"] == pytest.approx(
            single_gmm["prediction"]
        )
        assert int(gmm_batch.loc[test_date, "selected_k"]) == single_gmm["selected_k"]
        assert (
            int(gmm_batch.loc[test_date, "predicted_regime"])
            == single_gmm["predicted_regime"]
        )


def test_forecast_trailing_realized_vol_uses_current_test_date() -> None:
    """Read the trailing realized-volatility input at the current test date."""
    from volatility_regimes.walkforward.models import forecast_trailing_realized_vol

    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    trailing_realized_vol = pd.Series(
        [0.11, 0.12, 0.13, 0.14],
        index=dates,
        dtype=float,
    )

    forecast_value = forecast_trailing_realized_vol(
        trailing_realized_vol=trailing_realized_vol,
        test_date=dates[2],
    )

    assert forecast_value == 0.13


def test_forecast_regime_mean_gmm_returns_finite_value_and_selected_k() -> None:
    """Forecast the mean target of the inferred GMM regime from train data."""
    from volatility_regimes.walkforward.models import forecast_regime_mean

    rng = np.random.default_rng(42)

    low_vol_features = pd.DataFrame(
        {
            "atm_iv_near": rng.normal(loc=0.15, scale=0.01, size=80),
            "skew_near": rng.normal(loc=-0.01, scale=0.005, size=80),
        }
    )
    high_vol_features = pd.DataFrame(
        {
            "atm_iv_near": rng.normal(loc=0.35, scale=0.01, size=80),
            "skew_near": rng.normal(loc=-0.08, scale=0.005, size=80),
        }
    )
    train_features = pd.concat(
        [low_vol_features, high_vol_features],
        ignore_index=True,
    )
    train_target = pd.Series(
        np.concatenate(
            [
                rng.normal(loc=0.14, scale=0.01, size=80),
                rng.normal(loc=0.33, scale=0.01, size=80),
            ]
        ),
        dtype=float,
    )
    test_row = pd.Series(
        {
            "atm_iv_near": 0.36,
            "skew_near": -0.075,
        }
    )

    forecast_result = forecast_regime_mean(
        train_features=train_features,
        train_target=train_target,
        test_row=test_row,
        model_type="gmm",
        min_k=2,
        max_k=3,
    )

    assert set(forecast_result) >= {
        "prediction",
        "selected_k",
        "predicted_regime",
    }
    assert np.isfinite(forecast_result["prediction"])
    assert forecast_result["selected_k"] == 2
    assert isinstance(forecast_result["predicted_regime"], int)


def test_forecast_regime_mean_hmm_uses_sequence_aware_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decode the test point as the final step of the full observed sequence."""
    import volatility_regimes.walkforward.models as model_module

    class IdentityScaler:
        """Return the provided feature matrix unchanged."""

        def transform(self, feature_matrix: np.ndarray) -> np.ndarray:
            """Return the input array without modification."""
            return feature_matrix

    class FakeHMMModel:
        """Return different states for single-row vs sequence decoding."""

        def predict(self, feature_matrix: np.ndarray) -> np.ndarray:
            """Use sequence length to distinguish the decode path."""
            if len(feature_matrix) == 1:
                return np.array([0], dtype=int)

            decoded_states = np.array([0, 0, 1], dtype=int)
            return decoded_states

    def fake_standardize_features(
        feature_matrix: np.ndarray,
    ) -> tuple[np.ndarray, IdentityScaler]:
        """Bypass scaling so the test isolates HMM decode behavior."""
        return feature_matrix, IdentityScaler()

    def fake_fit_hmm(
        feature_matrix: np.ndarray,
        n_states: int,
        n_iter: int = 200,
        n_restarts: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, FakeHMMModel]:
        """Return fixed train labels and a model with path-dependent decoding."""
        _ = feature_matrix
        _ = n_states
        _ = n_iter
        _ = n_restarts
        train_labels = np.array([0, 1], dtype=int)
        transition_matrix = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=float)
        return train_labels, transition_matrix, FakeHMMModel()

    monkeypatch.setattr(
        model_module,
        "standardize_features",
        fake_standardize_features,
    )
    monkeypatch.setattr(
        model_module,
        "fit_hmm",
        fake_fit_hmm,
    )

    train_features = pd.DataFrame(
        {
            "atm_iv_near": [0.10, 0.30],
            "skew_near": [-0.01, -0.05],
        }
    )
    train_target = pd.Series([0.11, 0.29], dtype=float)
    test_row = pd.Series(
        {
            "atm_iv_near": 0.31,
            "skew_near": -0.06,
        }
    )

    forecast_result = model_module.forecast_regime_mean(
        train_features=train_features,
        train_target=train_target,
        test_row=test_row,
        model_type="hmm",
        min_k=2,
        max_k=2,
    )

    # The patched HMM returns state 0 for test-only decoding and state 1 as the
    # final state of the combined train+test sequence. Old single-row decoding
    # would therefore predict regime 0 and the low-vol mean instead.
    assert np.isfinite(forecast_result["prediction"])
    assert forecast_result["selected_k"] == 2
    assert forecast_result["predicted_regime"] == 1
    assert forecast_result["prediction"] == pytest.approx(0.29)


def test_forecast_regime_mean_hmm_returns_finite_value_and_selected_k() -> None:
    """Fit the HMM forecast path and return the expected output fields."""
    from volatility_regimes.walkforward.models import forecast_regime_mean

    rng = np.random.default_rng(7)

    low_vol_features = pd.DataFrame(
        {
            "atm_iv_near": rng.normal(loc=0.14, scale=0.01, size=80),
            "skew_near": rng.normal(loc=-0.01, scale=0.004, size=80),
        }
    )
    high_vol_features = pd.DataFrame(
        {
            "atm_iv_near": rng.normal(loc=0.33, scale=0.01, size=80),
            "skew_near": rng.normal(loc=-0.07, scale=0.004, size=80),
        }
    )
    train_features = pd.concat(
        [low_vol_features, high_vol_features],
        ignore_index=True,
    )
    train_target = pd.Series(
        np.concatenate(
            [
                rng.normal(loc=0.13, scale=0.01, size=80),
                rng.normal(loc=0.31, scale=0.01, size=80),
            ]
        ),
        dtype=float,
    )
    test_row = pd.Series(
        {
            "atm_iv_near": 0.34,
            "skew_near": -0.065,
        }
    )

    forecast_result = forecast_regime_mean(
        train_features=train_features,
        train_target=train_target,
        test_row=test_row,
        model_type="hmm",
        min_k=2,
        max_k=2,
        hmm_n_iter=50,
        hmm_random_restarts=2,
    )

    assert set(forecast_result) >= {
        "prediction",
        "selected_k",
        "predicted_regime",
    }
    assert np.isfinite(forecast_result["prediction"])
    assert forecast_result["selected_k"] == 2
    assert isinstance(forecast_result["predicted_regime"], int)


def test_forecast_regime_mean_uses_atm_iv_column_position_not_column_zero() -> None:
    """Order regimes by the actual ATM column even when it is not first."""
    from volatility_regimes.walkforward.models import forecast_regime_mean

    rng = np.random.default_rng(11)

    low_vol_features = pd.DataFrame(
        {
            "skew_near": rng.normal(loc=0.08, scale=0.005, size=80),
            "atm_iv_near": rng.normal(loc=0.15, scale=0.01, size=80),
        }
    )
    high_vol_features = pd.DataFrame(
        {
            "skew_near": rng.normal(loc=-0.08, scale=0.005, size=80),
            "atm_iv_near": rng.normal(loc=0.35, scale=0.01, size=80),
        }
    )
    train_features = pd.concat(
        [low_vol_features, high_vol_features],
        ignore_index=True,
    )
    train_target = pd.Series(
        np.concatenate(
            [
                rng.normal(loc=0.14, scale=0.01, size=80),
                rng.normal(loc=0.33, scale=0.01, size=80),
            ]
        ),
        dtype=float,
    )
    test_row = pd.Series(
        {
            "skew_near": -0.075,
            "atm_iv_near": 0.36,
        }
    )

    forecast_result = forecast_regime_mean(
        train_features=train_features,
        train_target=train_target,
        test_row=test_row,
        model_type="gmm",
        min_k=2,
        max_k=2,
    )

    assert forecast_result["predicted_regime"] == 1
    assert forecast_result["prediction"] > 0.25
