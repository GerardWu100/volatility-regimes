"""Tests for forward target construction in walk-forward research.

The targets combine forward realized volatility, variance risk premium, and
simple forward return on the same feature-date index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_build_forward_targets_aligns_to_feature_index() -> None:
    """Build forward targets on the feature index and verify the formulas."""
    from volatility_regimes.walkforward.targets import build_forward_targets

    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    prices = pd.DataFrame(
        {
            "date": dates,
            "close": [100.0, 110.0, 100.0, 105.0, 110.0],
        }
    )
    features = pd.DataFrame(
        {
            "atm_iv_near": [0.20, 0.21, 0.22, 0.23],
            "skew_near": [0.01, 0.01, 0.01, 0.01],
        },
        index=dates[:4],
    )

    targets = build_forward_targets(
        prices=prices,
        features=features,
        horizon=2,
        annualization=252,
    )

    assert list(targets.columns) == [
        "realized_vol",
        "variance_risk_premium",
        "forward_return",
    ]
    assert targets.index.equals(features.index)

    close_series = prices.set_index("date")["close"]
    log_returns = np.log(close_series / close_series.shift(1))
    future_window = log_returns.iloc[1:3]
    expected_realized_vol = future_window.std() * np.sqrt(252)
    expected_forward_return = close_series.shift(-2) / close_series - 1.0

    assert targets.loc[dates[0], "realized_vol"] == pytest.approx(expected_realized_vol)
    assert targets.loc[dates[0], "variance_risk_premium"] == pytest.approx(
        features.loc[dates[0], "atm_iv_near"] - expected_realized_vol
    )
    assert targets.loc[dates[0], "forward_return"] == pytest.approx(
        expected_forward_return.loc[dates[0]]
    )

    assert np.isnan(targets.loc[dates[3], "realized_vol"])
    assert np.isnan(targets.loc[dates[3], "forward_return"])


def test_build_forward_targets_rejects_horizon_one() -> None:
    """Reject a one-day horizon because realized volatility is undefined."""
    from volatility_regimes.walkforward.targets import build_forward_targets

    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    prices = pd.DataFrame(
        {
            "date": dates,
            "close": [100.0, 101.0, 102.0],
        }
    )
    features = pd.DataFrame(
        {"atm_iv_near": [0.20, 0.21, 0.22]},
        index=dates,
    )

    with pytest.raises(ValueError, match="horizon must be at least 2"):
        build_forward_targets(
            prices=prices,
            features=features,
            horizon=1,
            annualization=252,
        )


def test_build_forward_targets_rejects_duplicate_price_dates() -> None:
    """Reject duplicate price dates before target alignment."""
    from volatility_regimes.walkforward.targets import build_forward_targets

    dates = pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"])
    prices = pd.DataFrame(
        {
            "date": dates,
            "close": [100.0, 101.0, 102.0],
        }
    )
    features = pd.DataFrame(
        {"atm_iv_near": [0.20, 0.21]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    with pytest.raises(ValueError, match="Duplicate dates in prices"):
        build_forward_targets(
            prices=prices,
            features=features,
            horizon=2,
            annualization=252,
        )
