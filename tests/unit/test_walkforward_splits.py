"""Tests for leakage-safe walk-forward split construction."""

from __future__ import annotations

import pandas as pd
import pytest


def test_build_expanding_window_splits_never_use_future_rows() -> None:
    """Ensure each expanding split keeps training dates strictly before test dates."""
    from volatility_regimes.walkforward.splits import (
        build_expanding_window_splits,
    )

    dates = pd.date_range("2024-01-02", periods=8, freq="B")

    splits = build_expanding_window_splits(
        dates=dates,
        min_train_size=3,
        step_size=2,
    )

    assert len(splits) == 3
    assert splits[0].train_index.equals(dates[:3])
    assert splits[0].test_index.equals(dates[3:5])

    assert splits[1].train_index.equals(dates[:5])
    assert splits[1].test_index.equals(dates[5:7])

    assert splits[2].train_index.equals(dates[:7])
    assert splits[2].test_index.equals(dates[7:8])

    for split in splits:
        assert split.train_index.max() < split.test_index.min()
        assert split.train_index.intersection(split.test_index).empty


def test_build_rolling_window_splits_keep_fixed_training_history_length() -> None:
    """Ensure each rolling split preserves the requested training window size."""
    from volatility_regimes.walkforward.splits import build_rolling_window_splits

    dates = pd.date_range("2024-01-02", periods=9, freq="B")

    splits = build_rolling_window_splits(
        dates=dates,
        train_window_size=4,
        step_size=2,
    )

    assert len(splits) == 3
    assert splits[0].train_index.equals(dates[:4])
    assert splits[0].test_index.equals(dates[4:6])

    assert splits[1].train_index.equals(dates[2:6])
    assert splits[1].test_index.equals(dates[6:8])

    assert splits[2].train_index.equals(dates[4:8])
    assert splits[2].test_index.equals(dates[8:9])

    for split in splits:
        assert len(split.train_index) == 4
        assert split.train_index.max() < split.test_index.min()
        assert split.train_index.intersection(split.test_index).empty


def test_build_rolling_window_splits_rejects_nonpositive_sizes() -> None:
    """Reject invalid window and step sizes before building splits."""
    from volatility_regimes.walkforward.splits import build_rolling_window_splits

    dates = pd.date_range("2024-01-02", periods=5, freq="B")

    with pytest.raises(ValueError, match="train_window_size must be positive"):
        build_rolling_window_splits(
            dates=dates,
            train_window_size=0,
            step_size=1,
        )


@pytest.mark.parametrize(
    ("builder_name", "builder_kwargs"),
    [
        (
            "build_expanding_window_splits",
            {"min_train_size": 2, "step_size": 1},
        ),
        (
            "build_rolling_window_splits",
            {"train_window_size": 2, "step_size": 1},
        ),
    ],
)
def test_build_walkforward_splits_reject_duplicate_dates(
    builder_name: str,
    builder_kwargs: dict[str, int],
) -> None:
    """Reject duplicate timestamps before any split boundaries are built."""
    from volatility_regimes.walkforward import splits as splits_module

    builder = getattr(splits_module, builder_name)
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04"])

    with pytest.raises(
        ValueError,
        match="dates must not contain duplicate timestamps",
    ):
        builder(dates=dates, **builder_kwargs)
