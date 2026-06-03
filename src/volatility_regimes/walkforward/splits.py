"""Construct leakage-safe walk-forward train/test splits.

The split builders in this module operate on an ordered
``pandas.DatetimeIndex`` and return explicit train/test boundaries for each
walk-forward iteration. The train window always ends strictly before the test
window starts, which keeps the split leakage-safe for time-series research.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class WalkForwardSplit:
    """One walk-forward split with explicit train and test date indices.

    Parameters
    ----------
    train_index : pd.DatetimeIndex
        Dates used for model fitting.
    test_index : pd.DatetimeIndex
        Dates used for out-of-sample evaluation immediately after the train
        window.
    """

    train_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex


def _validate_dates(dates: pd.DatetimeIndex) -> None:
    """Validate the shared DatetimeIndex contract for split builders.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Ordered date index to split.
    """
    if not isinstance(dates, pd.DatetimeIndex):
        raise TypeError("dates must be a pandas.DatetimeIndex")
    if dates.has_duplicates:
        raise ValueError("dates must not contain duplicate timestamps")
    if not dates.is_monotonic_increasing:
        raise ValueError("dates must be sorted in ascending order")


def build_expanding_window_splits(
    dates: pd.DatetimeIndex,
    min_train_size: int,
    step_size: int,
) -> list[WalkForwardSplit]:
    """Build expanding-window walk-forward splits.

    The first split uses the first ``min_train_size`` rows for training and the
    next ``step_size`` rows for testing. Each subsequent split advances the
    test window by ``step_size`` rows while expanding the training window to
    include all rows seen so far.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Ordered date index to split.
    min_train_size : int
        Minimum number of rows required before the first test window starts.
    step_size : int
        Number of rows in each test chunk and the forward step between splits.

    Returns
    -------
    list[WalkForwardSplit]
        Expanding walk-forward splits in chronological order.
    """
    _validate_dates(dates=dates)
    if min_train_size <= 0:
        raise ValueError("min_train_size must be positive")
    if step_size <= 0:
        raise ValueError("step_size must be positive")

    splits: list[WalkForwardSplit] = []
    train_end = min_train_size

    while train_end < len(dates):
        test_start = train_end
        test_end = min(test_start + step_size, len(dates))

        # Training always uses everything strictly before the test window.
        train_index = dates[:train_end]
        test_index = dates[test_start:test_end]
        splits.append(
            WalkForwardSplit(
                train_index=train_index,
                test_index=test_index,
            )
        )

        # Expand the history by the same stride used for testing.
        train_end = test_end

    return splits


def build_rolling_window_splits(
    dates: pd.DatetimeIndex,
    train_window_size: int,
    step_size: int,
) -> list[WalkForwardSplit]:
    """Build rolling-window walk-forward splits.

    The first split uses the first ``train_window_size`` rows for training and
    the next ``step_size`` rows for testing. Each subsequent split advances by
    ``step_size`` rows while keeping the training window length fixed.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Ordered date index to split.
    train_window_size : int
        Fixed number of rows in each training window.
    step_size : int
        Number of rows in each test chunk and the forward step between splits.

    Returns
    -------
    list[WalkForwardSplit]
        Rolling walk-forward splits in chronological order.
    """
    _validate_dates(dates=dates)
    if train_window_size <= 0:
        raise ValueError("train_window_size must be positive")
    if step_size <= 0:
        raise ValueError("step_size must be positive")

    splits: list[WalkForwardSplit] = []
    train_end = train_window_size

    while train_end < len(dates):
        train_start = train_end - train_window_size
        test_start = train_end
        test_end = min(test_start + step_size, len(dates))

        # The rolling window keeps a fixed-length history by trimming the
        # oldest rows before each new split.
        train_index = dates[train_start:train_end]
        test_index = dates[test_start:test_end]
        splits.append(
            WalkForwardSplit(
                train_index=train_index,
                test_index=test_index,
            )
        )

        # Advance the window by the same step used to build the test chunk.
        train_end = test_end

    return splits
