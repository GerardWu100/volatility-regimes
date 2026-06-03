"""Build forward targets for walk-forward regime research.

The targets in this module are aligned to the feature matrix index so that
each row represents one trade date and the corresponding future outcomes used
for model fitting or evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _close_series_from_prices(prices: pd.DataFrame) -> pd.Series:
    """Return a sorted close-price series indexed by date.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily price table with columns `date` and `close`.

    Returns
    -------
    pd.Series
        Close-price series sorted in ascending date order.
    """
    required_columns = {"date", "close"}
    missing_columns = sorted(required_columns.difference(prices.columns))
    if missing_columns:
        missing_list = ", ".join(missing_columns)
        raise KeyError(f"Missing required price columns: {missing_list}")

    duplicate_dates = prices.loc[prices["date"].duplicated(), "date"].dropna()
    if not duplicate_dates.empty:
        duplicate_date_list = ", ".join(
            duplicate_dates.astype(str).drop_duplicates().tolist()
        )
        message = f"Duplicate dates in prices are not allowed: {duplicate_date_list}"
        raise ValueError(message)

    sorted_prices = prices.loc[:, ["date", "close"]].sort_values("date")
    close_series = sorted_prices.set_index("date")["close"].astype(float)
    return close_series


def _forward_realized_vol(
    close_series: pd.Series,
    horizon: int,
    annualization: int,
) -> pd.Series:
    """Compute forward annualized realized volatility from daily close prices.

    Definitions
    -----------
    Let `r_t = ln(close_t / close_{t-1})` be the daily log return.
    For a horizon of `h` trading days, forward realized volatility at time `t`
    is the annualized standard deviation of the next `h` log returns:

    `rv_t = std(r_{t+1}, ..., r_{t+h}) * sqrt(annualization)`

    Parameters
    ----------
    close_series : pd.Series
        Daily close prices indexed by date.
    horizon : int
        Number of forward trading days in the realized-volatility window.
    annualization : int
        Trading days per year used to annualize volatility.

    Returns
    -------
    pd.Series
        Forward realized volatility indexed by date.
    """
    if horizon < 2:
        raise ValueError(
            "horizon must be at least 2 to compute realized volatility"
        )

    log_returns = np.log(close_series / close_series.shift(1))
    rolling_std = log_returns.rolling(window=horizon).std()
    annualized_vol = rolling_std * np.sqrt(float(annualization))

    # Shift the trailing window backward so the value lands on the start date
    # of the forward-looking window.
    forward_realized_vol = annualized_vol.shift(-horizon)
    forward_realized_vol.name = "realized_vol"
    return forward_realized_vol


def build_forward_targets(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    horizon: int,
    annualization: int,
) -> pd.DataFrame:
    """Build forward targets aligned to a feature matrix index.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily close prices with columns `date` and `close`.
    features : pd.DataFrame
        Feature matrix indexed by trade date. The function reads
        `atm_iv_near` from this frame to form the variance risk premium.
    horizon : int
        Forward-looking horizon in trading days.
    annualization : int
        Trading days per year used to annualize realized volatility.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed like `features` with columns:
        - `realized_vol`
        - `variance_risk_premium`
        - `forward_return`
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if annualization <= 0:
        raise ValueError("annualization must be positive")

    close_series = _close_series_from_prices(prices)

    # Build the forward-looking targets on the price-date index first.
    realized_vol = _forward_realized_vol(
        close_series=close_series,
        horizon=horizon,
        annualization=annualization,
    )
    forward_return = close_series.shift(-horizon) / close_series - 1.0
    forward_return.name = "forward_return"

    # Align everything to the feature index so downstream model frames stay
    # row-matched even when the price history has extra dates.
    targets = features.iloc[:, 0:0].copy()
    targets["realized_vol"] = realized_vol.reindex(features.index)
    targets["variance_risk_premium"] = (
        features["atm_iv_near"] - targets["realized_vol"]
    )
    targets["forward_return"] = forward_return.reindex(features.index)
    return targets
