"""Extract daily volatility-surface features from options chain data.

This module converts cleaned options rows into a daily feature matrix used by
the regime classifiers. Features are computed in delta space because delta is a
 scale-normalized moneyness coordinate across strike levels and spot levels.

Daily output columns
--------------------
1. atm_iv_near
2. atm_iv_mid
3. skew_near
4. skew_mid
5. butterfly_near
6. butterfly_mid
7. term_slope

Feature-set registry
--------------------
The walk-forward regime research code selects deterministic subsets of these
columns. The registry below keeps those subsets in one place so tests and
callers can rely on a stable column order.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


FEATURE_SET_REGISTRY: dict[str, tuple[str, ...]] = {
    # Full surface feature set used by the regime models.
    "full": (
        "atm_iv_near",
        "atm_iv_mid",
        "skew_near",
        "skew_mid",
        "butterfly_near",
        "butterfly_mid",
        "term_slope",
    ),
    # ATM-only view keeps only the near-term ATM level.
    "atm_only": (
        "atm_iv_near",
    ),
    # ATM plus term structure isolates level and slope information.
    "atm_term": (
        "atm_iv_near",
        "atm_iv_mid",
        "term_slope",
    ),
    # ATM plus skew keeps the asymmetric smile tilt terms.
    "atm_skew": (
        "atm_iv_near",
        "atm_iv_mid",
        "skew_near",
        "skew_mid",
    ),
    # Near-only focuses on the front-end smile and level.
    "near_only": (
        "atm_iv_near",
        "skew_near",
        "butterfly_near",
    ),
}


def select_feature_columns(
    features: pd.DataFrame, feature_set: str
) -> pd.DataFrame:
    """Select a named, deterministic subset of feature columns.

    Parameters
    ----------
    features : pd.DataFrame
        Daily feature matrix indexed by trade date.
    feature_set : str
        Registry key in `FEATURE_SET_REGISTRY`.

    Returns
    -------
    pd.DataFrame
        Copy of `features` restricted to the selected columns in registry
        order.

    Raises
    ------
    ValueError
        If `feature_set_name` is not registered.
    KeyError
        If any required feature column is missing from `features`.
    """
    try:
        selected_columns = FEATURE_SET_REGISTRY[feature_set]
    except KeyError as error:
        available_sets = ", ".join(sorted(FEATURE_SET_REGISTRY))
        message = (
            f"Unknown feature set '{feature_set}'. "
            f"Available sets: {available_sets}"
        )
        raise ValueError(message) from error

    missing_columns = [column for column in selected_columns if column not in features]
    if missing_columns:
        missing_list = ", ".join(missing_columns)
        raise KeyError(f"Missing required feature columns: {missing_list}")

    # `loc` preserves the exact registry order, which keeps downstream modeling
    # deterministic and makes tests stable.
    selected_features = features.loc[:, selected_columns].copy()
    return selected_features


def interpolate_iv_at_delta(chain: pd.DataFrame, target_delta: float) -> float:
    """Interpolate implied volatility at a target delta.

    Parameters
    ----------
    chain : pd.DataFrame
        One option side for one expiry on one trade date.
        Required columns:
        - delta : float
        - mid_iv : float
    target_delta : float
        Delta at which implied volatility is requested.

    Returns
    -------
    float
        Interpolated implied volatility value. Returns `np.nan` when the target
        is outside the observed delta range.
    """
    sorted_chain = chain.sort_values("delta")

    deltas = sorted_chain["delta"].to_numpy(dtype=float)
    ivs = sorted_chain["mid_iv"].to_numpy(dtype=float)

    if len(deltas) == 0:
        return float(np.nan)

    exact_match_mask = np.isclose(deltas, target_delta, atol=1e-4)
    if exact_match_mask.any():
        exact_iv = ivs[exact_match_mask][0]
        return float(exact_iv)

    lower_mask = deltas < target_delta
    upper_mask = deltas > target_delta

    if not lower_mask.any():
        return float(np.nan)
    if not upper_mask.any():
        return float(np.nan)

    lower_index = np.where(lower_mask)[0][-1]
    upper_index = np.where(upper_mask)[0][0]

    delta_low = deltas[lower_index]
    delta_high = deltas[upper_index]
    iv_low = ivs[lower_index]
    iv_high = ivs[upper_index]

    delta_fraction = (target_delta - delta_low) / (delta_high - delta_low)
    interpolated_iv = iv_low + (iv_high - iv_low) * delta_fraction
    return float(interpolated_iv)


def compute_skew(
    put_chain: pd.DataFrame,
    call_chain: pd.DataFrame,
    wing_delta: float = 0.25,
) -> float:
    """Compute 25-delta skew as put wing IV minus call wing IV.

    Parameters
    ----------
    put_chain : pd.DataFrame
        Put rows with negative delta values.
    call_chain : pd.DataFrame
        Call rows with positive delta values.
    wing_delta : float, default=0.25
        Absolute delta used for wing interpolation.

    Returns
    -------
    float
        Skew value `iv_put_wing - iv_call_wing`.
    """
    put_wing_iv = interpolate_iv_at_delta(put_chain, target_delta=-wing_delta)
    call_wing_iv = interpolate_iv_at_delta(call_chain, target_delta=wing_delta)

    skew_value = put_wing_iv - call_wing_iv
    return float(skew_value)


def compute_butterfly(iv_25d_put: float, iv_25d_call: float, iv_atm: float) -> float:
    """Compute smile curvature proxy (butterfly).

    Formula
    -------
    butterfly = 0.5 * (iv_25d_put + iv_25d_call) - iv_atm

    Parameters
    ----------
    iv_25d_put : float
        Put implied volatility at -0.25 delta.
    iv_25d_call : float
        Call implied volatility at +0.25 delta.
    iv_atm : float
        ATM implied volatility measured at -0.50 put delta.

    Returns
    -------
    float
        Butterfly value.
    """
    butterfly_value = 0.5 * (iv_25d_put + iv_25d_call) - iv_atm
    return float(butterfly_value)


def _select_expiry(
    options_day: pd.DataFrame,
    dte_min: int,
    dte_target: int,
    dte_max: int,
) -> int | None:
    """Select the expiry whose DTE is closest to target within a bucket."""
    dte_mask = options_day["dte"].between(dte_min, dte_max)
    candidate_dtes = options_day.loc[dte_mask, "dte"].unique()

    if len(candidate_dtes) == 0:
        return None

    distances = np.abs(candidate_dtes - dte_target)
    closest_index = int(np.argmin(distances))
    selected_dte = int(candidate_dtes[closest_index])
    return selected_dte


def _assign_expiry_bucket_features(
    row: dict[str, float | pd.Timestamp],
    day_data: pd.DataFrame,
    bucket_prefix: str,
    selected_dte: int | None,
    wing_delta: float,
    min_strikes: int,
) -> None:
    """Fill ATM, skew, and butterfly columns for one near or mid expiry bucket."""
    column_names = ("atm_iv", "skew", "butterfly")
    output_keys = tuple(f"{name}_{bucket_prefix}" for name in column_names)

    if selected_dte is None:
        for output_key in output_keys:
            row[output_key] = float(np.nan)
        return

    expiry_slice = day_data.loc[day_data["dte"] == selected_dte]
    expiry_features = _extract_features_one_expiry(
        expiry_data=expiry_slice,
        wing_delta=wing_delta,
        min_strikes_per_side=min_strikes,
    )
    row[output_keys[0]] = expiry_features["atm_iv"]
    row[output_keys[1]] = expiry_features["skew"]
    row[output_keys[2]] = expiry_features["butterfly"]


def _extract_features_one_expiry(
    expiry_data: pd.DataFrame,
    wing_delta: float,
    min_strikes_per_side: int,
) -> dict[str, float]:
    """Extract ATM, skew, and butterfly for one date-expiry slice."""
    puts = expiry_data.loc[expiry_data["option_type"] == "p"]
    calls = expiry_data.loc[expiry_data["option_type"] == "c"]

    empty_result = {
        "atm_iv": float(np.nan),
        "skew": float(np.nan),
        "butterfly": float(np.nan),
    }

    if len(puts) < min_strikes_per_side:
        return empty_result
    if len(calls) < min_strikes_per_side:
        return empty_result

    atm_iv = interpolate_iv_at_delta(puts, target_delta=-0.50)
    skew_value = compute_skew(put_chain=puts, call_chain=calls, wing_delta=wing_delta)

    put_wing_iv = interpolate_iv_at_delta(puts, target_delta=-wing_delta)
    call_wing_iv = interpolate_iv_at_delta(calls, target_delta=wing_delta)
    butterfly_value = compute_butterfly(put_wing_iv, call_wing_iv, atm_iv)

    return {
        "atm_iv": atm_iv,
        "skew": skew_value,
        "butterfly": butterfly_value,
    }


def extract_features(
    options: pd.DataFrame,
    near_dte_min: int = 15,
    near_dte_target: int = 30,
    near_dte_max: int = 45,
    mid_dte_min: int = 45,
    mid_dte_target: int = 90,
    mid_dte_max: int = 120,
    wing_delta: float = 0.25,
    min_strikes: int = 5,
) -> pd.DataFrame:
    """Build the daily 7-feature matrix for regime modeling.

    Parameters
    ----------
    options : pd.DataFrame
        Cleaned options table from `data_loader.load_options`.
        Required columns:
        - trade_date : datetime64[ns]
        - expiry_date : datetime64[ns]
        - option_type : str in {'p', 'c'}
        - delta : float
        - mid_iv : float
        - dte : int
    near_dte_min : int, default=15
        Lower bound for near-term expiry bucket.
    near_dte_target : int, default=30
        Target DTE for selecting near-term expiry.
    near_dte_max : int, default=45
        Upper bound for near-term expiry bucket.
    mid_dte_min : int, default=45
        Lower bound for mid-term expiry bucket.
    mid_dte_target : int, default=90
        Target DTE for selecting mid-term expiry.
    mid_dte_max : int, default=120
        Upper bound for mid-term expiry bucket.
    wing_delta : float, default=0.25
        Wing delta magnitude for skew and butterfly.
    min_strikes : int, default=5
        Minimum number of put and call strikes required per expiry.

    Returns
    -------
    pd.DataFrame
        Index is `trade_date`, columns are the seven regime features.
        Rows can contain NaNs when one or both expiry buckets are unavailable.
    """
    trade_dates = sorted(options["trade_date"].unique())
    rows: list[dict[str, float | pd.Timestamp]] = []

    total_dates = len(trade_dates)

    for index, trade_date in enumerate(trade_dates, start=1):
        day_data = options.loc[options["trade_date"] == trade_date]

        near_dte = _select_expiry(day_data, near_dte_min, near_dte_target, near_dte_max)
        mid_dte = _select_expiry(day_data, mid_dte_min, mid_dte_target, mid_dte_max)

        row: dict[str, float | pd.Timestamp] = {"trade_date": trade_date}

        _assign_expiry_bucket_features(
            row=row,
            day_data=day_data,
            bucket_prefix="near",
            selected_dte=near_dte,
            wing_delta=wing_delta,
            min_strikes=min_strikes,
        )
        _assign_expiry_bucket_features(
            row=row,
            day_data=day_data,
            bucket_prefix="mid",
            selected_dte=mid_dte,
            wing_delta=wing_delta,
            min_strikes=min_strikes,
        )

        # term_slope = mid-term ATM IV - near-term ATM IV
        near_atm = float(row["atm_iv_near"])
        mid_atm = float(row["atm_iv_mid"])
        row["term_slope"] = mid_atm - near_atm

        rows.append(row)

        if index % 500 == 0:
            logger.info("Extracted features for %s/%s dates", index, total_dates)

    features = pd.DataFrame(rows).set_index("trade_date")

    valid_rows = features.dropna().shape[0]
    valid_ratio = valid_rows / len(features) if len(features) > 0 else 0.0
    logger.info(
        "Feature extraction complete: %s days, %s valid (%.1f%%)",
        len(features),
        valid_rows,
        100.0 * valid_ratio,
    )

    return features
