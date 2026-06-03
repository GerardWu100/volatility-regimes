"""Tests for deterministic regime feature-set selection.

These tests pin the registry contents and column order so feature subsets stay
stable across walk-forward training runs.
"""

from __future__ import annotations

import pandas as pd
import pytest


def test_feature_set_registry_and_selector_preserve_order() -> None:
    """Select a named feature set in registry order, not input order."""
    from volatility_regimes.features.surface import (
        FEATURE_SET_REGISTRY,
        select_feature_columns,
    )

    features = pd.DataFrame(
        {
            "term_slope": [0.03],
            "butterfly_mid": [0.01],
            "atm_iv_mid": [0.22],
            "skew_mid": [0.04],
            "atm_iv_near": [0.20],
            "skew_near": [0.05],
            "butterfly_near": [0.02],
        }
    )

    assert FEATURE_SET_REGISTRY == {
        "full": (
            "atm_iv_near",
            "atm_iv_mid",
            "skew_near",
            "skew_mid",
            "butterfly_near",
            "butterfly_mid",
            "term_slope",
        ),
        "atm_only": ("atm_iv_near",),
        "atm_term": ("atm_iv_near", "atm_iv_mid", "term_slope"),
        "atm_skew": ("atm_iv_near", "atm_iv_mid", "skew_near", "skew_mid"),
        "near_only": ("atm_iv_near", "skew_near", "butterfly_near"),
    }

    full_features = select_feature_columns(features, "full")
    atm_only_features = select_feature_columns(features, "atm_only")
    atm_skew_features = select_feature_columns(features, "atm_skew")

    assert list(full_features.columns) == [
        "atm_iv_near",
        "atm_iv_mid",
        "skew_near",
        "skew_mid",
        "butterfly_near",
        "butterfly_mid",
        "term_slope",
    ]
    assert list(atm_only_features.columns) == ["atm_iv_near"]
    assert list(atm_skew_features.columns) == [
        "atm_iv_near",
        "atm_iv_mid",
        "skew_near",
        "skew_mid",
    ]
    assert atm_skew_features.iloc[0].to_dict() == {
        "atm_iv_near": 0.20,
        "atm_iv_mid": 0.22,
        "skew_near": 0.05,
        "skew_mid": 0.04,
    }


def test_select_feature_columns_rejects_unknown_name() -> None:
    """Reject unsupported feature-set names with a clear error."""
    from volatility_regimes.features.surface import select_feature_columns

    features = pd.DataFrame({"atm_iv_near": [0.20]})

    with pytest.raises(ValueError, match="Unknown feature set"):
        select_feature_columns(features, "unknown")
