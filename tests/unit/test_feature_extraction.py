"""Tests for volatility surface feature extraction.

This test module validates numerical behavior for the core interpolation and
derived feature formulas used in the daily regime feature matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_chain(
    deltas: list[float], ivs: list[float], option_type: str
) -> pd.DataFrame:
    """Build a minimal one-expiry options chain for interpolation tests.

    Parameters
    ----------
    deltas : list[float]
        Option deltas.
    ivs : list[float]
        Mid implied volatility values aligned with `deltas`.
    option_type : str
        Option side identifier (`p` for puts, `c` for calls).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns `delta`, `mid_iv`, and `option_type`.
    """
    return pd.DataFrame(
        {
            "delta": deltas,
            "mid_iv": ivs,
            "option_type": [option_type] * len(deltas),
        }
    )


class TestInterpolateDelta:
    """Validate linear interpolation of implied volatility in delta space."""

    def test_exact_match(self) -> None:
        """Return exact implied volatility when target delta is present."""
        from volatility_regimes.features.surface import interpolate_iv_at_delta

        chain = _make_chain([-0.50, -0.40, -0.30], [0.20, 0.18, 0.16], "p")
        result = interpolate_iv_at_delta(chain, target_delta=-0.50)

        assert result == pytest.approx(0.20)

    def test_interpolation_between_strikes(self) -> None:
        """Interpolate linearly between bracketing deltas."""
        from volatility_regimes.features.surface import interpolate_iv_at_delta

        chain = _make_chain([-0.55, -0.45], [0.22, 0.18], "p")
        result = interpolate_iv_at_delta(chain, target_delta=-0.50)

        assert result == pytest.approx(0.20)

    def test_returns_nan_if_insufficient_data(self) -> None:
        """Return NaN when target delta is outside available range."""
        from volatility_regimes.features.surface import interpolate_iv_at_delta

        chain = _make_chain([-0.30, -0.20], [0.16, 0.14], "p")
        result = interpolate_iv_at_delta(chain, target_delta=-0.50)

        assert np.isnan(result)


class TestExtractDailyFeatures:
    """Validate signs and formulas for skew and butterfly calculations."""

    def test_skew_positive_in_normal_market(self) -> None:
        """Check skew sign for typical equity skew shape."""
        from volatility_regimes.features.surface import compute_skew

        put_chain = _make_chain([-0.30, -0.20], [0.23, 0.21], "p")
        call_chain = _make_chain([0.20, 0.30], [0.16, 0.14], "c")
        skew = compute_skew(put_chain, call_chain, wing_delta=0.25)

        assert skew > 0

    def test_butterfly_positive(self) -> None:
        """Check convex smile produces positive butterfly."""
        from volatility_regimes.features.surface import compute_butterfly

        butterfly = compute_butterfly(iv_25d_put=0.22, iv_25d_call=0.16, iv_atm=0.18)

        assert butterfly == pytest.approx(0.01)
        assert butterfly > 0
