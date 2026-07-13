"""Unit tests for walk-forward forecast reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volatility_regimes.walkforward.reporting import summarize_metrics


def test_summarize_metrics_scores_models_against_both_benchmarks() -> None:
    """Verify relative scores use each benchmark's mean squared error.

    The two observations have actual values 1 and 3. The ATM benchmark's
    mean squared error is 5, the historical mean's is 1, and the candidate
    model's is 0.5. Its expected relative scores are therefore 0.9 and 0.5.
    """
    common = {
        "symbol": "SPX",
        "horizon": 20,
        "feature_set": "atm_term",
    }
    rows: list[dict[str, object]] = []
    for model_name, predictions in {
        "atm_iv": (0.0, 0.0),
        "historical_mean": (2.0, 2.0),
        "candidate": (1.0, 2.0),
    }.items():
        for actual, prediction in zip((1.0, 3.0), predictions, strict=True):
            rows.append(
                {
                    **common,
                    "model_name": model_name,
                    "prediction": prediction,
                    "actual": actual,
                }
            )

    summary = summarize_metrics(pd.DataFrame(rows)).set_index("model_name")

    assert summary.loc["candidate", "oos_r_squared_vs_atm"] == pytest.approx(0.9)
    assert summary.loc[
        "candidate", "oos_r_squared_vs_historical_mean"
    ] == pytest.approx(0.5)
    assert summary.loc[
        "historical_mean", "oos_r_squared_vs_historical_mean"
    ] == pytest.approx(0.0)


def test_summarize_metrics_returns_nan_without_benchmark_rows() -> None:
    """Keep relative scores undefined when the required benchmark is absent."""
    panel = pd.DataFrame(
        {
            "symbol": ["SPX"],
            "horizon": [20],
            "feature_set": ["atm_term"],
            "model_name": ["candidate"],
            "prediction": [0.2],
            "actual": [0.3],
        }
    )

    summary = summarize_metrics(panel)

    assert np.isnan(summary.loc[0, "oos_r_squared_vs_atm"])
    assert np.isnan(summary.loc[0, "oos_r_squared_vs_historical_mean"])
