"""Reporting helpers for walk-forward regime research outputs.

This module converts row-level forecast results into compact research tables
and a markdown summary that can be reviewed without opening the CSV files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def summarize_metrics(forecast_panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate out-of-sample forecast errors across research dimensions.

    Parameters
    ----------
    forecast_panel : pd.DataFrame
        Row-level forecast results. Required columns are `symbol`, `horizon`,
        `feature_set`, `model_name`, `prediction`, and `actual`.

    Returns
    -------
    pd.DataFrame
        Metric summary with one row per symbol, horizon, feature set, and
        model. Columns include root mean squared error (RMSE), mean absolute
        error (MAE), out-of-sample R-squared versus the ATM IV and historical
        mean benchmarks, and forecast count.
    """
    if forecast_panel.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "horizon",
                "feature_set",
                "model_name",
                "rmse",
                "mae",
                "oos_r_squared_vs_atm",
                "oos_r_squared_vs_historical_mean",
                "n_forecasts",
            ]
        )

    benchmark_mse: dict[str, dict[tuple[object, object, object], float]] = {}
    for benchmark_name in ("atm_iv", "historical_mean"):
        benchmark_mse[benchmark_name] = {}
        benchmark_rows = forecast_panel.loc[
            forecast_panel["model_name"] == benchmark_name
        ]
        for benchmark_key, benchmark_frame in benchmark_rows.groupby(
            ["symbol", "horizon", "feature_set"],
            sort=True,
        ):
            benchmark_actual = benchmark_frame["actual"].to_numpy(dtype=float)
            benchmark_prediction = benchmark_frame["prediction"].to_numpy(dtype=float)
            benchmark_residual = benchmark_actual - benchmark_prediction
            benchmark_mse[benchmark_name][benchmark_key] = float(
                np.mean(benchmark_residual**2)
            )

    summary_rows: list[dict[str, object]] = []
    group_columns = ["symbol", "horizon", "feature_set", "model_name"]

    for group_key, group_frame in forecast_panel.groupby(group_columns, sort=True):
        actual = group_frame["actual"].to_numpy(dtype=float)
        prediction = group_frame["prediction"].to_numpy(dtype=float)
        residual = actual - prediction
        mse = float(np.mean(residual**2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(residual)))

        atm_group_key = (group_key[0], group_key[1], group_key[2])
        relative_scores: dict[str, float] = {}
        for benchmark_name in ("atm_iv", "historical_mean"):
            group_benchmark_mse = benchmark_mse[benchmark_name].get(atm_group_key)
            if group_benchmark_mse is None:
                relative_scores[benchmark_name] = float(np.nan)
            elif group_benchmark_mse == 0.0:
                relative_scores[benchmark_name] = 0.0
            else:
                relative_scores[benchmark_name] = 1.0 - mse / group_benchmark_mse

        summary_rows.append(
            {
                "symbol": group_key[0],
                "horizon": int(group_key[1]),
                "feature_set": group_key[2],
                "model_name": group_key[3],
                "rmse": rmse,
                "mae": mae,
                "oos_r_squared_vs_atm": relative_scores["atm_iv"],
                "oos_r_squared_vs_historical_mean": relative_scores["historical_mean"],
                "n_forecasts": int(len(group_frame)),
            }
        )

    metric_summary = pd.DataFrame(summary_rows)
    metric_summary = metric_summary.sort_values(
        ["symbol", "horizon", "feature_set", "rmse", "model_name"],
        ignore_index=True,
    )
    return metric_summary


def write_research_summary(
    metric_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a markdown summary of the strongest research configurations.

    Parameters
    ----------
    metric_summary : pd.DataFrame
        Aggregated metric summary produced by `summarize_metrics`.
    output_path : Path
        Destination path for the markdown summary.
    """
    if metric_summary.empty:
        lines = [
            "# Walk-Forward Regime Research Summary",
            "",
            "No forecasts were generated, so no metric summary is available.",
        ]
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return

    best_rows = metric_summary.sort_values(
        ["rmse", "mae", "model_name"],
        ignore_index=True,
    ).head(10)

    lines = [
        "# Walk-Forward Regime Research Summary",
        "",
        "## Best Forecasting Configurations",
        "",
        best_rows.to_string(index=False),
        "",
        "## Notes",
        "",
        "- Lower RMSE indicates better out-of-sample realized-volatility forecasts.",
        "- OOS R^2 is measured relative to the ATM IV benchmark within the same symbol, horizon, and feature set.",
        "- OOS R^2 versus the historical mean tests whether model structure adds value beyond the expanding unconditional target mean.",
        "- Compare regime models against ATM IV, the historical mean, and linear-feature baselines before interpreting incremental value.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
