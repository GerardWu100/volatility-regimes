"""Integration tests for offline-first cache behavior in pipeline execution.

These tests exercise `main.run_pipeline_for_symbol` and verify that both
options and prices loaders respect the cache contract under cache-hit and
cache-miss conditions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _make_pipeline_prices(symbol_seed: int) -> pd.DataFrame:
    """Create synthetic daily close prices for one symbol."""
    rng = np.random.default_rng(symbol_seed)
    dates = pd.bdate_range("2020-01-02", periods=180)

    daily_log_returns = rng.normal(loc=0.0002, scale=0.01, size=len(dates))
    cumulative_log_returns = np.cumsum(daily_log_returns)
    close_values = 100.0 * np.exp(cumulative_log_returns)

    prices = pd.DataFrame({"date": dates, "close": close_values})
    return prices


def _make_pipeline_options(symbol_seed: int) -> pd.DataFrame:
    """Create synthetic options chain rows with near and mid expiries daily."""
    rng = np.random.default_rng(symbol_seed)
    trade_dates = pd.bdate_range("2020-01-02", periods=180)

    put_deltas = [-0.60, -0.50, -0.40, -0.30, -0.20]
    call_deltas = [0.20, 0.30, 0.40, 0.50, 0.60]
    expiries = [30, 90]

    rows: list[dict[str, object]] = []
    for date_index, trade_date in enumerate(trade_dates):
        regime_block = (date_index // 45) % 4
        base_atm_iv = 0.14 + 0.04 * regime_block

        for dte_value in expiries:
            term_premium = 0.02 if dte_value == 90 else 0.0
            wing_skew = 0.07 + 0.01 * regime_block
            butterfly = 0.015 + 0.003 * regime_block

            expiry_date = trade_date + pd.Timedelta(days=dte_value)

            for delta_value in put_deltas:
                abs_delta = abs(delta_value)
                skew_component = wing_skew * (0.50 - abs_delta)
                curvature_component = butterfly * (abs_delta - 0.50) ** 2 * 12.0
                noise = rng.normal(loc=0.0, scale=0.002)
                mid_iv = (
                    base_atm_iv
                    + term_premium
                    + skew_component
                    + curvature_component
                    + noise
                )

                rows.append(
                    {
                        "trade_date": trade_date,
                        "expiry_date": expiry_date,
                        "option_type": "p",
                        "strike_price": 1000.0 + 100.0 * abs_delta,
                        "mid_iv": float(mid_iv),
                        "delta": float(delta_value),
                        "volume": 100 + int(abs_delta * 10),
                        "open_interest": 1000 + int(abs_delta * 100),
                        "dte": dte_value,
                    }
                )

            for delta_value in call_deltas:
                abs_delta = abs(delta_value)
                skew_component = -wing_skew * (0.50 - abs_delta)
                curvature_component = butterfly * (abs_delta - 0.50) ** 2 * 12.0
                noise = rng.normal(loc=0.0, scale=0.002)
                mid_iv = (
                    base_atm_iv
                    + term_premium
                    + skew_component
                    + curvature_component
                    + noise
                )

                rows.append(
                    {
                        "trade_date": trade_date,
                        "expiry_date": expiry_date,
                        "option_type": "c",
                        "strike_price": 1000.0 + 100.0 * abs_delta,
                        "mid_iv": float(mid_iv),
                        "delta": float(delta_value),
                        "volume": 100 + int(abs_delta * 10),
                        "open_interest": 1000 + int(abs_delta * 100),
                        "dte": dte_value,
                    }
                )

    options = pd.DataFrame(rows)
    return options


def _write_cache_files(
    cache_dir: Path,
    dataset_name: str,
    symbol: str,
    dataframe: pd.DataFrame,
    date_column: str,
) -> None:
    """Write dataset Parquet and metadata sidecar for integration tests."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = cache_dir / f"{dataset_name}_{symbol.lower()}.parquet"
    metadata_path = cache_dir / f"{dataset_name}_{symbol.lower()}.metadata.json"

    dataframe.to_parquet(parquet_path, index=False)

    metadata = {
        "metadata_version": 1,
        "dataset": dataset_name,
        "symbol": symbol,
        "row_count": len(dataframe),
        "date_coverage": {
            "start": str(pd.to_datetime(dataframe[date_column].min()).date()),
            "end": str(pd.to_datetime(dataframe[date_column].max()).date()),
        },
        "schema": {column: str(dtype) for column, dtype in dataframe.dtypes.items()},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))


def _pipeline_config() -> dict[str, object]:
    """Build a compact integration config for pipeline tests."""
    return {
        "data": {
            "symbols": ["SPX"],
            "start_date": "2020-01-02",
            "end_date": "2020-09-09",
        },
        "cache": {
            "mode": "offline_first",
            "required_dir": "data/raw",
            "metadata_version": 1,
            "require_full_date_coverage": True,
        },
        "features": {
            "delta_min": 0.05,
            "delta_max": 0.95,
            "near_term_dte_min": 15,
            "near_term_dte_target": 30,
            "near_term_dte_max": 45,
            "mid_term_dte_min": 45,
            "mid_term_dte_target": 90,
            "mid_term_dte_max": 120,
            "atm_delta": 0.50,
            "wing_delta": 0.25,
            "min_strikes_per_side": 5,
        },
        "regime": {
            "min_k": 2,
            "max_k": 3,
            "hmm_n_iter": 20,
            "hmm_random_restarts": 2,
        },
        "analysis": {
            "realized_vol_window": 20,
            "annualization_factor": 252,
        },
    }


def test_pipeline_cache_hit_runs_with_db_down(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pipeline succeeds from cache while ClickHouse query functions are down."""
    import volatility_regimes.data_access.loader as data_loader
    import volatility_regimes.pipelines.descriptive_pipeline as main
    from volatility_regimes.descriptive import plotting

    options = _make_pipeline_options(symbol_seed=7)
    prices = _make_pipeline_prices(symbol_seed=13)

    cache_dir = tmp_path / "data/raw"
    _write_cache_files(cache_dir, "options", "SPX", options, "trade_date")
    _write_cache_files(cache_dir, "prices", "SPX", prices, "date")

    monkeypatch.setattr(data_loader, "PROJECT_ROOT", tmp_path)

    def _raise_if_called(*_: object, **__: object) -> pd.DataFrame:
        raise AssertionError("ClickHouse should not be called on pipeline cache hit")

    monkeypatch.setattr(data_loader, "_query_options_from_clickhouse", _raise_if_called)
    monkeypatch.setattr(data_loader, "_query_prices_from_clickhouse", _raise_if_called)

    test_reports_dir = tmp_path / "outputs" / "reports" / "descriptive"
    test_figures_dir = tmp_path / "outputs" / "figures" / "descriptive"
    test_reports_dir.mkdir(parents=True, exist_ok=True)
    test_figures_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "REPORTS_DESCRIPTIVE_DIR", test_reports_dir)
    monkeypatch.setattr(plotting, "FIGURES_DESCRIPTIVE_DIR", test_figures_dir)

    config = _pipeline_config()
    result = main.run_pipeline_for_symbol("SPX", config)

    assert int(result["best_k"]) in {2, 3}


def test_pipeline_cache_miss_queries_db_and_writes_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pipeline uses DB query path on cache miss and writes both cache datasets."""
    import volatility_regimes.data_access.loader as data_loader
    import volatility_regimes.pipelines.descriptive_pipeline as main
    from volatility_regimes.descriptive import plotting

    options = _make_pipeline_options(symbol_seed=21)
    prices = _make_pipeline_prices(symbol_seed=29)

    monkeypatch.setattr(data_loader, "PROJECT_ROOT", tmp_path)

    option_query_count = {"count": 0}
    prices_query_count = {"count": 0}

    def _fake_options_query(*_: object, **__: object) -> pd.DataFrame:
        option_query_count["count"] += 1
        return options.copy()

    def _fake_prices_query(*_: object, **__: object) -> pd.DataFrame:
        prices_query_count["count"] += 1
        return prices.copy()

    monkeypatch.setattr(
        data_loader, "_query_options_from_clickhouse", _fake_options_query
    )
    monkeypatch.setattr(
        data_loader, "_query_prices_from_clickhouse", _fake_prices_query
    )

    test_reports_dir = tmp_path / "outputs" / "reports" / "descriptive"
    test_figures_dir = tmp_path / "outputs" / "figures" / "descriptive"
    test_reports_dir.mkdir(parents=True, exist_ok=True)
    test_figures_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "REPORTS_DESCRIPTIVE_DIR", test_reports_dir)
    monkeypatch.setattr(plotting, "FIGURES_DESCRIPTIVE_DIR", test_figures_dir)

    config = _pipeline_config()
    result = main.run_pipeline_for_symbol("SPX", config)

    assert option_query_count["count"] == 1
    assert prices_query_count["count"] == 1
    assert int(result["best_k"]) in {2, 3}

    cache_dir = tmp_path / "data/raw"
    assert (cache_dir / "options_spx.parquet").exists()
    assert (cache_dir / "options_spx.metadata.json").exists()
    assert (cache_dir / "prices_spx.parquet").exists()
    assert (cache_dir / "prices_spx.metadata.json").exists()
