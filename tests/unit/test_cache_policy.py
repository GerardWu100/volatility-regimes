"""Tests for offline-first Parquet cache behavior in data loading.

This module verifies the cache contract for options and daily price data:
- valid cache is used without a ClickHouse query,
- missing/invalid cache falls back to ClickHouse and refreshes cache,
- missing/invalid cache with unavailable ClickHouse fails with an actionable error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _make_options_dataframe() -> pd.DataFrame:
    """Build a minimal, schema-correct options DataFrame for cache tests."""
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "expiry_date": pd.to_datetime(["2020-01-31", "2020-01-31"]),
            "option_type": ["p", "c"],
            "strike_price": [3200.0, 3200.0],
            "mid_iv": [0.20, 0.19],
            "delta": [-0.50, 0.50],
            "volume": [100, 120],
            "open_interest": [1000, 1100],
            "dte": [29, 29],
        }
    )


def _build_cache_config() -> dict[str, object]:
    """Return cache configuration used in tests."""
    return {
        "required_dir": "data/raw",
        "metadata_version": 1,
        "require_full_date_coverage": True,
        "mode": "offline_first",
    }


def _write_options_cache(
    cache_dir: Path,
    symbol: str,
    dataframe: pd.DataFrame,
    metadata_version: int,
    row_count_override: int | None = None,
) -> tuple[Path, Path]:
    """Write options Parquet + metadata sidecar files for one symbol."""
    parquet_path = cache_dir / f"options_{symbol.lower()}.parquet"
    metadata_path = cache_dir / f"options_{symbol.lower()}.metadata.json"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataframe.to_parquet(parquet_path, index=False)

    row_count = len(dataframe)
    if row_count_override is not None:
        row_count = row_count_override

    metadata = {
        "metadata_version": metadata_version,
        "dataset": "options",
        "symbol": symbol,
        "row_count": row_count,
        "date_coverage": {
            "start": str(dataframe["trade_date"].min().date()),
            "end": str(dataframe["trade_date"].max().date()),
        },
        "schema": {column: str(dtype) for column, dtype in dataframe.dtypes.items()},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return parquet_path, metadata_path


def test_cache_hit_uses_parquet_without_db_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Load from valid Parquet cache and never call ClickHouse."""
    import volatility_regimes.data_access.loader as data_loader

    cache_config = _build_cache_config()
    cache_dir = tmp_path / "data/raw"
    options_df = _make_options_dataframe()

    _write_options_cache(cache_dir, "SPX", options_df, metadata_version=1)

    monkeypatch.setattr(data_loader, "PROJECT_ROOT", tmp_path)

    def _raise_if_called(*_: object, **__: object) -> pd.DataFrame:
        raise AssertionError("ClickHouse query should not be called on cache hit")

    monkeypatch.setattr(data_loader, "_query_options_from_clickhouse", _raise_if_called)

    loaded = data_loader.load_options(
        symbol="SPX",
        start_date="2020-01-02",
        end_date="2020-01-03",
        delta_min=0.05,
        delta_max=0.95,
        cache_config=cache_config,
    )

    pd.testing.assert_frame_equal(loaded.reset_index(drop=True), options_df)


def test_cache_miss_queries_db_and_writes_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Use ClickHouse on cache miss and then write Parquet + metadata."""
    import volatility_regimes.data_access.loader as data_loader

    cache_config = _build_cache_config()
    cache_dir = tmp_path / "data/raw"
    options_df = _make_options_dataframe()

    monkeypatch.setattr(data_loader, "PROJECT_ROOT", tmp_path)

    call_counter = {"count": 0}

    def _fake_clickhouse_query(*_: object, **__: object) -> pd.DataFrame:
        call_counter["count"] += 1
        return options_df.copy()

    monkeypatch.setattr(
        data_loader,
        "_query_options_from_clickhouse",
        _fake_clickhouse_query,
    )

    loaded = data_loader.load_options(
        symbol="SPX",
        start_date="2020-01-02",
        end_date="2020-01-03",
        delta_min=0.05,
        delta_max=0.95,
        cache_config=cache_config,
    )

    assert call_counter["count"] == 1
    pd.testing.assert_frame_equal(loaded.reset_index(drop=True), options_df)

    parquet_path = cache_dir / "options_spx.parquet"
    metadata_path = cache_dir / "options_spx.metadata.json"
    assert parquet_path.exists()
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text())
    assert metadata["row_count"] == len(options_df)
    assert metadata["metadata_version"] == 1


def test_cache_invalid_queries_db_and_refreshes_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Treat invalid sidecar as cache miss and rewrite cache from ClickHouse."""
    import volatility_regimes.data_access.loader as data_loader

    cache_config = _build_cache_config()
    cache_dir = tmp_path / "data/raw"
    options_df = _make_options_dataframe()

    _write_options_cache(
        cache_dir,
        "SPX",
        options_df,
        metadata_version=999,
        row_count_override=12345,
    )

    monkeypatch.setattr(data_loader, "PROJECT_ROOT", tmp_path)

    call_counter = {"count": 0}

    def _fake_clickhouse_query(*_: object, **__: object) -> pd.DataFrame:
        call_counter["count"] += 1
        return options_df.copy()

    monkeypatch.setattr(
        data_loader,
        "_query_options_from_clickhouse",
        _fake_clickhouse_query,
    )

    loaded = data_loader.load_options(
        symbol="SPX",
        start_date="2020-01-02",
        end_date="2020-01-03",
        delta_min=0.05,
        delta_max=0.95,
        cache_config=cache_config,
    )

    assert call_counter["count"] == 1
    pd.testing.assert_frame_equal(loaded.reset_index(drop=True), options_df)

    metadata_path = cache_dir / "options_spx.metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["metadata_version"] == 1
    assert metadata["row_count"] == len(options_df)


def test_cache_miss_and_db_down_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Raise a clear error with required filenames when DB is unavailable."""
    import volatility_regimes.data_access.loader as data_loader

    cache_config = _build_cache_config()
    monkeypatch.setattr(data_loader, "PROJECT_ROOT", tmp_path)

    def _raise_db_down(*_: object, **__: object) -> pd.DataFrame:
        raise RuntimeError("connection refused")

    monkeypatch.setattr(data_loader, "_query_options_from_clickhouse", _raise_db_down)

    with pytest.raises(RuntimeError) as error_info:
        data_loader.load_options(
            symbol="SPX",
            start_date="2020-01-02",
            end_date="2020-01-03",
            delta_min=0.05,
            delta_max=0.95,
            cache_config=cache_config,
        )

    message = str(error_info.value)
    assert "options_spx.parquet" in message
    assert "options_spx.metadata.json" in message
    assert str(tmp_path / "data/raw") in message
    assert "ClickHouse unavailable" in message


def test_cache_invalid_and_db_down_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Raise clear error when invalid cache cannot be refreshed from DB."""
    import volatility_regimes.data_access.loader as data_loader

    cache_config = _build_cache_config()
    cache_dir = tmp_path / "data/raw"
    options_df = _make_options_dataframe()

    _write_options_cache(
        cache_dir,
        "SPX",
        options_df,
        metadata_version=999,
        row_count_override=9999,
    )

    monkeypatch.setattr(data_loader, "PROJECT_ROOT", tmp_path)

    def _raise_db_down(*_: object, **__: object) -> pd.DataFrame:
        raise RuntimeError("connection refused")

    monkeypatch.setattr(data_loader, "_query_options_from_clickhouse", _raise_db_down)

    with pytest.raises(RuntimeError) as error_info:
        data_loader.load_options(
            symbol="SPX",
            start_date="2020-01-02",
            end_date="2020-01-03",
            delta_min=0.05,
            delta_max=0.95,
            cache_config=cache_config,
        )

    message = str(error_info.value)
    assert "options_spx.parquet" in message
    assert "options_spx.metadata.json" in message
    assert "cache is missing/invalid" in message
    assert "ClickHouse unavailable" in message
