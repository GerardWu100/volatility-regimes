"""Load and clean options and index data from ClickHouse.

This module is the data-access boundary for the volatility regime project.
It reads connection credentials from a local `.env` file and returns cleaned
Pandas DataFrames for downstream feature extraction.

Notes
-----
- Options source table: `firstrate.options`.
- Index prices source table: `firstrate.indices`.
- SPX/SPXW duplicate rows are resolved by selecting the contract with larger
  open interest for each (trade_date, expiry_date, option_type, strike_price).

The module also supports an offline-first cache policy backed by Parquet files
and JSON metadata sidecars.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import clickhouse_connect
import pandas as pd
from clickhouse_connect.driver import Client

logger = logging.getLogger(__name__)

from volatility_regimes.project_paths import PROJECT_ROOT

OPTIONS_CACHE_COLUMNS = [
    "trade_date",
    "expiry_date",
    "option_type",
    "strike_price",
    "mid_iv",
    "delta",
    "volume",
    "open_interest",
    "dte",
]

PRICES_CACHE_COLUMNS = ["date", "close"]

OPTIONS_DTYPE_ALLOWLIST = {
    "trade_date": {"datetime64[ns]", "datetime64[us]"},
    "expiry_date": {"datetime64[ns]", "datetime64[us]"},
    "option_type": {"object", "str", "string", "string[python]"},
    "strike_price": {"float64"},
    "mid_iv": {"float64"},
    "delta": {"float64"},
    "volume": {"int64"},
    "open_interest": {"int64"},
    "dte": {"int64"},
}

PRICES_DTYPE_ALLOWLIST = {
    "date": {"datetime64[ns]", "datetime64[us]"},
    "close": {"float64"},
}


def _load_env() -> dict[str, str]:
    """Parse local `.env` file into a dictionary.

    Returns
    -------
    dict[str, str]
        Mapping from environment variable name to value.

    Raises
    ------
    FileNotFoundError
        If `.env` does not exist in the project root.
    """
    env_path = PROJECT_ROOT / ".env"
    env_vars: dict[str, str] = {}

    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue

        key, separator, value = stripped.partition("=")
        if not separator:
            continue

        env_vars[key.strip()] = value.strip()

    return env_vars


def get_client() -> Client:
    """Create ClickHouse client from `.env` credentials.

    Returns
    -------
    Client
        Configured ClickHouse client.
    """
    env = _load_env()

    return clickhouse_connect.get_client(
        host=env["CLICKHOUSE_HOST"],
        port=int(env["CLICKHOUSE_PORT"]),
        username=env["CLICKHOUSE_USER"],
        password=env["CLICKHOUSE_PASSWORD"],
        secure=env.get("CLICKHOUSE_SECURE", "false").lower() == "true",
        verify=env.get("CLICKHOUSE_VERIFY", "false").lower() == "true",
    )


def _cache_root_dir(cache_config: dict[str, object]) -> Path:
    """Resolve cache root directory from project-relative config path."""
    relative_path = str(cache_config["required_dir"])
    return PROJECT_ROOT / relative_path


def _cache_paths(
    cache_root: Path,
    dataset_name: str,
    symbol: str,
) -> tuple[Path, Path]:
    """Build Parquet and metadata sidecar paths for one symbol dataset."""
    symbol_key = symbol.lower()
    parquet_path = cache_root / f"{dataset_name}_{symbol_key}.parquet"
    metadata_path = cache_root / f"{dataset_name}_{symbol_key}.metadata.json"
    return parquet_path, metadata_path


def _schema_signature(dataframe: pd.DataFrame) -> dict[str, str]:
    """Serialize DataFrame schema as column->dtype string mapping."""
    return {column: str(dtype) for column, dtype in dataframe.dtypes.items()}


def _to_iso_date(date_value: pd.Timestamp) -> str:
    """Convert pandas timestamp to YYYY-MM-DD text for sidecar metadata."""
    return date_value.date().isoformat()


def _date_coverage(
    dataframe: pd.DataFrame,
    date_column: str,
) -> dict[str, str]:
    """Compute inclusive date coverage for cached data."""
    min_date = pd.to_datetime(dataframe[date_column].min())
    max_date = pd.to_datetime(dataframe[date_column].max())
    return {
        "start": _to_iso_date(min_date),
        "end": _to_iso_date(max_date),
    }


def _validate_metadata_version(
    metadata: dict[str, Any],
    expected_version: int,
) -> tuple[bool, str]:
    """Check that sidecar metadata version matches config expectations."""
    metadata_version = metadata.get("metadata_version")
    if metadata_version != expected_version:
        return False, (
            f"metadata_version mismatch expected={expected_version} "
            f"found={metadata_version}"
        )
    return True, "ok"


def _validate_schema(
    metadata: dict[str, Any],
    dataframe: pd.DataFrame,
    required_columns: list[str],
    dtype_allowlist: dict[str, set[str]],
) -> tuple[bool, str]:
    """Validate expected schema against sidecar and loaded DataFrame."""
    sidecar_schema = metadata.get("schema")
    if not isinstance(sidecar_schema, dict):
        return False, "schema missing from metadata sidecar"

    if sorted(sidecar_schema) != sorted(required_columns):
        return False, "schema columns mismatch in metadata sidecar"

    actual_columns = dataframe.columns.tolist()
    if actual_columns != required_columns:
        return False, "schema columns mismatch in parquet data"

    actual_schema = _schema_signature(dataframe)
    if actual_schema != sidecar_schema:
        return False, f"schema mismatch in parquet data actual={actual_schema}"

    for column_name in required_columns:
        sidecar_dtype = str(sidecar_schema[column_name])
        allowed_dtypes = dtype_allowlist[column_name]
        if sidecar_dtype not in allowed_dtypes:
            allowed_as_text = sorted(allowed_dtypes)
            return False, (
                f"unsupported dtype for {column_name} "
                f"found={sidecar_dtype} allowed={allowed_as_text}"
            )

    return True, "ok"


def _validate_row_count(
    metadata: dict[str, Any],
    dataframe: pd.DataFrame,
) -> tuple[bool, str]:
    """Validate row count consistency between sidecar and parquet."""
    sidecar_row_count = metadata.get("row_count")
    actual_row_count = len(dataframe)
    if sidecar_row_count != actual_row_count:
        return False, (
            f"row_count mismatch sidecar={sidecar_row_count} parquet={actual_row_count}"
        )
    return True, "ok"


def _validate_date_coverage(
    metadata: dict[str, Any],
    dataframe: pd.DataFrame,
    date_column: str,
    requested_start_date: str,
    requested_end_date: str,
    require_full_date_coverage: bool,
) -> tuple[bool, str]:
    """Validate sidecar coverage and optional full coverage of request window."""
    metadata_coverage = metadata.get("date_coverage")
    if not isinstance(metadata_coverage, dict):
        return False, "date_coverage missing from metadata sidecar"

    metadata_start = metadata_coverage.get("start")
    metadata_end = metadata_coverage.get("end")
    if not isinstance(metadata_start, str):
        return False, "date_coverage.start missing from metadata sidecar"
    if not isinstance(metadata_end, str):
        return False, "date_coverage.end missing from metadata sidecar"

    actual_coverage = _date_coverage(dataframe, date_column)
    if metadata_start != actual_coverage["start"]:
        return False, "date_coverage.start mismatch between sidecar and parquet"
    if metadata_end != actual_coverage["end"]:
        return False, "date_coverage.end mismatch between sidecar and parquet"

    if not require_full_date_coverage:
        return True, "ok"

    requested_start = pd.Timestamp(requested_start_date)
    requested_end = pd.Timestamp(requested_end_date)
    cached_start = pd.Timestamp(metadata_start)
    cached_end = pd.Timestamp(metadata_end)

    if cached_start > requested_start:
        return False, (
            "cache date coverage starts too late "
            f"cached_start={metadata_start} requested_start={requested_start_date}"
        )
    if cached_end < requested_end:
        return False, (
            "cache date coverage ends too early "
            f"cached_end={metadata_end} requested_end={requested_end_date}"
        )

    return True, "ok"


def _build_cache_validation_result(
    metadata: dict[str, Any],
    dataframe: pd.DataFrame,
    required_columns: list[str],
    dtype_allowlist: dict[str, set[str]],
    requested_start_date: str,
    requested_end_date: str,
    date_column: str,
    metadata_version: int,
    require_full_date_coverage: bool,
) -> tuple[bool, str]:
    """Validate sidecar and parquet against policy contract."""
    checks = [
        _validate_metadata_version(metadata, metadata_version),
        _validate_schema(metadata, dataframe, required_columns, dtype_allowlist),
        _validate_row_count(metadata, dataframe),
        _validate_date_coverage(
            metadata,
            dataframe,
            date_column,
            requested_start_date,
            requested_end_date,
            require_full_date_coverage,
        ),
    ]

    for is_valid, reason in checks:
        if not is_valid:
            return False, reason

    return True, "ok"


def _load_cache_if_valid(
    dataset_name: str,
    symbol: str,
    requested_start_date: str,
    requested_end_date: str,
    required_columns: list[str],
    dtype_allowlist: dict[str, set[str]],
    date_column: str,
    cache_config: dict[str, object],
) -> tuple[pd.DataFrame | None, str, tuple[Path, Path]]:
    """Load cached dataset only when both parquet and sidecar pass checks."""
    cache_root = _cache_root_dir(cache_config)
    parquet_path, metadata_path = _cache_paths(cache_root, dataset_name, symbol)

    if not parquet_path.exists():
        return (
            None,
            f"missing parquet {parquet_path.name}",
            (parquet_path, metadata_path),
        )
    if not metadata_path.exists():
        return (
            None,
            f"missing metadata {metadata_path.name}",
            (parquet_path, metadata_path),
        )

    try:
        metadata_raw = metadata_path.read_text()
        metadata = json.loads(metadata_raw)
    except Exception as error:  # noqa: BLE001
        return (
            None,
            f"invalid metadata sidecar {metadata_path.name}: {error}",
            (parquet_path, metadata_path),
        )

    try:
        cached_dataframe = pd.read_parquet(parquet_path)
    except Exception as error:  # noqa: BLE001
        return (
            None,
            f"invalid parquet cache {parquet_path.name}: {error}",
            (parquet_path, metadata_path),
        )

    is_valid, reason = _build_cache_validation_result(
        metadata=metadata,
        dataframe=cached_dataframe,
        required_columns=required_columns,
        dtype_allowlist=dtype_allowlist,
        requested_start_date=requested_start_date,
        requested_end_date=requested_end_date,
        date_column=date_column,
        metadata_version=int(cache_config["metadata_version"]),
        require_full_date_coverage=bool(cache_config["require_full_date_coverage"]),
    )
    if not is_valid:
        return None, reason, (parquet_path, metadata_path)

    filtered_dataframe = cached_dataframe.loc[
        cached_dataframe[date_column].between(requested_start_date, requested_end_date)
    ].copy()
    logger.info(
        "Loaded %s rows from cache for %s (%s)",
        f"{len(filtered_dataframe):,}",
        symbol,
        dataset_name,
    )
    return filtered_dataframe, "cache-hit", (parquet_path, metadata_path)


def _write_cache(
    dataframe: pd.DataFrame,
    dataset_name: str,
    symbol: str,
    date_column: str,
    cache_config: dict[str, object],
) -> tuple[Path, Path]:
    """Write parquet cache and metadata sidecar after a DB query."""
    cache_root = _cache_root_dir(cache_config)
    cache_root.mkdir(parents=True, exist_ok=True)

    parquet_path, metadata_path = _cache_paths(cache_root, dataset_name, symbol)

    dataframe.to_parquet(parquet_path, index=False)

    metadata = {
        "metadata_version": int(cache_config["metadata_version"]),
        "dataset": dataset_name,
        "symbol": symbol,
        "row_count": len(dataframe),
        "date_coverage": _date_coverage(dataframe, date_column),
        "schema": _schema_signature(dataframe),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    logger.info(
        "Wrote cache files for %s (%s): %s and %s",
        symbol,
        dataset_name,
        parquet_path.name,
        metadata_path.name,
    )
    return parquet_path, metadata_path


def _offline_error_message(
    dataset_name: str,
    symbol: str,
    cache_reason: str,
    cache_paths: tuple[Path, Path],
    cache_root: Path,
    start_date: str,
    end_date: str,
    db_error: Exception,
) -> str:
    """Build actionable offline-first failure message with exact file paths."""
    parquet_path, metadata_path = cache_paths
    return (
        "Offline-first data load failed: cache is missing/invalid and ClickHouse "
        "unavailable. "
        f"dataset={dataset_name} symbol={symbol} start_date={start_date} "
        f"end_date={end_date}. cache_reason={cache_reason}. "
        f"Required cache directory: {cache_root}. "
        f"Required files: {parquet_path.name}, {metadata_path.name}. "
        f"ClickHouse unavailable: {db_error}."
    )


def _normalize_options_dataframe(options: pd.DataFrame) -> pd.DataFrame:
    """Coerce options data into deterministic schema and ordering."""
    options = options.copy()

    options["trade_date"] = pd.to_datetime(options["trade_date"])
    options["expiry_date"] = pd.to_datetime(options["expiry_date"])
    options["option_type"] = options["option_type"].astype(str)
    options["strike_price"] = options["strike_price"].astype(float)
    options["mid_iv"] = options["mid_iv"].astype(float)
    options["delta"] = options["delta"].astype(float)
    options["volume"] = options["volume"].astype("int64")
    options["open_interest"] = options["open_interest"].astype("int64")
    options["dte"] = options["dte"].astype("int64")

    group_columns = ["trade_date", "expiry_date", "option_type", "strike_price"]
    ascending_flags = [True, True, True, True, False]
    options = options.sort_values(
        group_columns + ["open_interest"], ascending=ascending_flags
    )
    options = options.drop_duplicates(subset=group_columns, keep="first")
    options = options.sort_values(group_columns).reset_index(drop=True)

    ordered_options = options.loc[:, OPTIONS_CACHE_COLUMNS]
    return ordered_options


def _normalize_prices_dataframe(prices: pd.DataFrame) -> pd.DataFrame:
    """Coerce prices data into deterministic schema and ordering."""
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices["close"] = prices["close"].astype(float)
    prices = prices.sort_values("date").reset_index(drop=True)
    ordered_prices = prices.loc[:, PRICES_CACHE_COLUMNS]
    return ordered_prices


def _query_options_from_clickhouse(
    symbol: str,
    start_date: str,
    end_date: str,
    delta_min: float,
    delta_max: float,
) -> pd.DataFrame:
    """Fetch options data from ClickHouse and apply canonical normalization."""
    client = get_client()

    query = """
        SELECT
            trade_date,
            expiry_date,
            option_type,
            strike_price,
            (bid_iv + ask_iv) / 2 AS mid_iv,
            delta,
            volume,
            open_interest,
            dateDiff('day', trade_date, expiry_date) AS dte
        FROM firstrate.options
        WHERE symbol = {symbol:String}
          AND trade_date BETWEEN {start:String} AND {end:String}
          AND bid_iv > 0
          AND ask_iv > 0
          AND abs(delta) BETWEEN {d_min:Float64} AND {d_max:Float64}
        ORDER BY trade_date, expiry_date, option_type, strike_price
    """

    result = client.query(
        query,
        parameters={
            "symbol": symbol,
            "start": start_date,
            "end": end_date,
            "d_min": delta_min,
            "d_max": delta_max,
        },
    )

    options = pd.DataFrame(result.result_rows, columns=result.column_names)
    logger.info("Loaded %s raw option rows for %s", f"{len(options):,}", symbol)

    normalized_options = _normalize_options_dataframe(options)
    logger.info(
        "After deduplication %s rows for %s",
        f"{len(normalized_options):,}",
        symbol,
    )
    return normalized_options


def _query_prices_from_clickhouse(
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch daily prices from ClickHouse and apply canonical normalization."""
    client = get_client()

    query = """
        SELECT
            toDate(ts) AS date,
            argMax(close, ts) AS close
        FROM firstrate.indices
        WHERE symbol = {symbol:String}
          AND toDate(ts) BETWEEN {start:String} AND {end:String}
        GROUP BY date
        ORDER BY date
    """

    result = client.query(
        query,
        parameters={"symbol": symbol, "start": start_date, "end": end_date},
    )

    prices = pd.DataFrame(result.result_rows, columns=result.column_names)
    normalized_prices = _normalize_prices_dataframe(prices)

    logger.info("Loaded %s daily prices for %s", f"{len(normalized_prices):,}", symbol)
    return normalized_prices


def _load_dataset_with_cache(
    dataset_name: str,
    symbol: str,
    start_date: str,
    end_date: str,
    required_columns: list[str],
    dtype_allowlist: dict[str, set[str]],
    date_column: str,
    cache_config: dict[str, object] | None,
    normalize: Callable[[pd.DataFrame], pd.DataFrame],
    query_from_clickhouse: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    """Shared offline-first load path for options and daily prices.

    When ``cache_config`` is set, valid Parquet + sidecar files are preferred.
    On cache miss or invalid cache, ClickHouse is queried and the cache is
    refreshed when caching is enabled.
    """
    cache_reason = "cache disabled"
    cache_paths: tuple[Path, Path] = (Path(""), Path(""))

    if cache_config is not None:
        cached_frame, cache_reason, cache_paths = _load_cache_if_valid(
            dataset_name=dataset_name,
            symbol=symbol,
            requested_start_date=start_date,
            requested_end_date=end_date,
            required_columns=required_columns,
            dtype_allowlist=dtype_allowlist,
            date_column=date_column,
            cache_config=cache_config,
        )
        if cached_frame is not None:
            return normalize(cached_frame)

        logger.warning(
            "%s cache not used for %s: %s",
            dataset_name.capitalize(),
            symbol,
            cache_reason,
        )

    try:
        dataframe = query_from_clickhouse()
    except Exception as error:
        if cache_config is None:
            raise

        cache_root = _cache_root_dir(cache_config)
        message = _offline_error_message(
            dataset_name=dataset_name,
            symbol=symbol,
            cache_reason=cache_reason,
            cache_paths=cache_paths,
            cache_root=cache_root,
            start_date=start_date,
            end_date=end_date,
            db_error=error,
        )
        raise RuntimeError(message) from error

    if cache_config is not None:
        _write_cache(
            dataframe=dataframe,
            dataset_name=dataset_name,
            symbol=symbol,
            date_column=date_column,
            cache_config=cache_config,
        )

    return dataframe


def load_options(
    symbol: str,
    start_date: str,
    end_date: str,
    delta_min: float = 0.05,
    delta_max: float = 0.95,
    cache_config: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Load and clean options chain rows for one underlying symbol.

    Parameters
    ----------
    symbol : str
        Underlying symbol, for example `SPX` or `NDX`.
    start_date : str
        Inclusive start date in `YYYY-MM-DD` format.
    end_date : str
        Inclusive end date in `YYYY-MM-DD` format.
    delta_min : float, default=0.05
        Lower bound for absolute delta filter.
    delta_max : float, default=0.95
        Upper bound for absolute delta filter.
    cache_config : dict[str, object] | None, default=None
        Offline-first cache policy config from `config.toml`.
        When provided, cache is validated using Parquet + JSON sidecar before
        any ClickHouse query.

    Returns
    -------
    pd.DataFrame
        Cleaned options rows with columns:
        `trade_date`, `expiry_date`, `option_type`, `strike_price`, `mid_iv`,
        `delta`, `volume`, `open_interest`, `dte`.

    Raises
    ------
    RuntimeError
        If cache is missing or invalid and ClickHouse is unavailable while
        `cache_config` is provided.
    """
    return _load_dataset_with_cache(
        dataset_name="options",
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        required_columns=OPTIONS_CACHE_COLUMNS,
        dtype_allowlist=OPTIONS_DTYPE_ALLOWLIST,
        date_column="trade_date",
        cache_config=cache_config,
        normalize=_normalize_options_dataframe,
        query_from_clickhouse=lambda: _query_options_from_clickhouse(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            delta_min=delta_min,
            delta_max=delta_max,
        ),
    )


def load_daily_prices(
    symbol: str,
    start_date: str,
    end_date: str,
    cache_config: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Load daily close prices from intraday index data.

    Parameters
    ----------
    symbol : str
        Index symbol, for example `SPX` or `NDX`.
    start_date : str
        Inclusive start date in `YYYY-MM-DD` format.
    end_date : str
        Inclusive end date in `YYYY-MM-DD` format.
    cache_config : dict[str, object] | None, default=None
        Offline-first cache policy config from `config.toml`.

    Returns
    -------
    pd.DataFrame
        Daily close prices with columns `date` and `close`.

    Raises
    ------
    RuntimeError
        If cache is missing or invalid and ClickHouse is unavailable while
        `cache_config` is provided.
    """
    return _load_dataset_with_cache(
        dataset_name="prices",
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        required_columns=PRICES_CACHE_COLUMNS,
        dtype_allowlist=PRICES_DTYPE_ALLOWLIST,
        date_column="date",
        cache_config=cache_config,
        normalize=_normalize_prices_dataframe,
        query_from_clickhouse=lambda: _query_prices_from_clickhouse(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        ),
    )
