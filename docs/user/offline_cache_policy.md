# Offline-First Raw Data Policy

This document defines the developer contract for required raw input files.

## Purpose

Enable full pipeline execution without ClickHouse by shipping required Parquet
inputs plus metadata sidecars in `data/raw/`.

## Terms

- **Parquet**: Columnar file format for local dataset files.
- **Metadata sidecar**: JSON file paired with each Parquet file.
- **Valid raw file set**: Both Parquet and sidecar exist and all checks pass.

## Required directory and filenames

Required directory (repo-relative):

- `data/raw/`

Required files per symbol:

- `options_<symbol>.parquet`
- `options_<symbol>.metadata.json`
- `prices_<symbol>.parquet`
- `prices_<symbol>.metadata.json`

For current default symbols (`SPX`, `NDX`), all eight files must be present.

## Validation checks

Each raw file read validates:

1. `metadata_version` equals configured version.
2. Sidecar schema columns match expected dataset columns.
3. Sidecar schema dtypes are in allowed dtype sets.
4. Parquet schema equals sidecar schema.
5. `row_count` in sidecar equals Parquet row count.
6. Sidecar date coverage equals Parquet min and max date.
7. Cache date coverage includes full requested date range when
   `require_full_date_coverage = true`.

## Runtime behavior

1. **Raw hit**: Load from Parquet, do not call ClickHouse.
2. **Raw miss or invalid + ClickHouse available**: Query ClickHouse,
   normalize data, and rewrite both Parquet and sidecar.
3. **Raw miss or invalid + ClickHouse unavailable**: Raise `RuntimeError`
   with an actionable message containing:
   - required cache directory path,
   - required filenames,
    - raw-file failure reason,
   - database failure details.

## Config keys (`config.toml`)

- `[cache].mode`: currently `"offline_first"`.
- `[cache].required_dir`: required raw-data location.
- `[cache].metadata_version`: sidecar version gate.
- `[cache].require_full_date_coverage`: full-range coverage requirement.
