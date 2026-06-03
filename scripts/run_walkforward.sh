#!/usr/bin/env bash
set -euo pipefail

uv run python -m volatility_regimes.cli.walkforward
