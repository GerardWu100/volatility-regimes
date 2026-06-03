"""Repository root and standard output path constants.

All runtime paths are resolved from the repository root so scripts, tests,
notebooks, and package modules share one layout contract.
"""

from __future__ import annotations

from pathlib import Path

# src/volatility_regimes/project_paths.py -> parents[2] is the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROOT_CONFIG_PATH = PROJECT_ROOT / "config.toml"
WALKFORWARD_CONFIG_PATH = PROJECT_ROOT / "walkforward.toml"

REPORTS_DESCRIPTIVE_DIR = PROJECT_ROOT / "outputs" / "reports" / "descriptive"
FIGURES_DESCRIPTIVE_DIR = PROJECT_ROOT / "outputs" / "figures" / "descriptive"
REPORTS_WALKFORWARD_DIR = PROJECT_ROOT / "outputs" / "reports" / "walkforward"
FIGURES_WALKFORWARD_DIR = PROJECT_ROOT / "outputs" / "figures" / "walkforward"
