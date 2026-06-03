"""Contract tests for the project demo walkthrough notebook."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "project_demo_walkthrough.ipynb"


def _load_notebook_cells() -> list[dict[str, object]]:
    """Load notebook JSON and return its cells."""
    with NOTEBOOK_PATH.open("r", encoding="utf-8") as notebook_handle:
        notebook_payload = json.load(notebook_handle)
    return notebook_payload["cells"]


def _cell_source_text(cell: dict[str, object]) -> str:
    """Join the cell source payload into one string."""
    source_value = cell.get("source", "")
    if isinstance(source_value, list):
        return "".join(source_value)
    return str(source_value)


def test_demo_notebook_has_required_offline_theory_and_package_usage() -> None:
    """Ensure notebook teaches offline workflow and calls package functions."""
    cells = _load_notebook_cells()

    markdown_text = "\n".join(
        _cell_source_text(cell) for cell in cells if cell.get("cell_type") == "markdown"
    )
    code_text = "\n".join(
        _cell_source_text(cell) for cell in cells if cell.get("cell_type") == "code"
    )

    required_markdown_fragments = [
        "Research question",
        "offline",
        "data/raw",
        "Forward realized volatility",
        "variance risk premium",
        "embargo",
        "walk-forward",
        "Failure modes",
        "What Else?",
        "TL;DR",
    ]
    for fragment in required_markdown_fragments:
        assert fragment.lower() in markdown_text.lower()

    assert "from volatility_regimes" in code_text
    assert "subprocess" not in code_text
    assert "uv run python" not in code_text


def test_demo_notebook_declares_comfortable_demo_window_defaults() -> None:
    """Ensure notebook defaults to a notebook-only demo window for speed."""
    cells = _load_notebook_cells()
    code_text = "\n".join(
        _cell_source_text(cell) for cell in cells if cell.get("cell_type") == "code"
    )

    assert 'demo_start_date = "2022-01-03"' in code_text
    assert 'demo_end_date = "2024-12-31"' in code_text
    assert "start_date = demo_start_date" in code_text
    assert "end_date = demo_end_date" in code_text


def test_demo_notebook_strictly_alternates_markdown_and_code_cells() -> None:
    """Ensure the notebook alternates Markdown and code cells from start to end."""
    cells = _load_notebook_cells()

    assert len(cells) >= 2
    assert cells[0].get("cell_type") == "markdown"

    for index, cell in enumerate(cells):
        expected_cell_type = "markdown" if index % 2 == 0 else "code"
        assert cell.get("cell_type") == expected_cell_type
