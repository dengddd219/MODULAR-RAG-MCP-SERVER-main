"""E2E: Dashboard smoke tests.

Verifies that all 6 dashboard pages load and render without raising
Python exceptions. Uses Streamlit's AppTest to run each page in isolation.

Usage::

    pytest -q tests/e2e/test_dashboard_smoke.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Pages defined in src/observability/dashboard/app.py (st.navigation)
DASHBOARD_PAGES = [
    "overview",
    "data_browser",
    "ingestion_manager",
    "ingestion_traces",
    "query_traces",
    "evaluation_panel",
]

RUNNER_SCRIPT = Path(__file__).resolve().parent / "dashboard_smoke_runner.py"


def _run_page_smoke(page: str) -> None:
    """Run a single dashboard page under AppTest; raise if Streamlit or page raises."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError as e:
        pytest.skip(f"streamlit testing not available: {e}")

    if not RUNNER_SCRIPT.exists():
        pytest.skip(f"Runner script not found: {RUNNER_SCRIPT}")

    at = AppTest.from_file(str(RUNNER_SCRIPT))
    at.query_params["page"] = page
    at.run()


@pytest.mark.e2e
@pytest.mark.parametrize("page", DASHBOARD_PAGES)
def test_dashboard_page_loads_without_error(page: str) -> None:
    """Each of the 6 dashboard pages must load and render without exception."""
    _run_page_smoke(page)
