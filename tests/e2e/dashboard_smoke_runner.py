"""Single-file runner for dashboard smoke tests.

Used by test_dashboard_smoke.py: AppTest loads this script and sets
query_params["page"] to one of overview|data_browser|ingestion_manager|
ingestion_traces|query_traces|evaluation_panel so each page renders in isolation.
Do not run this file directly; it is only for pytest AppTest.from_file().
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Dashboard Smoke",
    page_icon="📊",
    layout="wide",
)

page = st.query_params.get("page", "overview")

if page == "overview":
    from src.observability.dashboard.pages.overview import render
    render()
elif page == "data_browser":
    from src.observability.dashboard.pages.data_browser import render
    render()
elif page == "ingestion_manager":
    from src.observability.dashboard.pages.ingestion_manager import render
    render()
elif page == "ingestion_traces":
    from src.observability.dashboard.pages.ingestion_traces import render
    render()
elif page == "query_traces":
    from src.observability.dashboard.pages.query_traces import render
    render()
elif page == "evaluation_panel":
    from src.observability.dashboard.pages.evaluation_panel import render
    render()
else:
    st.error(f"Unknown page: {page}")
