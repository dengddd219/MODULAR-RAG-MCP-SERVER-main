"""Modular RAG Dashboard – multi-page Streamlit application.

Entry-point: ``streamlit run src/observability/dashboard/app.py``

Pages are registered via ``st.navigation()`` and rendered by their
respective modules under ``pages/``.  Pages not yet implemented show
a placeholder message.
"""

from __future__ import annotations

import streamlit as st

from src.observability.dashboard.i18n import LANGUAGE_OPTIONS, ensure_language, t


# ── Page definitions ─────────────────────────────────────────────────

def _page_overview() -> None:
    from src.observability.dashboard.pages.overview import render
    render()


def _page_data_browser() -> None:
    from src.observability.dashboard.pages.data_browser import render
    render()


def _page_ingestion_manager() -> None:
    from src.observability.dashboard.pages.ingestion_manager import render
    render()


def _page_ingestion_traces() -> None:
    from src.observability.dashboard.pages.ingestion_traces import render
    render()


def _page_query_traces() -> None:
    from src.observability.dashboard.pages.query_traces import render
    render()


def _page_evaluation_panel() -> None:
    from src.observability.dashboard.pages.evaluation_panel import render
    render()


def _page_chat_interface() -> None:
    from src.observability.dashboard.pages.chat_interface import render
    render()


def _page_llm_arena() -> None:
    from src.observability.dashboard.pages.llm_arena import render
    render()


def _pages() -> list[st.Page]:
    return [
        st.Page(_page_overview, title=t("Overview", "总览"), icon="📊", default=True),
        st.Page(_page_chat_interface, title=t("Chat", "对话"), icon="💬"),
        st.Page(_page_data_browser, title=t("Data Browser", "数据浏览"), icon="🔍"),
        st.Page(_page_ingestion_manager, title=t("Ingestion Manager", "导入管理"), icon="📥"),
        st.Page(_page_ingestion_traces, title=t("Ingestion Traces", "导入追踪"), icon="🔬"),
        st.Page(_page_query_traces, title=t("Query Traces", "查询追踪"), icon="🔎"),
        st.Page(_page_evaluation_panel, title=t("Evaluation Panel", "评测面板"), icon="📏"),
        st.Page(_page_llm_arena, title="LLM Arena", icon="🏟️"),
    ]


def main() -> None:
    st.set_page_config(
        page_title="Modular RAG Dashboard",
        page_icon="📊",
        layout="wide",
    )

    ensure_language()
    st.sidebar.selectbox(
        t("Language", "语言"),
        options=LANGUAGE_OPTIONS,
        key="dashboard_language",
    )
    st.sidebar.divider()

    nav = st.navigation(_pages())
    nav.run()


if __name__ == "__main__":
    main()
else:
    # When run directly via `streamlit run app.py`
    main()
