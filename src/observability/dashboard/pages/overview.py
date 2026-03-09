"""Overview page – system configuration and data statistics.

Displays:
- Component configuration cards (LLM, Embedding, VectorStore …)
- Collection statistics (document count, chunk count, image count)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import streamlit as st

from src.observability.dashboard.i18n import t
from src.observability.dashboard.services.config_service import ConfigService


def _safe_collection_stats() -> Dict[str, Any]:
    """Attempt to load collection statistics from ChromaDB.

    Returns empty dict on failure so the page still renders.
    """
    try:
        from src.core.settings import load_settings
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        settings = load_settings()
        store = VectorStoreFactory.create(settings)
        collections = store.list_collections() if hasattr(store, "list_collections") else []
        stats: Dict[str, Any] = {}
        for name in collections:
            count = store.count(collection_name=name) if hasattr(store, "count") else "?"
            stats[name] = {"chunk_count": count}
        return stats
    except Exception:
        return {}


def render() -> None:
    """Render the Overview page."""
    st.header(t("📊 System Overview", "📊 系统总览"))

    # ── Component configuration cards ──────────────────────────────
    st.subheader(t("🔧 Component Configuration", "🔧 组件配置"))

    try:
        config_service = ConfigService()
        cards = config_service.get_component_cards()
    except Exception as exc:
        st.error(t("Failed to load configuration: ", "加载配置失败：") + str(exc))
        return

    cols = st.columns(min(len(cards), 3))
    for idx, card in enumerate(cards):
        with cols[idx % len(cols)]:
            st.markdown(f"**{card.name}**")
            st.caption(
                t("Provider", "提供方") + f": `{card.provider}`  \n" + t("Model", "模型") + f": `{card.model}`"
            )
            with st.expander(t("Details", "详情")):
                for k, v in card.extra.items():
                    st.text(f"{k}: {v}")

    # ── Collection statistics ──────────────────────────────────────
    st.subheader(t("📁 Collection Statistics", "📁 Collection 统计"))

    stats = _safe_collection_stats()
    if stats:
        for name, info in stats.items():
            st.metric(label=name, value=info.get("chunk_count", "?"))
    else:
        st.info(t("No collections found or ChromaDB unavailable. Ingest some documents first!", "未找到任何 collection，或 ChromaDB 不可用。请先导入文档。"))

    # ── Trace file statistics ──────────────────────────────────────
    st.subheader(t("📈 Trace Statistics", "📈 Trace 统计"))

    from src.core.settings import resolve_path
    traces_path = resolve_path("logs/traces.jsonl")
    if traces_path.exists():
        line_count = sum(1 for _ in traces_path.open(encoding="utf-8"))
        st.metric(t("Total traces", "Trace 总数"), line_count)
    else:
        st.info(t("No traces recorded yet. Run a query or ingestion first.", "还没有记录任何 trace。请先运行查询或导入流程。"))
