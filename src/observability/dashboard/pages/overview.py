"""Overview page – system configuration and data statistics.

Displays:
- Component configuration cards (LLM, Embedding, VectorStore …)
- Collection statistics (document count, chunk count, image count)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import streamlit as st

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
    st.header("📊 System Overview")

    # ── Component configuration cards ──────────────────────────────
    st.subheader("🔧 Component Configuration")

    try:
        config_service = ConfigService()
        cards = config_service.get_component_cards()
    except Exception as exc:
        st.error(f"Failed to load configuration: {exc}")
        return

    cols = st.columns(min(len(cards), 3))
    for idx, card in enumerate(cards):
        with cols[idx % len(cols)]:
            st.markdown(f"**{card.name}**")
            st.caption(f"Provider: `{card.provider}`  \nModel: `{card.model}`")
            with st.expander("Details"):
                for k, v in card.extra.items():
                    st.text(f"{k}: {v}")

    # ── Collection statistics ──────────────────────────────────────
    st.subheader("📁 Collection Statistics")

    stats = _safe_collection_stats()
    if stats:
        for name, info in stats.items():
            st.metric(label=name, value=info.get("chunk_count", "?"))
    else:
        st.info("No collections found or ChromaDB unavailable. Ingest some documents first!")

    # ── Trace file statistics ──────────────────────────────────────
    st.subheader("📈 Trace Statistics")

    from src.core.settings import resolve_path
    traces_path = resolve_path("logs/traces.jsonl")
    if traces_path.exists():
        line_count = sum(1 for _ in traces_path.open(encoding="utf-8"))
        st.metric("Total traces", line_count)
    else:
        st.info("No traces recorded yet. Run a query or ingestion first.")
