"""Query Traces page – browse query trace history with stage waterfall.

Layout:
1. Optional keyword search filter
2. Trace list (reverse-chronological, filtered to trace_type=="query")
3. Detail view: stage waterfall + Dense vs Sparse comparison + Rerank delta
4. Per-trace Ragas evaluation button (LLM-as-Judge scoring)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import streamlit as st

from src.observability.dashboard.i18n import t
from src.observability.dashboard.services.trace_service import TraceService

logger = logging.getLogger(__name__)


def render() -> None:
    """Render the Query Traces page."""
    st.header(t("🔎 Query Traces", "🔎 查询追踪"))

    svc = TraceService()
    traces = svc.list_traces(trace_type="query")

    if not traces:
        st.info(t("No query traces recorded yet. Run a query first!", "还没有记录任何查询 trace。请先运行一次查询。"))
        return

    # ── Keyword filter ─────────────────────────────────────────────
    keyword = st.text_input(
        t("Search by query keyword", "按查询关键词搜索"),
        value="",
        key="qt_keyword",
    )
    if keyword.strip():
        kw = keyword.strip().lower()
        traces = [
            t
            for t in traces
            if kw in str(t.get("metadata", {})).lower()
            or kw in str(t.get("stages", [])).lower()
        ]

    st.subheader(t(f"📋 Query History ({len(traces)})", f"📋 查询历史 ({len(traces)})"))

    for idx, trace in enumerate(traces):
        trace_id = trace.get("trace_id", "unknown")
        started = trace.get("started_at", "—")
        total_ms = trace.get("elapsed_ms")
        total_label = f"{total_ms:.0f} ms" if total_ms is not None else "—"
        meta = trace.get("metadata", {})
        query_text = meta.get("query", "")
        source = meta.get("source", "unknown")

        # ── Expander title: show query text ────────────────────
        query_preview = (
            query_text[:40] + "…" if len(query_text) > 40 else query_text
        ) if query_text else "—"
        expander_title = (
            f"🔍 \"{query_preview}\"  ·  {total_label}  ·  {started[:19]}"
        )

        with st.expander(expander_title, expanded=(idx == 0)):
            # ── 1. Query overview ──────────────────────────────
            st.markdown(t("#### 💬 Query", "#### 💬 查询"))
            col_q, col_meta = st.columns([3, 1])
            with col_q:
                st.markdown(f"> {query_text}")
            with col_meta:
                source_emoji = "🤖" if source == "mcp" else "📡"
                st.markdown(f"**{t('Source', '来源')}:** {source_emoji} `{source}`")
                st.markdown(f"**Top-K:** `{meta.get('top_k', '—')}`")
                st.markdown(f"**{t('Collection', 'Collection')}:** `{meta.get('collection', '—')}`")

            st.divider()

            # ── 2. Overview metrics ────────────────────────────
            timings = svc.get_stage_timings(trace)
            stages_by_name = {t["stage_name"]: t for t in timings}

            dense_d = (stages_by_name.get("dense_retrieval", {}).get("data") or {})
            sparse_d = (stages_by_name.get("sparse_retrieval", {}).get("data") or {})
            fusion_d = (stages_by_name.get("fusion", {}).get("data") or {})
            rerank_d = (stages_by_name.get("rerank", {}).get("data") or {})

            dense_count = dense_d.get("result_count", 0)
            sparse_count = sparse_d.get("result_count", 0)
            fusion_count = fusion_d.get("result_count", 0)
            rerank_count = rerank_d.get("output_count", 0)

            rc1, rc2, rc3, rc4, rc5 = st.columns(5)
            with rc1:
                st.metric(t("Dense Hits", "Dense 命中数"), dense_count)
            with rc2:
                st.metric(t("Sparse Hits", "Sparse 命中数"), sparse_count)
            with rc3:
                st.metric(t("Fused", "融合后"), fusion_count or (dense_count + sparse_count))
            with rc4:
                st.metric(t("After Rerank", "重排后"), rerank_count if rerank_d else "—")
            with rc5:
                st.metric(t("Total Time", "总耗时"), total_label)

            st.divider()

            # ── 3. Stage timing waterfall ──────────────────────
            main_stage_names = ("query_processing", "dense_retrieval", "sparse_retrieval", "fusion", "rerank")
            main_timings = [t for t in timings if t["stage_name"] in main_stage_names]
            if main_timings:
                st.markdown(t("#### ⏱️ Stage Timings", "#### ⏱️ 阶段耗时"))
                chart_data = {t["stage_name"]: t["elapsed_ms"] for t in main_timings}
                st.bar_chart(chart_data, horizontal=True)
                st.table([
                    {
                        t("Stage", "阶段"): t["stage_name"],
                        t("Elapsed (ms)", "耗时 (ms)"): round(t["elapsed_ms"], 2),
                    }
                    for t in main_timings
                ])

            st.divider()

            # ── 4. Per-stage detail tabs ───────────────────────
            st.markdown(t("#### 🔍 Stage Details", "#### 🔍 阶段详情"))

            tab_defs = []
            if "query_processing" in stages_by_name:
                tab_defs.append(("🔤 Query Processing", "query_processing"))
            if "dense_retrieval" in stages_by_name:
                tab_defs.append(("🟦 Dense Retrieval", "dense_retrieval"))
            if "sparse_retrieval" in stages_by_name:
                tab_defs.append(("🟨 Sparse Retrieval", "sparse_retrieval"))
            if "fusion" in stages_by_name:
                tab_defs.append(("🟩 Fusion (RRF)", "fusion"))
            if "rerank" in stages_by_name:
                tab_defs.append(("🟪 Rerank", "rerank"))

            if tab_defs:
                tabs = st.tabs([label for label, _ in tab_defs])
                for tab, (label, key) in zip(tabs, tab_defs):
                    with tab:
                        stage = stages_by_name[key]
                        data = stage.get("data", {})
                        elapsed = stage.get("elapsed_ms")
                        if elapsed is not None:
                            st.caption(f"⏱️ {elapsed:.1f} ms")

                        if key == "query_processing":
                            _render_query_processing_stage(data)
                        elif key == "dense_retrieval":
                            _render_retrieval_stage(data, "Dense")
                        elif key == "sparse_retrieval":
                            _render_retrieval_stage(data, "Sparse")
                        elif key == "fusion":
                            _render_fusion_stage(data)
                        elif key == "rerank":
                            _render_rerank_stage(data)
            else:
                st.info(t("No stage details available.", "没有可用的阶段详情。"))

            # ── 5. Ragas Evaluate button ───────────────────────
            _render_evaluate_button(trace, idx)


def _render_evaluate_button(trace: Dict[str, Any], idx: int) -> None:
    """Render a Ragas evaluate button for a single query trace.

    Re-runs retrieval for the stored query and evaluates with
    RagasEvaluator (LLM-as-Judge).  Only works when query text
    is available in trace metadata.
    """
    meta = trace.get("metadata", {})
    query = meta.get("query", "")
    if not query:
        return

    st.divider()
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        clicked = st.button(
            t("📏 Ragas Evaluate", "📏 Ragas 评测"),
            key=f"eval_trace_{idx}",
            help=t("Re-run this query and score with Ragas (LLM-as-Judge)", "重新运行这条查询，并使用 Ragas（LLM-as-Judge）进行评分"),
        )
    with col_info:
        st.caption(
            t(
                "Uses Ragas to score faithfulness, answer relevancy, and context precision. Calls LLM and may take a few seconds.",
                "使用 Ragas 评估忠实度、答案相关性和上下文精确度。会调用 LLM，可能需要几秒钟。",
            )
        )

    # Show previous result from session state
    result_key = f"eval_result_{idx}"
    if result_key in st.session_state and not clicked:
        _display_eval_metrics(st.session_state[result_key])

    if clicked:
        with st.spinner(t("Running Ragas evaluation…", "正在运行 Ragas 评测…")):
            result = _evaluate_single_trace(query, meta)
        st.session_state[result_key] = result
        _display_eval_metrics(result)


def _evaluate_single_trace(
    query: str,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Re-run retrieval and evaluate a single query with Ragas.

    Returns dict with 'metrics' (score dict) or 'error' (str).
    """
    try:
        from dataclasses import replace as dc_replace

        from src.core.settings import load_settings, EvaluationSettings
        from src.libs.evaluator.evaluator_factory import EvaluatorFactory

        settings = load_settings()

        # Override evaluation settings to force Ragas (frozen dataclass, use replace)
        ragas_eval = EvaluationSettings(
            enabled=True,
            provider="ragas",
            metrics=list(settings.evaluation.metrics) if settings.evaluation.metrics else [],
        )
        settings = dc_replace(settings, evaluation=ragas_eval)
        evaluator = EvaluatorFactory.create(settings)

        # Re-run retrieval
        collection = meta.get("collection", "default")
        top_k = meta.get("top_k", 10)
        chunks = _retrieve_chunks(settings, query, top_k, collection)

        if not chunks:
            return {"error": "No chunks retrieved — is data indexed?"}

        # Build a concise answer from top chunks (limit length to avoid
        # Ragas LLM max_tokens overflow during statement extraction)
        _MAX_ANSWER_CHARS = 1500
        texts = []
        for c in chunks:
            if hasattr(c, "text"):
                texts.append(c.text)
            elif isinstance(c, dict):
                texts.append(c.get("text", str(c)))
            else:
                texts.append(str(c))
        answer = " ".join(texts[:3])
        if len(answer) > _MAX_ANSWER_CHARS:
            answer = answer[:_MAX_ANSWER_CHARS]

        # Evaluate
        metrics = evaluator.evaluate(
            query=query,
            retrieved_chunks=chunks,
            generated_answer=answer,
        )
        return {"metrics": metrics}

    except ImportError as exc:
        return {"error": f"Ragas not installed: {exc}"}
    except Exception as exc:
        logger.exception("Ragas evaluation failed")
        return {"error": str(exc)}


def _retrieve_chunks(
    settings: Any,
    query: str,
    top_k: int,
    collection: str,
) -> list:
    """Re-run HybridSearch to retrieve chunks for evaluation."""
    try:
        from src.core.query_engine.hybrid_search import create_hybrid_search
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.dense_retriever import create_dense_retriever
        from src.core.query_engine.sparse_retriever import create_sparse_retriever
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        vector_store = VectorStoreFactory.create(
            settings, collection_name=collection,
        )
        embedding_client = EmbeddingFactory.create(settings)
        dense_retriever = create_dense_retriever(
            settings=settings,
            embedding_client=embedding_client,
            vector_store=vector_store,
        )
        from src.core.settings import resolve_path
        bm25_indexer = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}")))
        sparse_retriever = create_sparse_retriever(
            settings=settings,
            bm25_indexer=bm25_indexer,
            vector_store=vector_store,
        )
        sparse_retriever.default_collection = collection
        query_processor = QueryProcessor()
        hybrid_search = create_hybrid_search(
            settings=settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
        )

        results = hybrid_search.search(query=query, top_k=top_k)
        return results if isinstance(results, list) else results.results
    except Exception as exc:
        logger.warning("Retrieval for evaluation failed: %s", exc)
        return []


def _display_eval_metrics(result: Dict[str, Any]) -> None:
    """Display evaluation result (metrics or error)."""
    if "error" in result:
        st.error(f"❌ Evaluation failed: {result['error']}")
        return

    metrics = result.get("metrics", {})
    if not metrics:
        st.warning(t("No metrics returned.", "没有返回任何指标。"))
        return

    st.markdown("**📏 Ragas Scores**")
    cols = st.columns(min(len(metrics), 4))
    for i, (name, value) in enumerate(sorted(metrics.items())):
        with cols[i % len(cols)]:
            st.metric(
                label=name.replace("_", " ").title(),
                value=f"{value:.4f}",
            )


def _extract_pipeline_chunks(
    timings: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract chunk lists from each pipeline stage."""
    result: Dict[str, List[Dict[str, Any]]] = {}
    for stage in timings:
        name = stage.get("stage_name", "")
        data = stage.get("data") or {}
        chunks = data.get("chunks")
        if chunks and isinstance(chunks, list):
            result[name] = chunks
    final = meta.get("final_results") or meta.get("results")
    if final and isinstance(final, list):
        result["final"] = final
    return result


# ═══════════════════════════════════════════════════════════════
# Per-stage renderers
# ═══════════════════════════════════════════════════════════════

def _render_query_processing_stage(data: Dict[str, Any]) -> None:
    """Render Query Processing stage: original query → keywords."""
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Original Query**")
        st.info(data.get("original_query", "—"))
    with c2:
        st.markdown("**Method**")
        st.code(data.get("method", "—"))

    keywords = data.get("keywords", [])
    if keywords:
        st.markdown("**Extracted Keywords**")
        st.markdown(" · ".join(f"`{kw}`" for kw in keywords))
    else:
        st.warning(t("No keywords extracted.", "未提取到关键词。"))


def _render_retrieval_stage(data: Dict[str, Any], label: str) -> None:
    """Render Dense or Sparse retrieval stage: method, counts, chunk list."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(t("Method", "方法"), data.get("method", "—"))
    with c2:
        extra = data.get("provider", data.get("keyword_count", "—"))
        extra_label = "Provider" if "provider" in data else "Keywords"
        st.metric(extra_label, extra)
    with c3:
        st.metric(t("Results", "结果数"), data.get("result_count", 0))

    st.markdown(f"**Top-K requested:** `{data.get('top_k', '—')}`")

    chunks = data.get("chunks", [])
    if chunks:
        _render_chunk_list(chunks, prefix=f"{label.lower().replace(' ', '_')}_chunk")
    else:
        st.info(f"No {label.lower()} results returned.")


def _render_fusion_stage(data: Dict[str, Any]) -> None:
    """Render Fusion (RRF) stage: input lists, fused result count, chunk list."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(t("Method", "方法"), data.get("method", "rrf"))
    with c2:
        st.metric(t("Input Lists", "输入列表数"), data.get("input_lists", "—"))
    with c3:
        st.metric(t("Fused Results", "融合结果数"), data.get("result_count", 0))

    st.markdown(f"**Top-K:** `{data.get('top_k', '—')}`")

    chunks = data.get("chunks", [])
    if chunks:
        _render_chunk_list(chunks, prefix="fusion_chunk")
    else:
        st.info(t("No fusion results.", "没有融合结果。"))


def _render_rerank_stage(data: Dict[str, Any]) -> None:
    """Render Rerank stage: method, input/output counts, reranked chunk list."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(t("Method", "方法"), data.get("method", "—"))
    with c2:
        st.metric(t("Provider", "提供方"), data.get("provider", "—"))
    with c3:
        st.metric(t("Input", "输入数"), data.get("input_count", "—"))
    with c4:
        st.metric(t("Output", "输出数"), data.get("output_count", "—"))

    chunks = data.get("chunks", [])
    if chunks:
        _render_chunk_list(chunks, prefix="rerank_chunk")
    else:
        st.info(t("No reranked results.", "没有重排结果。"))


def _render_chunk_list(chunks: List[Dict[str, Any]], prefix: str = "chunk") -> None:
    """Render a list of chunk dicts as a compact, readable table with expandable text."""
    for ci, chunk in enumerate(chunks):
        score = chunk.get("score", 0)
        text = chunk.get("text", "")
        chunk_id = chunk.get("chunk_id", "")
        source = chunk.get("source", "")
        title = chunk.get("title", "")

        # Colour-coded score indicator
        if score >= 0.8:
            score_bar = "🟢"
        elif score >= 0.5:
            score_bar = "🟡"
        else:
            score_bar = "🔴"

        header = f"{score_bar} **#{ci + 1}** — Score: `{score:.4f}`"
        if title:
            header += f" — {title}"

        with st.expander(header, expanded=False):
            cols = st.columns([2, 3])
            with cols[0]:
                st.caption(f"Chunk ID: `{chunk_id}`")
            with cols[1]:
                if source:
                    st.caption(f"Source: `{source}`")
            # Show chunk text (scrollable)
            if text:
                st.text_area(
                    f"{prefix}_{ci}",
                    value=text,
                    height=max(80, min(len(text) // 2, 400)),
                    disabled=True,
                    label_visibility="collapsed",
                )
            else:
                st.caption(t("_No text available_", "_没有可用文本_"))


def _find_stage(timings, name):
    """Find a stage dict by name, or None."""
    for t in timings:
        if t["stage_name"] == name:
            return t
    return None
