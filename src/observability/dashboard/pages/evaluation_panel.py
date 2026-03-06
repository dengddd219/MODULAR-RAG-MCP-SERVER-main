"""Evaluation Panel page – run evaluations and view metrics.

Layout:
1. Configuration section: select evaluator backend, golden test set, top_k
2. Run button with progress indicator
3. Results section: aggregate metrics, per-query detail table
4. Optional: historical evaluation results comparison
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

logger = logging.getLogger(__name__)

# Default golden test set location
DEFAULT_GOLDEN_SET = Path("tests/fixtures/golden_test_set.json")
# Evaluation results history file
EVAL_HISTORY_PATH = Path("logs/eval_history.jsonl")


def render() -> None:
    """Render the Evaluation Panel page."""
    st.header("📏 Evaluation Panel")
    st.markdown(
        "Run evaluation against a **golden test set** to measure retrieval "
        "and generation quality. Results include per-query details and "
        "aggregate metrics."
    )

    # ── Configuration Section ──────────────────────────────────────
    st.subheader("⚙️ Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        backend = st.selectbox(
            "Evaluator Backend",
            options=["custom", "ragas", "composite"],
            index=1,
            key="eval_backend",
            help="Select which evaluator backend to use.",
        )

    with col2:
        top_k = st.number_input(
            "Top-K",
            min_value=1,
            max_value=50,
            value=3,
            key="eval_top_k",
            help="Number of chunks to retrieve per query.",
        )

    with col3:
        collection = st.text_input(
            "Collection (optional)",
            value="default",
            key="eval_collection",
            help="Limit retrieval to a specific collection.",
        )

    # Golden test set file selection
    golden_path_str = st.text_input(
        "Golden Test Set Path",
        value=str(DEFAULT_GOLDEN_SET),
        key="eval_golden_path",
        help="Path to the golden_test_set.json file.",
    )
    golden_path = Path(golden_path_str)

    # Validate golden set exists
    if not golden_path.exists():
        st.warning(f"⚠️ Golden test set not found: `{golden_path}`")

    # ── Run Evaluation ─────────────────────────────────────────────
    st.divider()

    run_clicked = st.button(
        "▶️  Run Evaluation",
        type="primary",
        key="eval_run_btn",
        disabled=not golden_path.exists(),
    )

    if run_clicked:
        _run_evaluation(
            backend=backend,
            golden_path=golden_path,
            top_k=int(top_k),
            collection=collection.strip() or None,
        )

    # ── Historical Results ─────────────────────────────────────────
    st.divider()
    _render_history()


def _run_evaluation(
    backend: str,
    golden_path: Path,
    top_k: int,
    collection: Optional[str],
) -> None:
    """Execute an evaluation run and display results.

    Attempts to load the evaluator, run the golden test set, and
    display aggregate + per-query metrics.  Falls back to a graceful
    error message on failure.
    """
    with st.spinner("Loading evaluator and running evaluation…"):
        try:
            report_dict = _execute_evaluation(
                backend=backend,
                golden_path=golden_path,
                top_k=top_k,
                collection=collection,
            )
        except Exception as exc:
            st.error(f"❌ Evaluation failed: {exc}")
            logger.exception("Evaluation failed")
            return

    # ── Display results ────────────────────────────────────────────
    st.success("✅ Evaluation complete!")

    _render_aggregate_metrics(report_dict)
    _render_query_details(report_dict)

    # Save to history
    _save_to_history(report_dict)


def _execute_evaluation(
    backend: str,
    golden_path: Path,
    top_k: int,
    collection: Optional[str],
) -> Dict[str, Any]:
    """Run the evaluation pipeline and return the report dict."""
    collection = collection or "default"
    
    import dataclasses
    from dotenv import load_dotenv
    load_dotenv()

    from src.core.settings import load_settings
    from src.libs.evaluator.evaluator_factory import EvaluatorFactory
    from src.observability.evaluation.eval_runner import EvalRunner

    settings = load_settings()

    # 1. 覆盖 Vector Store 配置
    if collection:
        new_vs = dataclasses.replace(settings.vector_store, collection_name=collection)
        settings = dataclasses.replace(settings, vector_store=new_vs)
        
    # 2. 覆盖 Retrieval 配置
    if top_k > 0:
        new_retrieval = dataclasses.replace(
            settings.retrieval, 
            dense_top_k=top_k, 
            sparse_top_k=top_k
        )
        settings = dataclasses.replace(settings, retrieval=new_retrieval)

    # 3. 【本次修复点】：正确覆写 evaluation 配置，并将完整的 settings 传给工厂
    if hasattr(settings, "evaluation"):
        new_eval = dataclasses.replace(
            settings.evaluation,
            enabled=True,
            provider=backend
        )
        settings = dataclasses.replace(settings, evaluation=new_eval)
    
    # 传入完整的 settings！这样 Ragas 就能找到 settings.llm 了
    evaluator = EvaluatorFactory.create(settings)

    # 4. 初始化检索引擎
    hybrid_search = _try_create_hybrid_search(settings, collection)

    # 5. 组装并运行
    runner = EvalRunner(
        settings=settings,
        hybrid_search=hybrid_search,
        evaluator=evaluator,
    )

    report = runner.run(
        test_set_path=golden_path,
        top_k=top_k,
        collection=None,  # 保持 None，绕过严格的标签过滤
    )

    return report.to_dict()
def _try_create_hybrid_search(settings: Any, collection: Optional[str] = None) -> Any:
    """Attempt to create a fully equipped HybridSearch instance."""
    try:
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.core.query_engine.dense_retriever import create_dense_retriever
        from src.core.query_engine.sparse_retriever import create_sparse_retriever
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.hybrid_search import create_hybrid_search
        from src.ingestion.storage.bm25_indexer import BM25Indexer

        collection_name = collection or "default"

        # 1. 初始化数据库
        vector_store = VectorStoreFactory.create(settings, collection_name=collection_name)

        # 2. 初始化稠密检索器 (Dense)
        embedding_client = EmbeddingFactory.create(settings)
        dense_retriever = create_dense_retriever(
            settings=settings,
            embedding_client=embedding_client,
            vector_store=vector_store,
        )

        # 3. 初始化稀疏检索器 (Sparse/BM25)
        bm25_indexer = BM25Indexer(index_dir=f"data/db/bm25/{collection_name}")
        sparse_retriever = create_sparse_retriever(
            settings=settings,
            bm25_indexer=bm25_indexer,
            vector_store=vector_store,
        )
        sparse_retriever.default_collection = collection_name

        # 4. 组装终极搜索引擎
        hybrid_search = create_hybrid_search(
            settings=settings,
            query_processor=QueryProcessor(),
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
        )
        return hybrid_search
    except Exception as exc:
        logger.warning("Could not create HybridSearch: %s", exc)
        return None

def _render_aggregate_metrics(report: Dict[str, Any]) -> None:
    """Display aggregate metrics as metric cards."""
    st.subheader("📊 Aggregate Metrics")

    agg = report.get("aggregate_metrics", {})

    if not agg:
        st.info("No aggregate metrics available.")
        return

    cols = st.columns(min(len(agg), 4))
    for idx, (name, value) in enumerate(sorted(agg.items())):
        with cols[idx % len(cols)]:
            st.metric(
                label=name.replace("_", " ").title(),
                value=f"{value:.4f}",
            )

    st.caption(
        f"Evaluator: **{report.get('evaluator_name', '—')}** · "
        f"Queries: **{report.get('query_count', 0)}** · "
        f"Total time: **{report.get('total_elapsed_ms', 0):.0f} ms**"
    )


def _render_query_details(report: Dict[str, Any]) -> None:
    """Display per-query evaluation results in an expandable table."""
    st.subheader("🔍 Per-Query Details")

    query_results = report.get("query_results", [])
    if not query_results:
        st.info("No per-query results available.")
        return

    for idx, qr in enumerate(query_results):
        query = qr.get("query", "—")
        elapsed = qr.get("elapsed_ms", 0)
        metrics = qr.get("metrics", {})

        # Build metric summary for the expander label
        metric_summary = " · ".join(
            f"{k}: {v:.3f}" for k, v in sorted(metrics.items())
        )
        if not metric_summary:
            metric_summary = "no metrics"

        with st.expander(
            f"**Q{idx + 1}**: {query[:80]} — {elapsed:.0f} ms — {metric_summary}",
            expanded=False,
        ):
            # Metrics
            if metrics:
                mcols = st.columns(min(len(metrics), 4))
                for midx, (mname, mval) in enumerate(sorted(metrics.items())):
                    with mcols[midx % len(mcols)]:
                        st.metric(mname, f"{mval:.4f}")

            # Retrieved chunks
            chunks = qr.get("retrieved_chunk_ids", [])
            if chunks:
                st.markdown(f"**Retrieved Chunks** ({len(chunks)}):")
                st.code(", ".join(chunks[:20]), language=None)

            # Generated answer
            answer = qr.get("generated_answer")
            if answer:
                st.markdown("**Generated Answer:**")
                st.text(answer[:500])


def _render_history() -> None:
    """Display historical evaluation results for comparison."""
    st.subheader("📈 Evaluation History")

    history = _load_history()
    if not history:
        st.info(
            "No evaluation history yet. Run an evaluation to start tracking!"
        )
        return

    # Show recent runs as a table
    rows = []
    for entry in history[-10:]:  # last 10 runs
        rows.append(
            {
                "Timestamp": entry.get("timestamp", "—"),
                "Evaluator": entry.get("evaluator_name", "—"),
                "Queries": entry.get("query_count", 0),
                "Time (ms)": round(entry.get("total_elapsed_ms", 0)),
                **{
                    k: round(v, 4)
                    for k, v in entry.get("aggregate_metrics", {}).items()
                },
            }
        )

    st.dataframe(rows, width="stretch")


def _save_to_history(report: Dict[str, Any]) -> None:
    """Append an evaluation report to the history file."""
    try:
        EVAL_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **report,
        }
        with EVAL_HISTORY_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Failed to save evaluation history: %s", exc)


def _load_history() -> List[Dict[str, Any]]:
    """Load evaluation history from JSONL file."""
    if not EVAL_HISTORY_PATH.exists():
        return []

    entries: List[Dict[str, Any]] = []
    try:
        with EVAL_HISTORY_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as exc:
        logger.warning("Failed to load evaluation history: %s", exc)

    return entries
