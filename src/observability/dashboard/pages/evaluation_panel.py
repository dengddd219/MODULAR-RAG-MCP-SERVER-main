"""Evaluation Panel - retrieval strategy benchmark and comparison.

This page provides two isolated workflows:
1) Run benchmark: execute strategy A/B on a golden test set with progress/resume.
2) History: inspect previous benchmark runs from JSONL history.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from src.core.query_engine.graph_retriever import create_graph_retriever
from src.core.query_engine.hybrid_search import HybridSearchConfig, create_hybrid_search
from src.core.query_engine.reranker import create_core_reranker
from src.core.settings import load_settings
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.observability.dashboard.services.scoring_engine import ScoringEngine, StrategyMetrics

logger = logging.getLogger(__name__)

DEFAULT_GOLDEN_SET = Path("tests/fixtures/golden_test_set.json")
EVAL_PROGRESS_PATH = Path("logs/evaluation_progress.json")
EVAL_HISTORY_PATH = Path("logs/evaluation_history.jsonl")
PENDING_SAVE_KEY = "evaluation_panel_pending_save"
BASELINE_STRATEGY_NAME = "Strategy A: Baseline (Dense+Sparse+Rerank)"


def render() -> None:
    """Render the evaluation panel page."""
    st.header("📏 Evaluation Panel")
    st.caption("对比基线检索与图增强检索，复用 LLM Arena 评分体系。")

    run_tab, history_tab = st.tabs(["▶️ 运行压测", "📈 历史结果"])
    with run_tab:
        _render_run_tab()
    with history_tab:
        _render_history_tab()


def _render_run_tab() -> None:
    st.subheader("压测配置")
    c1, c2, c3 = st.columns(3)
    with c1:
        backend = st.selectbox("Evaluator Backend", ["ragas", "composite", "custom"], index=0)
    with c2:
        top_k = int(st.number_input("Top-K", min_value=1, max_value=50, value=5))
    with c3:
        exec_mode = st.radio("执行模式", ["串行", "并行"], index=1, horizontal=True)

    st.caption(
        "串行：同一条测试样本按策略一个个顺序执行；并行：同一条样本的多策略同时执行（通常更快）。"
    )

    c4, c5 = st.columns(2)
    with c4:
        collection = st.text_input("Collection (可选)", value="default").strip() or None
    with c5:
        golden_path_str = st.text_input("Golden Test Set", value=str(DEFAULT_GOLDEN_SET))

    strategy_mode = st.selectbox(
        "检索策略模式",
        options=[
            "二路（Dense + Sparse + Rerank）",
            "三路（Dense + Sparse + Graph + Rerank）",
            "二路 vs 三路（横向对比）",
        ],
        index=2,
    )

    c6, c7 = st.columns(2)
    with c6:
        run_name = st.text_input("运行名称", value=time.strftime("%Y-%m-%d %H:%M:%S"))
    with c7:
        run_note = st.text_input("备注", value="")

    pending_payload = st.session_state.get(PENDING_SAVE_KEY)
    if isinstance(pending_payload, dict):
        st.info(
            "已生成本次 leaderboard，可选择是否保存到历史。"
        )
        s1, s2 = st.columns(2)
        if s1.button("💾 保存此次结果", key="eval_panel_save_latest"):
            _save_to_history(pending_payload)
            st.session_state.pop(PENDING_SAVE_KEY, None)
            st.success("本次结果已保存到历史记录。")
        if s2.button("🗑️ 丢弃此次结果", key="eval_panel_discard_latest"):
            st.session_state.pop(PENDING_SAVE_KEY, None)
            st.warning("已丢弃当前未保存结果。")

    golden_path = Path(golden_path_str)
    if not golden_path.exists():
        st.warning(f"测试集不存在：`{golden_path}`")
        return

    saved = _load_progress()
    can_resume = (
        isinstance(saved, dict)
        and not saved.get("completed", False)
        and saved.get("test_set_path") == str(golden_path)
        and int(saved.get("top_k", 0)) == top_k
        and str(saved.get("strategy_mode", "二路 vs 三路（横向对比）")) == strategy_mode
    )

    b1, b2, b3 = st.columns(3)
    start_clicked = b1.button("▶️ 开始压测", type="primary")
    resume_clicked = b2.button("⏯️ 继续上次压测", disabled=not can_resume)
    clear_clicked = b3.button("🗑️ 清除进度")

    if clear_clicked:
        _clear_progress()
        st.success("已清除压测进度。")

    if start_clicked or resume_clicked:
        _run_benchmark(
            backend=backend,
            golden_path=golden_path,
            top_k=top_k,
            collection=collection,
            strategy_mode=strategy_mode,
            run_name=run_name.strip() or time.strftime("%Y-%m-%d %H:%M:%S"),
            run_note=run_note.strip(),
            parallel=(exec_mode == "并行"),
            resume_progress=saved if resume_clicked and can_resume else None,
        )


def _run_benchmark(
    backend: str,
    golden_path: Path,
    top_k: int,
    collection: Optional[str],
    strategy_mode: str,
    run_name: str,
    run_note: str,
    parallel: bool,
    resume_progress: Optional[Dict[str, Any]],
) -> None:
    test_cases = _load_test_cases(golden_path)
    if not test_cases:
        st.warning("测试集为空，无法执行压测。")
        return

    base_settings = load_settings()
    benchmark_settings = _build_benchmark_settings(base_settings, backend)
    benchmark_model = f"{benchmark_settings.llm.provider}-{benchmark_settings.llm.model}"
    st.info(f"Ragas 裁判模型已强制绑定：`{benchmark_model}`")

    evaluator = EvaluatorFactory.create(benchmark_settings)
    from src.core.response.rag_generator import RAGGenerator
    generator = RAGGenerator.create(settings=benchmark_settings)
    reranker = create_core_reranker(settings=benchmark_settings)

    all_defs = [
        {"name": "Strategy A: Baseline (Dense+Sparse+Rerank)", "use_graph": False},
        {"name": "Strategy B: Graph (Dense+Sparse+Graph+Rerank)", "use_graph": True},
    ]
    if strategy_mode.startswith("二路（"):
        strategy_defs = [all_defs[0]]
    elif strategy_mode.startswith("三路（"):
        strategy_defs = [all_defs[1]]
    else:
        strategy_defs = all_defs
    searchers = {
        s["name"]: _create_hybrid_search(benchmark_settings, collection, top_k, use_graph=s["use_graph"])
        for s in strategy_defs
    }

    strategy_results: Dict[str, List[Dict[str, Any]]] = {
        s["name"]: [] for s in strategy_defs
    }
    if resume_progress:
        saved_results = resume_progress.get("strategy_results", {})
        for name in strategy_results:
            if isinstance(saved_results.get(name), list):
                strategy_results[name] = saved_results[name]

    total_tasks = len(test_cases) * len(strategy_defs)
    done_tasks = sum(len(v) for v in strategy_results.values())
    progress_bar = st.progress(min(1.0, done_tasks / total_tasks))
    status_text = st.empty()

    for idx, tc in enumerate(test_cases, start=1):
        query = str(tc.get("query", "")).strip()
        if not query:
            continue

        if parallel:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_map = {}
                for s in strategy_defs:
                    strategy_name = s["name"]
                    if _query_done(strategy_results[strategy_name], query):
                        continue
                    future = executor.submit(
                        _run_single_case,
                        query,
                        tc,
                        strategy_name,
                        searchers[strategy_name],
                        reranker,
                        generator,
                        evaluator,
                        top_k,
                    )
                    future_map[future] = strategy_name

                for future in as_completed(future_map):
                    strategy_name = future_map[future]
                    strategy_results[strategy_name].append(future.result())
                    done_tasks += 1
                    progress_bar.progress(min(1.0, done_tasks / total_tasks))
                    status_text.markdown(
                        f"进度：**{done_tasks}/{total_tasks}** · 当前 Query：`{query[:60]}` · 策略：`{strategy_name}`"
                    )
        else:
            for s in strategy_defs:
                strategy_name = s["name"]
                if _query_done(strategy_results[strategy_name], query):
                    continue
                result = _run_single_case(
                    query,
                    tc,
                    strategy_name,
                    searchers[strategy_name],
                    reranker,
                    generator,
                    evaluator,
                    top_k,
                )
                strategy_results[strategy_name].append(result)
                done_tasks += 1
                progress_bar.progress(min(1.0, done_tasks / total_tasks))
                status_text.markdown(
                    f"进度：**{done_tasks}/{total_tasks}** · 当前 Query：`{query[:60]}` · 策略：`{strategy_name}`"
                )

        _save_progress(
            {
                "run_name": run_name,
                "run_note": run_note,
                "test_set_path": str(golden_path),
                "top_k": top_k,
                "backend": backend,
                "strategy_mode": strategy_mode,
                "parallel": parallel,
                "strategy_results": strategy_results,
                "completed": False,
            }
        )

    all_metrics = _compute_aggregated_metrics(strategy_results, test_cases)
    scoring_engine = ScoringEngine()
    _display_leaderboard(all_metrics, scoring_engine)
    _display_visualizations(all_metrics)

    history_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": run_name,
        "note": run_note,
        "backend": backend,
        "strategy_mode": strategy_mode,
        "top_k": top_k,
        "test_set_path": str(golden_path),
        "strategy_results": strategy_results,
        "metrics": [_metrics_to_dict(m) for m in all_metrics],
    }
    st.session_state[PENDING_SAVE_KEY] = history_entry
    _save_progress(
        {
            **history_entry,
            "completed": True,
        }
    )
    st.success("压测完成，leaderboard 已生成。可点击“保存此次结果”写入历史。")


def _run_single_case(
    query: str,
    test_case: Dict[str, Any],
    strategy_name: str,
    hybrid_search: Any,
    reranker: Any,
    generator: Any,
    evaluator: Any,
    top_k: int,
) -> Dict[str, Any]:
    t0 = time.monotonic()
    result: Dict[str, Any] = {
        "query": query,
        "strategy_name": strategy_name,
        "success": False,
        "gen_time_s": 0.0,
        "eval_time_s": 0.0,
        "e2e_latency_s": 0.0,
        "retrieved_chunk_ids": [],
        "quality_score": 50.0,
        "ragas_metrics": {},
        "tokens": 0,
        "cost": 0.0,
    }
    try:
        retrieved = hybrid_search.search(query=query, top_k=top_k)
        chunks = retrieved if isinstance(retrieved, list) else getattr(retrieved, "results", [])

        if reranker is not None and getattr(reranker, "is_enabled", False):
            reranked = reranker.rerank(query=query, results=chunks, top_k=top_k)
            chunks = reranked.results

        result["retrieved_chunk_ids"] = [r.chunk_id for r in chunks]

        t_gen = time.monotonic()
        answer = generator.generate(query=query, results=chunks)
        result["gen_time_s"] = round(time.monotonic() - t_gen, 3)

        t_eval = time.monotonic()
        ragas_metrics = evaluator.evaluate(
            query=query,
            retrieved_chunks=chunks,
            generated_answer=answer,
            ground_truth=test_case.get("reference_answer") or test_case.get("expected_chunk_ids"),
        )
        result["eval_time_s"] = round(time.monotonic() - t_eval, 3)

        faithfulness = _safe_metric(ragas_metrics.get("faithfulness"), 0.5)
        answer_relevancy = _safe_metric(ragas_metrics.get("answer_relevancy"), 0.5)
        context_precision = _safe_metric(ragas_metrics.get("context_precision"), 0.5)
        quality = (faithfulness + answer_relevancy + context_precision) / 3.0

        result["ragas_metrics"] = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
        }
        result["quality_score"] = round(quality * 100.0, 3)

        in_tokens = _estimate_tokens(query + " " + " ".join([c.text for c in chunks]))
        out_tokens = _estimate_tokens(answer)
        result["tokens"] = in_tokens + out_tokens
        result["cost"] = round(_estimate_cost_gpt4o_mini(in_tokens, out_tokens), 6)
        result["success"] = True
    except Exception as exc:
        result["error"] = str(exc)
        logger.warning("Single evaluation failed for strategy=%s query=%s: %s", strategy_name, query[:60], exc)
    finally:
        result["e2e_latency_s"] = round(time.monotonic() - t0, 3)
    return result


def _create_hybrid_search(settings: Any, collection: Optional[str], top_k: int, use_graph: bool) -> Any:
    from src.core.query_engine.dense_retriever import create_dense_retriever
    from src.core.query_engine.query_processor import QueryProcessor
    from src.core.query_engine.sparse_retriever import create_sparse_retriever
    from src.ingestion.storage.bm25_indexer import BM25Indexer
    from src.libs.embedding.embedding_factory import EmbeddingFactory
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory

    collection_name = collection or "default"
    vector_store = VectorStoreFactory.create(settings, collection_name=collection_name)
    embedding_client = EmbeddingFactory.create(settings)
    dense_retriever = create_dense_retriever(settings=settings, embedding_client=embedding_client, vector_store=vector_store)
    sparse_retriever = create_sparse_retriever(
        settings=settings,
        bm25_indexer=BM25Indexer(index_dir=f"data/db/bm25/{collection_name}"),
        vector_store=vector_store,
    )
    sparse_retriever.default_collection = collection_name

    graph_retriever = create_graph_retriever(settings=settings) if use_graph else None
    cfg = HybridSearchConfig(
        dense_top_k=top_k,
        sparse_top_k=top_k,
        graph_top_k=top_k,
        fusion_top_k=top_k,
        enable_dense=True,
        enable_sparse=True,
        enable_graph=use_graph,
        parallel_retrieval=True,
        metadata_filter_post=True,
    )
    return create_hybrid_search(
        settings=settings,
        query_processor=QueryProcessor(),
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        graph_retriever=graph_retriever,
        config=cfg,
    )


def _compute_aggregated_metrics(
    strategy_results: Dict[str, List[Dict[str, Any]]],
    test_cases: List[Dict[str, Any]],
) -> List[StrategyMetrics]:
    metrics_list: List[StrategyMetrics] = []
    total_cases = max(1, len(test_cases))
    for strategy_name, rows in strategy_results.items():
        success_rows = [r for r in rows if r.get("success", False)]
        success_rate = len(success_rows) / total_cases

        avg_gen = _mean([float(r.get("gen_time_s", 0.0)) for r in success_rows])
        avg_eval = _mean([float(r.get("eval_time_s", 0.0)) for r in success_rows])
        p95_latency = _p95([float(r.get("e2e_latency_s", 0.0)) for r in success_rows])
        avg_tokens = _mean([float(r.get("tokens", 0.0)) for r in success_rows])
        avg_cost = _mean([float(r.get("cost", 0.0)) for r in success_rows])
        total_cost = sum(float(r.get("cost", 0.0)) for r in success_rows)

        faithfulness = _mean([_safe_metric(r.get("ragas_metrics", {}).get("faithfulness"), 0.5) for r in success_rows])
        answer_rel = _mean([_safe_metric(r.get("ragas_metrics", {}).get("answer_relevancy"), 0.5) for r in success_rows])
        ctx_prec = _mean([_safe_metric(r.get("ragas_metrics", {}).get("context_precision"), 0.5) for r in success_rows])
        quality = _mean([float(r.get("quality_score", 50.0)) / 100.0 for r in success_rows])
        raw_faithfulness = [_safe_metric(r.get("ragas_metrics", {}).get("faithfulness"), 0.5) for r in success_rows]
        raw_answer_rel = [_safe_metric(r.get("ragas_metrics", {}).get("answer_relevancy"), 0.5) for r in success_rows]
        raw_ctx_prec = [_safe_metric(r.get("ragas_metrics", {}).get("context_precision"), 0.5) for r in success_rows]
        raw_latency = [float(r.get("e2e_latency_s", 0.0)) for r in success_rows]
        raw_cost = [float(r.get("cost", 0.0)) for r in success_rows]

        sm = StrategyMetrics(
            strategy_name=strategy_name,
            success_rate=success_rate,
            avg_latency_s=avg_gen,
            p95_latency_s=p95_latency,
            avg_eval_time_s=avg_eval,
            avg_tokens_per_query=avg_tokens,
            avg_cost_per_query=avg_cost,
            total_cost=total_cost,
            avg_quality_score=quality,
            avg_faithfulness=faithfulness,
            avg_answer_relevancy=answer_rel,
            avg_context_precision=ctx_prec,
            routing_total_accuracy=None,
            routing_simple_accuracy=None,
            routing_complex_accuracy=None,
            raw_faithfulness_scores=raw_faithfulness,
            raw_answer_relevancy_scores=raw_answer_rel,
            raw_context_precision_scores=raw_ctx_prec,
            raw_latency_scores=raw_latency,
            raw_cost_scores=raw_cost,
        )
        sm.avg_e2e_latency_s = _mean([float(r.get("e2e_latency_s", 0.0)) for r in success_rows])  # type: ignore[attr-defined]
        metrics_list.append(sm)
    return metrics_list


def _display_leaderboard(all_metrics: List[StrategyMetrics], scoring_engine: ScoringEngine) -> None:
    st.subheader("🏆 Leaderboard")
    if not all_metrics:
        st.info("暂无可展示结果。")
        return

    with st.expander("📊 评分规则说明", expanded=False):
        st.markdown(
            """
基线相对评分（替代 Min-Max）：
- 固定基线 `Strategy A` 单项分 = 80
- 质量指标（越高越好）：`80 * (1 + 相对变化率)`
- 逆向指标（时延/成本，越低越好）：`80 * (1 - 相对变化率)`
- 单项分裁剪到 [0, 100]
- 综合分 = 质量(60%) + 延迟(30%) + 成本(10%)

显著性检验：
- 使用配对样本 t 检验 `scipy.stats.ttest_rel`
- 标记规则：`p < 0.01 -> **`，`p < 0.05 -> *`，否则 `ns`
"""
        )

    relative_scores = scoring_engine.compute_relative_scores(
        all_metrics=all_metrics,
        baseline_strategy_name=BASELINE_STRATEGY_NAME,
    )
    for m in all_metrics:
        m.composite_score = float(relative_scores.get(m.strategy_name, {}).get("composite_score", 0.0))  # type: ignore[attr-defined]
    all_metrics.sort(key=lambda x: getattr(x, "composite_score", 0.0), reverse=True)

    def to_cn_rank(rank: int) -> str:
        mapping = {1: "第一名", 2: "第二名", 3: "第三名"}
        return mapping.get(rank, f"第{rank}名")

    rows: List[Dict[str, Any]] = []
    for i, m in enumerate(all_metrics, start=1):
        rows.append(
            {
                "排名": to_cn_rank(i),
                "策略": m.strategy_name,
                "综合评分": f"{getattr(m, 'composite_score', 0.0):.2f}",
                "成功率(%)": f"{m.success_rate * 100:.2f}",
                "平均生成时长(s)": f"{m.avg_latency_s:.3f}",
                "平均评估时长(s)": f"{m.avg_eval_time_s:.3f}",
                "端到端时延(s)": f"{getattr(m, 'avg_e2e_latency_s', 0.0):.3f}",
                "单次平均成本($)": f"{m.avg_cost_per_query:.6f}",
                "总成本($)": f"{m.total_cost:.6f}",
                "Faithfulness": f"{(m.avg_faithfulness or 0.0):.3f}",
                "Answer Relevancy": f"{(m.avg_answer_relevancy or 0.0):.3f}",
                "Context Precision": f"{(m.avg_context_precision or 0.0):.3f}",
            }
        )
    st.dataframe(rows, width="stretch")

    st.markdown("### 📈 详细评分分解")
    baseline = None
    for m in all_metrics:
        if m.strategy_name == BASELINE_STRATEGY_NAME:
            baseline = m
            break
    if baseline is None:
        st.info("未找到基线策略（Strategy A），无法展示相对显著性对比。")
        return

    def _sig_label(p: float) -> str:
        if p < 0.01:
            return "**", "极显著"
        if p < 0.05:
            return "*", "显著"
        return "ns", "不显著"

    def _pct_change(base: float, challenger: float) -> float:
        if abs(base) <= 1e-12:
            return 0.0
        return (challenger - base) / abs(base) * 100.0

    st.markdown("**策略差异对比（基于 Query 级原始分数 + 配对 t 检验）**")
    for m in all_metrics:
        if m.strategy_name == baseline.strategy_name:
            continue
        pvals = relative_scores.get(m.strategy_name, {}).get("p_values", {})
        st.markdown(f"### {m.strategy_name}")

        faith_delta = _pct_change(baseline.avg_faithfulness or 0.0, m.avg_faithfulness or 0.0)
        faith_arrow = "🟢" if faith_delta >= 0 else "🔴"
        faith_trend = "提升" if faith_delta >= 0 else "下降"
        faith_mark, faith_desc = _sig_label(float(pvals.get("faithfulness", 1.0)))
        st.markdown(
            f"- Faithfulness: {(baseline.avg_faithfulness or 0.0):.3f} vs {(m.avg_faithfulness or 0.0):.3f} | "
            f"{faith_trend} {abs(faith_delta):.1f}% {faith_arrow} | p-value: {float(pvals.get('faithfulness', 1.0)):.3f} {faith_mark} ({faith_desc}{'提升' if faith_delta >= 0 else '下降'})"
        )

        ar_delta = _pct_change(baseline.avg_answer_relevancy or 0.0, m.avg_answer_relevancy or 0.0)
        ar_arrow = "🟢" if ar_delta >= 0 else "🔴"
        ar_trend = "提升" if ar_delta >= 0 else "下降"
        ar_mark, ar_desc = _sig_label(float(pvals.get("answer_relevancy", 1.0)))
        st.markdown(
            f"- Answer Relevancy: {(baseline.avg_answer_relevancy or 0.0):.3f} vs {(m.avg_answer_relevancy or 0.0):.3f} | "
            f"{ar_trend} {abs(ar_delta):.1f}% {ar_arrow} | p-value: {float(pvals.get('answer_relevancy', 1.0)):.3f} {ar_mark} ({ar_desc}{'提升' if ar_delta >= 0 else '下降'})"
        )

        cp_delta = _pct_change(baseline.avg_context_precision or 0.0, m.avg_context_precision or 0.0)
        cp_arrow = "🟢" if cp_delta >= 0 else "🔴"
        cp_trend = "提升" if cp_delta >= 0 else "下降"
        cp_mark, cp_desc = _sig_label(float(pvals.get("context_precision", 1.0)))
        st.markdown(
            f"- Context Precision: {(baseline.avg_context_precision or 0.0):.3f} vs {(m.avg_context_precision or 0.0):.3f} | "
            f"{cp_trend} {abs(cp_delta):.1f}% {cp_arrow} | p-value: {float(pvals.get('context_precision', 1.0)):.3f} {cp_mark} ({cp_desc}{'提升' if cp_delta >= 0 else '下降'})"
        )

        base_latency = float(getattr(baseline, "avg_e2e_latency_s", baseline.p95_latency_s))
        cur_latency = float(getattr(m, "avg_e2e_latency_s", m.p95_latency_s))
        lat_delta = _pct_change(base_latency, cur_latency)
        lat_arrow = "🟢" if lat_delta <= 0 else "🔴"
        lat_trend = "变快" if lat_delta <= 0 else "变慢"
        lat_mark, lat_desc = _sig_label(float(pvals.get("latency", 1.0)))
        st.markdown(
            f"- End-to-End Latency: {base_latency:.3f}s vs {cur_latency:.3f}s | "
            f"{lat_trend} {abs(lat_delta):.1f}% {lat_arrow} | p-value: {float(pvals.get('latency', 1.0)):.3f} {lat_mark} ({'差异不显著' if lat_mark == 'ns' else lat_desc + ('变快' if lat_delta <= 0 else '变慢')})"
        )


def _display_visualizations(all_metrics: List[StrategyMetrics]) -> None:
    if not all_metrics:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        st.info("未安装 matplotlib，跳过图表展示。")
        return

    tab1, tab2, tab3 = st.tabs(["质量分析", "成本分析", "运行时间分析"])

    with tab1:
        sorted_m = sorted(all_metrics, key=lambda m: m.avg_quality_score, reverse=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh([m.strategy_name for m in sorted_m], [m.avg_quality_score * 100 for m in sorted_m], color="#2ca02c")
        ax.set_title("质量分析（分数从高到低）", pad=20)
        ax.set_xlabel("平均质量得分 (%)")
        ax.invert_yaxis()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab2:
        sorted_m = sorted(all_metrics, key=lambda m: m.avg_cost_per_query)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh([m.strategy_name for m in sorted_m], [m.avg_cost_per_query for m in sorted_m], color="#1f77b4")
        ax.set_title("成本分析（从低到高）", pad=20)
        ax.set_xlabel("单次平均成本 ($)")
        ax.invert_yaxis()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab3:
        sorted_m = sorted(all_metrics, key=lambda m: (m.avg_latency_s + m.avg_eval_time_s))
        names = [m.strategy_name for m in sorted_m]
        gen = [m.avg_latency_s for m in sorted_m]
        ev = [m.avg_eval_time_s for m in sorted_m]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(names, gen, marker="o", label="平均生成时长(s)")
        ax.plot(names, ev, marker="o", label="平均评估时长(s)")
        ax.set_title("运行时间分析（从快到慢）", pad=20)
        ax.set_ylabel("秒 (s)")
        ax.tick_params(axis="x", rotation=20)
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def _render_history_tab() -> None:
    st.subheader("历史压测")
    history = _load_history()
    if not history:
        st.info("暂无历史记录。")
        return
    st.caption("点击任意一条历史记录即可展开查看完整细节（可收回）。")

    indexed_recent_entries = list(reversed(list(enumerate(history))[-20:]))  # latest first
    rows: List[Dict[str, Any]] = []
    rebuilt_metrics_cache: Dict[int, List[StrategyMetrics]] = {}
    for idx, (history_idx, entry) in enumerate(indexed_recent_entries):
        rebuilt = _rebuild_metrics_from_history_entry(entry)
        rebuilt_metrics_cache[idx] = rebuilt
        best_score = max([getattr(m, "composite_score", 0.0) for m in rebuilt], default=0.0)
        rows.append(
            {
                "时间": entry.get("timestamp", "—"),
                "运行名称": entry.get("run_name", "—"),
                "Top-K": entry.get("top_k", 0),
                "Backend": entry.get("backend", "—"),
                "最佳综合评分(重算)": f"{best_score:.2f}",
                "备注": entry.get("note", ""),
                "开发日记": "已记录" if str(entry.get("dev_diary", "")).strip() else "—",
                "下一步建议": "已记录" if str(entry.get("next_step_suggestion", "")).strip() else "—",
            }
        )
    st.dataframe(rows, width="stretch")

    st.markdown("### 抽屉式详情")
    for idx, (history_idx, entry) in enumerate(indexed_recent_entries):
        metrics = rebuilt_metrics_cache.get(idx, [])
        best_score = max([getattr(m, "composite_score", 0.0) for m in metrics], default=0.0)
        title = (
            f"{entry.get('timestamp', '—')} | {entry.get('run_name', '—')} "
            f"| 综合最高 {best_score:.2f}"
        )
        with st.expander(title, expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Top-K", f"{entry.get('top_k', 0)}")
            with c2:
                st.metric("Backend", f"{entry.get('backend', '—')}")
            with c3:
                st.metric("策略模式", f"{entry.get('strategy_mode', '—')}")
            with c4:
                st.metric("运行名称", f"{entry.get('run_name', '—')}")

            note = entry.get("note", "")
            if note:
                st.caption(f"备注：{note}")

            if not metrics:
                st.warning("该历史记录缺少可重建的策略明细，无法展示详细排行榜。")
                continue

            st.markdown("#### Leaderboard（历史重算）")
            _display_leaderboard(metrics, ScoringEngine())
            st.markdown("#### 对比图（历史重算）")
            _display_visualizations(metrics)

            _render_history_query_details(entry)
            _render_history_memo_section(entry=entry, history_index=history_idx)


def _save_progress(progress: Dict[str, Any]) -> None:
    try:
        EVAL_PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {**progress, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        EVAL_PROGRESS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to save progress: %s", exc)


def _load_progress() -> Optional[Dict[str, Any]]:
    if not EVAL_PROGRESS_PATH.exists():
        return None
    try:
        return json.loads(EVAL_PROGRESS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load progress: %s", exc)
        return None


def _clear_progress() -> None:
    try:
        if EVAL_PROGRESS_PATH.exists():
            EVAL_PROGRESS_PATH.unlink()
    except Exception as exc:
        logger.warning("Failed to clear progress: %s", exc)


def _save_to_history(report: Dict[str, Any]) -> None:
    """Append an evaluation report to history file."""
    try:
        EVAL_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **report}
        with EVAL_HISTORY_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Failed to save evaluation history: %s", exc)


def _load_history() -> List[Dict[str, Any]]:
    if not EVAL_HISTORY_PATH.exists():
        return []
    entries: List[Dict[str, Any]] = []
    try:
        with EVAL_HISTORY_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        logger.warning("Failed to load evaluation history: %s", exc)
    return entries


def _update_history_note_at(index: int, note: str) -> None:
    _update_history_entry_at(index=index, updates={"note": note})


def _update_history_entry_at(index: int, updates: Dict[str, Any]) -> None:
    history = _load_history()
    if index < 0 or index >= len(history):
        return
    history[index].update(updates)
    try:
        EVAL_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with EVAL_HISTORY_PATH.open("w", encoding="utf-8") as f:
            for row in history:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Failed to update evaluation history entry: %s", exc)


def _load_test_cases(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    test_cases = data.get("test_cases", [])
    return [tc for tc in test_cases if isinstance(tc, dict)]


def _query_done(rows: List[Dict[str, Any]], query: str) -> bool:
    return any(str(r.get("query", "")) == query for r in rows)


def _build_benchmark_settings(settings: Any, backend: str) -> Any:
    """Force evaluator to use benchmark judge model from settings.

    If settings model is not gpt-4o-mini family, fallback to gpt-4o-mini to keep
    judge consistency with LLM Arena benchmark policy.
    """
    llm_model = getattr(settings.llm, "model", "gpt-4o-mini")
    if llm_model.startswith("api-"):
        llm_model = llm_model.replace("api-", "", 1)
    if "gpt-4o-mini" not in llm_model:
        llm_model = "gpt-4o-mini"

    new_llm = dataclasses.replace(settings.llm, model=llm_model)
    new_eval = dataclasses.replace(
        settings.evaluation,
        enabled=True,
        provider=backend,
        metrics=["faithfulness", "answer_relevancy", "context_precision"],
    )
    return dataclasses.replace(settings, llm=new_llm, evaluation=new_eval)


def _metrics_to_dict(m: StrategyMetrics) -> Dict[str, Any]:
    return {
        "strategy_name": m.strategy_name,
        "success_rate": m.success_rate,
        "avg_latency_s": m.avg_latency_s,
        "p95_latency_s": m.p95_latency_s,
        "avg_eval_time_s": m.avg_eval_time_s,
        "avg_tokens_per_query": m.avg_tokens_per_query,
        "avg_cost_per_query": m.avg_cost_per_query,
        "total_cost": m.total_cost,
        "avg_quality_score": m.avg_quality_score,
        "avg_faithfulness": m.avg_faithfulness,
        "avg_answer_relevancy": m.avg_answer_relevancy,
        "avg_context_precision": m.avg_context_precision,
        "routing_total_accuracy": m.routing_total_accuracy,
        "composite_score": getattr(m, "composite_score", 0.0),
        "raw_faithfulness_scores": m.raw_faithfulness_scores,
        "raw_answer_relevancy_scores": m.raw_answer_relevancy_scores,
        "raw_context_precision_scores": m.raw_context_precision_scores,
        "raw_latency_scores": m.raw_latency_scores,
        "raw_cost_scores": m.raw_cost_scores,
    }


def _rebuild_metrics_from_history_entry(entry: Dict[str, Any]) -> List[StrategyMetrics]:
    """Rebuild metrics from history entry with latest scoring logic."""
    strategy_results = entry.get("strategy_results")
    if isinstance(strategy_results, dict) and strategy_results:
        max_cases = max((len(v) for v in strategy_results.values() if isinstance(v, list)), default=0)
        pseudo_test_cases = [{} for _ in range(max_cases)]
        metrics = _compute_aggregated_metrics(strategy_results, pseudo_test_cases)
        relative_scores = ScoringEngine().compute_relative_scores(
            all_metrics=metrics,
            baseline_strategy_name=BASELINE_STRATEGY_NAME,
        )
        for m in metrics:
            m.composite_score = float(relative_scores.get(m.strategy_name, {}).get("composite_score", 0.0))  # type: ignore[attr-defined]
        return metrics

    # Fallback for legacy entries that only contain aggregate metrics
    metrics_data = entry.get("metrics", [])
    rebuilt: List[StrategyMetrics] = []
    if isinstance(metrics_data, list):
        for m in metrics_data:
            if not isinstance(m, dict):
                continue
            sm = StrategyMetrics(
                strategy_name=str(m.get("strategy_name", "Unknown")),
                success_rate=_safe_float(m.get("success_rate"), 0.0),
                avg_latency_s=_safe_float(m.get("avg_latency_s"), 0.0),
                p95_latency_s=_safe_float(m.get("p95_latency_s"), 0.0),
                avg_eval_time_s=_safe_float(m.get("avg_eval_time_s"), 0.0),
                avg_tokens_per_query=_safe_float(m.get("avg_tokens_per_query"), 0.0),
                avg_cost_per_query=_safe_float(m.get("avg_cost_per_query"), 0.0),
                total_cost=_safe_float(m.get("total_cost"), 0.0),
                avg_quality_score=_safe_float(m.get("avg_quality_score"), 0.0),
                avg_faithfulness=m.get("avg_faithfulness"),
                avg_answer_relevancy=m.get("avg_answer_relevancy"),
                avg_context_precision=m.get("avg_context_precision"),
                routing_total_accuracy=m.get("routing_total_accuracy"),
                routing_simple_accuracy=m.get("routing_simple_accuracy"),
                routing_complex_accuracy=m.get("routing_complex_accuracy"),
                raw_faithfulness_scores=m.get("raw_faithfulness_scores", []) or [],
                raw_answer_relevancy_scores=m.get("raw_answer_relevancy_scores", []) or [],
                raw_context_precision_scores=m.get("raw_context_precision_scores", []) or [],
                raw_latency_scores=m.get("raw_latency_scores", []) or [],
                raw_cost_scores=m.get("raw_cost_scores", []) or [],
            )
            sm.composite_score = _safe_float(m.get("composite_score"), 0.0)  # type: ignore[attr-defined]
            rebuilt.append(sm)
    return rebuilt


def _render_history_query_details(entry: Dict[str, Any]) -> None:
    """Render per-query detail table in history drawer."""
    strategy_results = entry.get("strategy_results")
    if not isinstance(strategy_results, dict) or not strategy_results:
        return

    with st.expander("查看 Query 级明细", expanded=False):
        rows: List[Dict[str, Any]] = []
        for strategy_name, results in strategy_results.items():
            if not isinstance(results, list):
                continue
            for r in results:
                if not isinstance(r, dict):
                    continue
                ragas = r.get("ragas_metrics", {}) if isinstance(r.get("ragas_metrics", {}), dict) else {}
                rows.append(
                    {
                        "策略": strategy_name,
                        "Query": str(r.get("query", ""))[:120],
                        "成功": bool(r.get("success", False)),
                        "端到端时延(s)": f"{_safe_float(r.get('e2e_latency_s'), 0.0):.3f}",
                        "生成时长(s)": f"{_safe_float(r.get('gen_time_s'), 0.0):.3f}",
                        "评估时长(s)": f"{_safe_float(r.get('eval_time_s'), 0.0):.3f}",
                        "Cost($)": f"{_safe_float(r.get('cost'), 0.0):.6f}",
                        "Faithfulness": f"{_safe_float(ragas.get('faithfulness'), 0.0):.3f}",
                        "Answer Rel.": f"{_safe_float(ragas.get('answer_relevancy'), 0.0):.3f}",
                        "Context Prec.": f"{_safe_float(ragas.get('context_precision'), 0.0):.3f}",
                    }
                )
        if rows:
            st.dataframe(rows, width="stretch")


def _render_history_memo_section(entry: Dict[str, Any], history_index: int) -> None:
    """Render editable memo area for a history drawer."""
    st.markdown("#### 📝 开发备忘")
    st.caption("用于记录本次开发日记与下一步建议，保存后会写入历史记录。")

    diary_key = f"history_dev_diary_{history_index}"
    next_key = f"history_next_step_{history_index}"
    save_key = f"history_save_memo_{history_index}"

    dev_diary = st.text_area(
        "开发日记",
        value=str(entry.get("dev_diary", "")),
        height=120,
        key=diary_key,
        placeholder="记录本次改动、遇到的问题、排查过程、结论等。",
    )
    next_step = st.text_area(
        "下一步建议",
        value=str(entry.get("next_step_suggestion", "")),
        height=90,
        key=next_key,
        placeholder="例如：补齐测试、优化性能、修复遗留问题、验证线上数据等。",
    )

    cols = st.columns([1, 2])
    with cols[0]:
        if st.button("💾 保存备忘", key=save_key):
            _update_history_entry_at(
                index=history_index,
                updates={
                    "dev_diary": dev_diary.strip(),
                    "next_step_suggestion": next_step.strip(),
                    "memo_updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            st.success("备忘已保存。")
            st.rerun()
    with cols[1]:
        updated_at = str(entry.get("memo_updated_at", "")).strip()
        if updated_at:
            st.caption(f"上次保存时间：{updated_at}")


def _mean(values: List[float]) -> float:
    vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v)]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _p95(values: List[float]) -> float:
    vals = sorted([v for v in values if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v)])
    if not vals:
        return 0.0
    if len(vals) == 1:
        return float(vals[0])
    try:
        return float(statistics.quantiles(vals, n=100)[94])
    except Exception:
        return float(vals[-1])


def _safe_metric(value: Any, default: float) -> float:
    if not isinstance(value, (int, float)):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return float(max(0.0, min(1.0, value)))


def _safe_float(value: Any, default: float) -> float:
    if not isinstance(value, (int, float)):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return float(value)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Rough token estimator for dashboard cost comparison.
    return max(1, int(len(text) / 4))


def _estimate_cost_gpt4o_mini(input_tokens: int, output_tokens: int) -> float:
    # Approx rates per 1M tokens: input $0.15 / output $0.60.
    return (input_tokens / 1_000_000.0) * 0.15 + (output_tokens / 1_000_000.0) * 0.60
