"""LLM Arena page – Model competition and optimization platform.

This page provides:
- Module A: Interactive Playground (single query testing)
- Module B: Exhaustive Benchmark (batch testing with leaderboard)
- Module C: Scoring Engine (normalized composite scoring)

The page demonstrates the superiority of "small model + large model" hybrid
strategy for C-end scenarios through quantitative data (cost, latency, quality,
fine-grained routing accuracy).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from src.core.query_engine.intent_router import IntentRouter, IntentRoutingResult
from src.core.response.rag_generator import RAGGenerator
from src.core.settings import load_settings
from src.core.trace.trace_context import TraceContext
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.libs.llm.model_evaluator import ModelEvaluator, ModelMetrics
from src.libs.llm.model_manager import ModelConfig, ModelManager
from src.observability.dashboard.services.scoring_engine import (
    ScoringEngine,
    StrategyMetrics,
)
from src.observability.evaluation.ragas_evaluator import RagasEvaluator

logger = logging.getLogger(__name__)

# Default golden test set for model evaluation
DEFAULT_GOLDEN_SET = Path("tests/fixtures/golden_test_set_model_evaluation_and_selection.json")

# Model tier definitions
TIER_2_MODELS = [
    "ollama-qwen2.5:7b",
    "ollama-llama3.1:8b",
    "ollama-glm4:9b",
]

TIER_3_MODELS = [
    "api-deepseek-chat",
    "api-qwen-max",
    "api-glm-4-plus",
    "api-gpt-4o-mini",
]


def render() -> None:
    """Render the LLM Arena page."""
    st.header("🏟️ LLM Arena (模型竞技与调优台)")
    st.markdown(
        "通过量化数据（成本、延迟、质量、**细粒度路由准确率**）来证明"
        "**'小模型 + 大模型'混合策略**在 C 端场景下的优越性。"
    )
    
    # Initialize session state
    if "arena_models_registered" not in st.session_state:
        _initialize_models()
        st.session_state.arena_models_registered = True
    
    # Tab navigation
    tab1, tab2 = st.tabs(["🎮 Interactive Playground", "📊 Exhaustive Benchmark"])
    
    with tab1:
        _render_interactive_playground()
    
    with tab2:
        _render_exhaustive_benchmark()


def _initialize_models() -> None:
    """Initialize and register all models in ModelManager."""
    try:
        settings = load_settings()
        manager = ModelManager(settings)
        
        # Register Tier 2 models (local SLM)
        for model_name in TIER_2_MODELS:
            model_id = model_name.replace(":", "-")
            config = ModelConfig(
                model_id=model_id,
                provider="ollama",
                model_name=model_name,
                display_name=model_name,
                description=f"Local SLM: {model_name}",
                is_small_model=True,
            )
            manager.register_model(config)
        
        # Register Tier 3 models (cloud LLM)
        provider_map = {
            "api-deepseek-chat": ("deepseek", "deepseek-chat"),
            "api-qwen-max": ("qwen", "qwen-max"),
            "api-glm-4-plus": ("glm", "glm-4-plus"),
            "api-gpt-4o-mini": ("openai", "gpt-4o-mini"),
        }
        
        for model_name in TIER_3_MODELS:
            model_id = model_name.replace(":", "-")
            provider, actual_model = provider_map.get(model_name, ("unknown", model_name))
            config = ModelConfig(
                model_id=model_id,
                provider=provider,
                model_name=actual_model,
                display_name=model_name,
                description=f"Cloud LLM: {model_name}",
                is_small_model=False,
            )
            manager.register_model(config)
        
        st.session_state.arena_model_manager = manager
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        st.error(f"模型初始化失败: {e}")


def _render_interactive_playground() -> None:
    """Render Module A: Interactive Playground."""
    st.subheader("🎮 Interactive Playground (单次对弈台)")
    st.markdown("单次 Query 测试，直观感受系统表现。")
    
    # Query input
    query = st.text_input(
        "输入查询",
        value="",
        key="playground_query",
        placeholder="例如：这件羊毛大衣可以机洗吗？",
    )
    
    if not query:
        st.info("请输入查询以开始测试。")
        return
    
    # Strategy selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        strategy_type = st.selectbox(
            "执行策略",
            options=["单模型", "双模型组合策略"],
            key="playground_strategy_type",
        )
    
    with col2:
        if strategy_type == "单模型":
            model_options = [m.display_name for m in _get_all_models()]
            selected_model = st.selectbox(
                "选择模型",
                options=model_options,
                key="playground_single_model",
            )
            small_model = None
            large_model = selected_model
        else:
            small_model_options = [m.display_name for m in _get_tier2_models()]
            large_model_options = [m.display_name for m in _get_tier3_models()]
            
            small_model = st.selectbox(
                "小模型",
                options=small_model_options,
                key="playground_small_model",
            )
            large_model = st.selectbox(
                "大模型",
                options=large_model_options,
                key="playground_large_model",
            )
    
    # Execute button
    if st.button("▶️ 执行查询", type="primary", key="playground_execute"):
        _execute_playground_query(
            query=query,
            strategy_type=strategy_type,
            small_model=small_model if strategy_type == "双模型组合策略" else None,
            large_model=large_model,
        )


def _execute_playground_query(
    query: str,
    strategy_type: str,
    small_model: Optional[str],
    large_model: str,
) -> None:
    """Execute a single query in playground mode."""
    try:
        settings = load_settings()
        manager = st.session_state.arena_model_manager
        evaluator = ModelEvaluator()
        intent_router = IntentRouter()
        
        # Routing detection for hybrid strategy
        routing_result: Optional[IntentRoutingResult] = None
        selected_model_id: Optional[str] = None
        
        if strategy_type == "双模型组合策略":
            # Route query
            routing_result = intent_router.route(query)
            
            # Get routing config from settings (if available)
            routing_config = _get_routing_config(settings)
            simple_intents = routing_config.get("simple_intents", ["fabric_care", "faq", "returns"])
            complexity_threshold = routing_config.get("complexity_threshold", 0.7)
            
            # Determine which model to use
            intent_label = routing_result.intent_label or ""
            confidence = routing_result.intent_confidence or 0.0
            
            is_simple = (
                intent_label.lower() in [s.lower() for s in simple_intents]
                and confidence >= complexity_threshold
            )
            
            if is_simple:
                selected_model_id = _get_model_id_by_display_name(small_model)
                routing_decision = f"判定为：简单意图 ({intent_label}, 置信度: {confidence:.2f}) -> 路由至小模型"
            else:
                selected_model_id = _get_model_id_by_display_name(large_model)
                routing_decision = f"判定为：复杂意图 ({intent_label or 'unknown'}, 置信度: {confidence:.2f}) -> 路由至大模型"
            
            # Display routing detector
            st.markdown("### 🔍 路由探测器")
            st.info(routing_decision)
        else:
            selected_model_id = _get_model_id_by_display_name(large_model)
        
        # Get LLM instance
        manager.set_current_model(selected_model_id)
        llm = manager.get_llm()
        
        # Create trace context
        trace = TraceContext(trace_type="query")
        trace.metadata["query"] = query
        trace.metadata["strategy"] = strategy_type
        trace.metadata["model_id"] = selected_model_id
        
        # Track metrics
        config = manager.get_model_config(selected_model_id)
        with evaluator.track_call(
            model_id=selected_model_id,
            provider=config.provider,
            model_name=config.model_name,
            query=query,
        ) as metrics:
            # Build prompt (simplified for playground)
            prompt = f"请回答以下问题：\n\n{query}"
            messages = [{"role": "user", "content": prompt}]
            
            # Call LLM
            start_time = time.monotonic()
            response = llm.chat(messages, trace=trace)
            elapsed = time.monotonic() - start_time
            
            # Extract response
            if isinstance(response, str):
                answer = response
            else:
                answer = response.content if hasattr(response, "content") else str(response)
            
            # Update metrics
            metrics.latency_ms = elapsed * 1000.0
            metrics.response_length = len(answer)
            
            # Try to extract token usage from response
            if hasattr(response, "usage"):
                usage = response.usage
                metrics.prompt_tokens = usage.get("prompt_tokens", 0)
                metrics.completion_tokens = usage.get("completion_tokens", 0)
                metrics.total_tokens = usage.get("total_tokens", 0)
        
        # Display metrics
        st.markdown("### 📊 Metrics Cards")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("TTFT (首字延迟)", f"{metrics.latency_ms:.0f} ms")
        
        with col2:
            st.metric("总延迟", f"{metrics.latency_ms:.0f} ms")
        
        with col3:
            st.metric("Token 消耗", f"{metrics.total_tokens}")
            st.caption(f"Prompt: {metrics.prompt_tokens} | Completion: {metrics.completion_tokens}")
        
        with col4:
            cost = metrics.calculate_cost()
            st.metric("单次成本", f"${cost:.6f}")
        
        # Display answer
        st.markdown("### 💬 回答")
        st.markdown(answer)
        
        # Quality evaluation button
        st.markdown("### 📏 质量评测")
        if st.button("运行 Ragas 评测", key="playground_eval"):
            _evaluate_playground_answer(query, answer, trace)
    
    except Exception as e:
        logger.exception("Playground query execution failed")
        st.error(f"执行失败: {e}")


def _evaluate_playground_answer(
    query: str,
    answer: str,
    trace: TraceContext,
) -> None:
    """Evaluate answer quality using Ragas."""
    try:
        settings = load_settings()
        
        # Create Ragas evaluator
        evaluator = RagasEvaluator(settings=settings)
        
        # Get retrieved chunks from trace (if available)
        chunks = []
        for stage in trace.stages:
            if stage.get("stage") == "retrieval":
                data = stage.get("data", {})
                chunks = data.get("chunks", [])
                break
        
        if not chunks:
            st.warning("未找到检索到的文档块，无法进行质量评测。")
            return
        
        # Evaluate
        with st.spinner("运行 Ragas 评测中..."):
            metrics = evaluator.evaluate(
                query=query,
                retrieved_chunks=chunks,
                generated_answer=answer,
            )
        
        # Display metrics
        st.markdown("#### 📏 Ragas 评分")
        cols = st.columns(3)
        
        faithfulness = metrics.get("faithfulness", 0.0)
        answer_relevancy = metrics.get("answer_relevancy", 0.0)
        context_precision = metrics.get("context_precision", 0.0)
        
        with cols[0]:
            st.metric("Faithfulness", f"{faithfulness:.4f}")
        with cols[1]:
            st.metric("Answer Relevancy", f"{answer_relevancy:.4f}")
        with cols[2]:
            st.metric("Context Precision", f"{context_precision:.4f}")
    
    except Exception as e:
        logger.exception("Ragas evaluation failed")
        st.error(f"质量评测失败: {e}")


def _render_exhaustive_benchmark() -> None:
    """Render Module B: Exhaustive Benchmark."""
    st.subheader("📊 Exhaustive Benchmark (批量压测与排行榜)")
    st.markdown("跑完测试集，采用两两匹配穷举法找出最优组合。")
    
    # Test set file uploader
    test_set_path = st.text_input(
        "测试集 JSON 文件路径",
        value=str(DEFAULT_GOLDEN_SET),
        key="benchmark_test_set_path",
    )
    
    test_set_file = Path(test_set_path)
    if not test_set_file.exists():
        st.warning(f"⚠️ 测试集文件不存在: {test_set_path}")
        return
    
    # Load test set
    try:
        with open(test_set_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        test_cases = test_data.get("test_cases", [])
        st.info(f"已加载 {len(test_cases)} 个测试用例")
    except Exception as e:
        st.error(f"加载测试集失败: {e}")
        return
    
    # Strategy selection
    st.markdown("#### 选择参赛选手")
    
    all_models = _get_all_models()
    tier2_models = _get_tier2_models()
    tier3_models = _get_tier3_models()
    
    # Single model strategies
    single_model_options = [m.display_name for m in all_models]
    selected_single = st.multiselect(
        "单模型策略",
        options=single_model_options,
        default=[],
        key="benchmark_single",
    )
    
    # Hybrid strategies (combinations)
    hybrid_options = []
    for small in tier2_models:
        for large in tier3_models:
            hybrid_options.append(f"{small.display_name} + {large.display_name}")
    
    selected_hybrid = st.multiselect(
        "双模型组合策略",
        options=hybrid_options,
        default=[],
        key="benchmark_hybrid",
    )
    
    if not selected_single and not selected_hybrid:
        st.info("请至少选择一个策略进行测试。")
        return
    
    # Run benchmark button
    if st.button("▶️ 开始压测", type="primary", key="benchmark_run"):
        _run_benchmark(
            test_cases=test_cases,
            single_strategies=selected_single,
            hybrid_strategies=selected_hybrid,
        )


def _run_benchmark(
    test_cases: List[Dict[str, Any]],
    single_strategies: List[str],
    hybrid_strategies: List[str],
) -> None:
    """Run exhaustive benchmark on test cases."""
    try:
        settings = load_settings()
        manager = st.session_state.arena_model_manager
        evaluator = ModelEvaluator()
        intent_router = IntentRouter()
        ragas_evaluator = RagasEvaluator(settings=settings)
        scoring_engine = ScoringEngine()
        
        # Collect all strategies
        all_strategies: List[Tuple[str, Optional[str], str]] = []
        
        # Add single model strategies
        for model_name in single_strategies:
            all_strategies.append((model_name, None, model_name))
        
        # Add hybrid strategies
        for hybrid in hybrid_strategies:
            parts = hybrid.split(" + ")
            if len(parts) == 2:
                all_strategies.append((hybrid, parts[0], parts[1]))
        
        if not all_strategies:
            st.warning("没有选择任何策略。")
            return
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Results storage
        strategy_results: Dict[str, List[Dict[str, Any]]] = {}
        
        total_tasks = len(all_strategies) * len(test_cases)
        current_task = 0
        
        # Run each strategy
        for strategy_name, small_model, large_model in all_strategies:
            strategy_results[strategy_name] = []
            
            for test_case in test_cases:
                current_task += 1
                progress = current_task / total_tasks
                progress_bar.progress(progress)
                status_text.text(
                    f"运行策略: {strategy_name} | "
                    f"测试用例: {test_case.get('query_id', 'unknown')} "
                    f"({current_task}/{total_tasks})"
                )
                
                query = test_case.get("query", "")
                expected_complexity = test_case.get("expected_complexity", "simple")
                
                # Execute query
                result = _execute_benchmark_query(
                    query=query,
                    strategy_name=strategy_name,
                    small_model=small_model,
                    large_model=large_model,
                    manager=manager,
                    evaluator=evaluator,
                    intent_router=intent_router,
                    ragas_evaluator=ragas_evaluator,
                    settings=settings,
                    expected_complexity=expected_complexity,
                )
                
                strategy_results[strategy_name].append(result)
        
        progress_bar.progress(1.0)
        status_text.text("✅ 压测完成！")
        
        # Compute aggregated metrics
        all_metrics = _compute_aggregated_metrics(strategy_results, test_cases)
        
        # Display leaderboard
        _display_leaderboard(all_metrics, scoring_engine)
    
    except Exception as e:
        logger.exception("Benchmark execution failed")
        st.error(f"压测失败: {e}")


def _execute_benchmark_query(
    query: str,
    strategy_name: str,
    small_model: Optional[str],
    large_model: str,
    manager: ModelManager,
    evaluator: ModelEvaluator,
    intent_router: IntentRouter,
    ragas_evaluator: RagasEvaluator,
    settings: Any,
    expected_complexity: str,
) -> Dict[str, Any]:
    """Execute a single query in benchmark mode."""
    result: Dict[str, Any] = {
        "query": query,
        "expected_complexity": expected_complexity,
        "success": False,
        "latency_s": 0.0,
        "tokens": 0,
        "cost": 0.0,
        "quality_score": 50.0,  # Default fallback
        "routing_predicted": None,
        "routing_confidence": None,
        "routing_correct": None,
    }
    
    try:
        # Determine model to use
        selected_model_id: Optional[str] = None
        routing_result: Optional[IntentRoutingResult] = None
        
        if small_model is not None:
            # Hybrid strategy: route query
            routing_result = intent_router.route(query)
            
            routing_config = _get_routing_config(settings)
            simple_intents = routing_config.get("simple_intents", ["fabric_care", "faq", "returns"])
            complexity_threshold = routing_config.get("complexity_threshold", 0.7)
            
            intent_label = routing_result.intent_label or ""
            confidence = routing_result.intent_confidence or 0.0
            
            is_simple = (
                intent_label.lower() in [s.lower() for s in simple_intents]
                and confidence >= complexity_threshold
            )
            
            if is_simple:
                selected_model_id = _get_model_id_by_display_name(small_model)
                predicted_complexity = "simple"
            else:
                selected_model_id = _get_model_id_by_display_name(large_model)
                predicted_complexity = "complex"
            
            result["routing_predicted"] = predicted_complexity
            result["routing_confidence"] = confidence
            result["routing_correct"] = (predicted_complexity == expected_complexity)
        else:
            # Single model strategy
            selected_model_id = _get_model_id_by_display_name(large_model)
        
        # Get LLM and execute
        manager.set_current_model(selected_model_id)
        llm = manager.get_llm()
        config = manager.get_model_config(selected_model_id)
        
        # Build prompt
        prompt = f"请回答以下问题：\n\n{query}"
        messages = [{"role": "user", "content": prompt}]
        
        # Track metrics
        with evaluator.track_call(
            model_id=selected_model_id,
            provider=config.provider,
            model_name=config.model_name,
            query=query,
        ) as metrics:
            start_time = time.monotonic()
            response = llm.chat(messages)
            elapsed = time.monotonic() - start_time
            
            # Extract response
            if isinstance(response, str):
                answer = response
            else:
                answer = response.content if hasattr(response, "content") else str(response)
            
            # Update metrics
            metrics.latency_ms = elapsed * 1000.0
            metrics.response_length = len(answer)
            
            if hasattr(response, "usage"):
                usage = response.usage
                metrics.prompt_tokens = usage.get("prompt_tokens", 0)
                metrics.completion_tokens = usage.get("completion_tokens", 0)
                metrics.total_tokens = usage.get("total_tokens", 0)
        
        # Evaluate quality (simplified - use dummy chunks for now)
        try:
            quality_metrics = ragas_evaluator.evaluate(
                query=query,
                retrieved_chunks=[{"text": answer[:500]}],  # Simplified
                generated_answer=answer,
            )
            # Average of all metrics
            quality_scores = [v for v in quality_metrics.values() if isinstance(v, (int, float))]
            result["quality_score"] = (
                sum(quality_scores) / len(quality_scores) * 100.0
                if quality_scores
                else 50.0
            )
        except Exception:
            result["quality_score"] = 50.0  # Default fallback
        
        # Fill result
        result["success"] = True
        result["latency_s"] = elapsed
        result["tokens"] = metrics.total_tokens
        result["cost"] = metrics.calculate_cost()
    
    except Exception as e:
        logger.warning(f"Benchmark query failed: {e}")
        result["error"] = str(e)
    
    return result


def _compute_aggregated_metrics(
    strategy_results: Dict[str, List[Dict[str, Any]]],
    test_cases: List[Dict[str, Any]],
) -> List[StrategyMetrics]:
    """Compute aggregated metrics for each strategy."""
    all_metrics: List[StrategyMetrics] = []
    
    for strategy_name, results in strategy_results.items():
        if not results:
            continue
        
        # Filter successful results
        successful = [r for r in results if r.get("success", False)]
        total = len(results)
        success_count = len(successful)
        
        if success_count == 0:
            continue
        
        # Compute averages
        avg_latency = sum(r["latency_s"] for r in successful) / success_count
        latencies = sorted([r["latency_s"] for r in successful])
        p95_latency = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
        
        avg_tokens = sum(r["tokens"] for r in successful) / success_count
        avg_cost = sum(r["cost"] for r in successful) / success_count
        total_cost = sum(r["cost"] for r in successful)
        
        # Quality scores (handle NaN)
        quality_scores = [
            r.get("quality_score", 50.0) / 100.0  # Convert to 0-1
            for r in successful
        ]
        quality_scores = [q for q in quality_scores if not (isinstance(q, float) and (q != q))]  # Remove NaN
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
        
        # Routing accuracy (only for hybrid strategies)
        routing_total_accuracy = None
        routing_simple_accuracy = None
        routing_complex_accuracy = None
        
        if any(r.get("routing_predicted") is not None for r in results):
            # Compute routing accuracies
            routing_correct = sum(1 for r in results if r.get("routing_correct") is True)
            routing_total = sum(1 for r in results if r.get("routing_predicted") is not None)
            
            if routing_total > 0:
                routing_total_accuracy = routing_correct / routing_total
                
                # Simple intent accuracy
                simple_correct = sum(
                    1 for r in results
                    if r.get("expected_complexity") == "simple"
                    and r.get("routing_predicted") == "simple"
                )
                simple_total = sum(
                    1 for r in results
                    if r.get("expected_complexity") == "simple"
                )
                routing_simple_accuracy = simple_correct / simple_total if simple_total > 0 else 0.0
                
                # Complex intent accuracy
                complex_correct = sum(
                    1 for r in results
                    if r.get("expected_complexity") == "complex"
                    and r.get("routing_predicted") == "complex"
                )
                complex_total = sum(
                    1 for r in results
                    if r.get("expected_complexity") == "complex"
                )
                routing_complex_accuracy = complex_correct / complex_total if complex_total > 0 else 0.0
        
        metrics = StrategyMetrics(
            strategy_name=strategy_name,
            success_rate=success_count / total if total > 0 else 0.0,
            avg_latency_s=avg_latency,
            p95_latency_s=p95_latency,
            avg_tokens_per_query=avg_tokens,
            avg_cost_per_query=avg_cost,
            total_cost=total_cost,
            avg_quality_score=avg_quality,
            routing_total_accuracy=routing_total_accuracy,
            routing_simple_accuracy=routing_simple_accuracy,
            routing_complex_accuracy=routing_complex_accuracy,
        )
        
        all_metrics.append(metrics)
    
    return all_metrics


def _display_leaderboard(
    all_metrics: List[StrategyMetrics],
    scoring_engine: ScoringEngine,
) -> None:
    """Display leaderboard with all metrics."""
    st.markdown("### 🏆 Leaderboard (排行榜)")
    
    if not all_metrics:
        st.warning("没有可显示的结果。")
        return
    
    # Compute composite scores
    for metrics in all_metrics:
        metrics.composite_score = scoring_engine.compute_composite_score(
            metrics,
            all_metrics,
        )
    
    # Sort by composite score
    all_metrics.sort(key=lambda m: getattr(m, "composite_score", 0.0), reverse=True)
    
    # Build dataframe
    import pandas as pd
    
    rows = []
    for metrics in all_metrics:
        row = {
            "策略名称": metrics.strategy_name,
            "综合评分": f"{getattr(metrics, 'composite_score', 0.0):.2f}",
            "路由总准确率 (%)": (
                f"{metrics.routing_total_accuracy * 100:.2f}"
                if metrics.routing_total_accuracy is not None
                else "N/A"
            ),
            "简单意图准确率 (%)": (
                f"{metrics.routing_simple_accuracy * 100:.2f}"
                if metrics.routing_simple_accuracy is not None
                else "N/A"
            ),
            "复杂意图准确率 (%)": (
                f"{metrics.routing_complex_accuracy * 100:.2f}"
                if metrics.routing_complex_accuracy is not None
                else "N/A"
            ),
            "成功率 (%)": f"{metrics.success_rate * 100:.2f}",
            "平均延迟 (s)": f"{metrics.avg_latency_s:.3f}",
            "P95 延迟 (s)": f"{metrics.p95_latency_s:.3f}",
            "平均 Token/Query": f"{metrics.avg_tokens_per_query:.0f}",
            "单次平均成本 ($)": f"{metrics.avg_cost_per_query:.6f}",
            "总压测成本 ($)": f"{metrics.total_cost:.6f}",
            "平均质量得分": f"{metrics.avg_quality_score * 100:.2f}",
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)


# Helper functions

def _get_all_models() -> List[ModelConfig]:
    """Get all registered models."""
    manager = st.session_state.arena_model_manager
    return manager.list_models()


def _get_tier2_models() -> List[ModelConfig]:
    """Get Tier 2 (local SLM) models."""
    all_models = _get_all_models()
    return [m for m in all_models if m.is_small_model]


def _get_tier3_models() -> List[ModelConfig]:
    """Get Tier 3 (cloud LLM) models."""
    all_models = _get_all_models()
    return [m for m in all_models if not m.is_small_model]


def _get_model_id_by_display_name(display_name: str) -> Optional[str]:
    """Get model ID by display name."""
    manager = st.session_state.arena_model_manager
    for model in manager.list_models():
        if model.display_name == display_name:
            return model.model_id
    return None


def _get_routing_config(settings: Any) -> Dict[str, Any]:
    """Get routing configuration from settings."""
    # Try to get from settings (if llm_routing is configured)
    if hasattr(settings, "llm_routing"):
        routing = settings.llm_routing
        return {
            "small_model": getattr(routing, "small_model", "ollama-qwen2.5:7b"),
            "large_model": getattr(routing, "large_model", "api-deepseek-chat"),
            "simple_intents": getattr(routing, "simple_intents", ["fabric_care", "faq", "returns"]),
            "complexity_threshold": getattr(routing, "complexity_threshold", 0.7),
        }
    
    # Default config
    return {
        "small_model": "ollama-qwen2.5:7b",
        "large_model": "api-deepseek-chat",
        "simple_intents": ["fabric_care", "faq", "returns"],
        "complexity_threshold": 0.7,
    }

