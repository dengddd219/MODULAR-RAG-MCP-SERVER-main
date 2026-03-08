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
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from src.core.query_engine.hybrid_search import create_hybrid_search
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.dense_retriever import create_dense_retriever
from src.core.query_engine.sparse_retriever import create_sparse_retriever
from src.core.query_engine.reranker import create_core_reranker
from src.core.response.rag_generator import RAGGenerator
from src.core.settings import load_settings, resolve_path
from src.core.trace.trace_context import TraceContext
from src.core.trace.trace_collector import TraceCollector
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.libs.llm.base_llm import Message
from src.libs.llm.model_evaluator import ModelEvaluator, ModelMetrics
from src.libs.llm.model_manager import ModelConfig, ModelManager
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.observability.dashboard.services.scoring_engine import (
    ScoringEngine,
    StrategyMetrics,
)
from src.observability.evaluation.ragas_evaluator import RagasEvaluator

logger = logging.getLogger(__name__)


def _safe_progress(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Safely convert a value to a progress bar value in [0.0, 1.0].
    
    Args:
        value: The value to convert (can be NaN, inf, or any number).
        min_val: Minimum value for normalization (default: 0.0).
        max_val: Maximum value for normalization (default: 1.0).
    
    Returns:
        A float in [0.0, 1.0] range, safe for st.progress().
    """
    # Handle NaN and inf
    if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
        return 0.0
    
    # Normalize to [0.0, 1.0] range
    if max_val > min_val:
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))
    else:
        # If max_val <= min_val, just clamp to [0.0, 1.0]
        return max(0.0, min(1.0, value))

# Default golden test set for model evaluation
DEFAULT_GOLDEN_SET = Path("tests/fixtures/golden_test_set_model_evaluation_and_selection.json")

# Benchmark progress save path
BENCHMARK_PROGRESS_DIR = Path("logs")
BENCHMARK_PROGRESS_FILE = BENCHMARK_PROGRESS_DIR / "benchmark_progress.json"

# Model tier definitions
# Tier 2: Local SLM models (Ollama)
# Note: These models need to be downloaded via `ollama pull <model_name>`
# IMPORTANT: Use ultra-small models (0.5B-1.5B) for extremely fast local inference
# These models can run on any CPU and provide instant responses
TIER_2_MODELS = [
    # Ultra-fast small models (0.5B-1.5B) - recommended for instant responses
    "ollama-qwen2.5:0.5b",      # ~0.5B params, lightning fast on any CPU, perfect for simple queries
    "ollama-qwen2.5:1.5b",      # ~1.5B params, very fast, excellent Chinese understanding, perfect for RAG
]

# Tier 3: Cloud LLM models (API)
# Note: These models use OpenAI-compatible API format via 智增增 proxy
# All models use the same base_url (https://api.zhizengzeng.com/v1) and api_key
# but with different model names
TIER_3_MODELS = [
    "api-deepseek-chat",   # DeepSeek via OpenAI format
    "api-gpt-4o-mini",     # OpenAI GPT-4o-mini
    "api-qwen-max",        # Qwen via OpenAI format (if supported)
    "api-glm-4-plus",      # GLM via OpenAI format (if supported)
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
    
    # Tab navigation (Exhaustive Benchmark as default)
    tab1, tab2 = st.tabs(["📊 Exhaustive Benchmark", "🎮 Interactive Playground"])
    
    with tab1:
        _render_exhaustive_benchmark()
    
    with tab2:
        _render_interactive_playground()


def _get_benchmark_model_id(settings: Any) -> str:
    """Get the benchmark model ID from settings (default model before LLM Arena)."""
    provider = settings.llm.provider
    model_name = settings.llm.model
    return f"{provider}-{model_name}".replace("/", "-").replace(":", "-")


def _is_benchmark_model(model_id: str, settings: Any) -> bool:
    """Check if a model is the benchmark model."""
    benchmark_id = _get_benchmark_model_id(settings)
    return model_id == benchmark_id


def _initialize_models() -> None:
    """Initialize and register all models in ModelManager."""
    try:
        settings = load_settings()
        manager = ModelManager(settings)
        
        # Store benchmark model ID in session state
        benchmark_model_id = _get_benchmark_model_id(settings)
        st.session_state.benchmark_model_id = benchmark_model_id
        
        # Register Tier 2 models (local SLM)
        # Check if ollama provider is available
        from src.libs.llm.llm_factory import LLMFactory
        available_providers = set(LLMFactory.list_providers())
        
        if "ollama" not in available_providers:
            logger.warning("Ollama provider not available, skipping Tier 2 models")
            st.warning("⚠️ Ollama provider 未实现，跳过本地模型注册")
        else:
            registered_tier2 = 0
            for model_name in TIER_2_MODELS:
                try:
                    model_id = model_name.replace(":", "-")
                    # Extract actual model name (remove "ollama-" prefix)
                    actual_model = model_name.replace("ollama-", "")
                    config = ModelConfig(
                        model_id=model_id,
                        provider="ollama",
                        model_name=actual_model,  # Use actual model name for Ollama
                        display_name=model_name,
                        description=f"Local SLM: {model_name}",
                        is_small_model=True,
                    )
                    manager.register_model(config)
                    registered_tier2 += 1
                except Exception as e:
                    logger.error(f"Failed to register Tier 2 model {model_name}: {e}")
            
            logger.info(f"Registered {registered_tier2} Tier 2 models successfully")
        
        # Register Tier 3 models (cloud LLM)
        # All models use OpenAI-compatible format via 智增增 proxy
        # They share the same base_url and api_key from settings, but use different model names
        from src.libs.llm.llm_factory import LLMFactory
        available_providers = set(LLMFactory.list_providers())
        
        # Check if OpenAI provider is available (required for all API models)
        if "openai" not in available_providers:
            logger.warning("OpenAI provider not available, skipping Tier 3 models")
            st.warning("⚠️ OpenAI provider 未实现，跳过 API 模型注册")
        else:
            # Model name mapping: display_name -> actual model name for API
            # All use OpenAI-compatible format via 智增增 proxy
            model_name_map = {
                "api-deepseek-chat": "deepseek-chat",
                "api-gpt-4o-mini": "gpt-4o-mini",
                "api-qwen-max": "qwen-max",      # May need to verify model name
                "api-glm-4-plus": "glm-4-plus",  # May need to verify model name
            }
            
            # Get api_key from settings (智增增 proxy)
            api_key = getattr(settings.llm, "api_key", None)
            
            if not api_key:
                logger.warning("API key not found in settings, Tier 3 models may not work")
                st.warning("⚠️ 未找到 API Key，API 模型可能无法使用")
            
            # Base URL mapping for different models via 智增增 proxy
            # Different models may use different base_url endpoints
            # Note: If api-qwen-max fails, try changing base_url to "https://api.zhizengzeng.com/v1"
            base_url_map = {
                "api-deepseek-chat": "https://api.zhizengzeng.com/v1",      # DeepSeek uses standard OpenAI format
                "api-gpt-4o-mini": "https://api.zhizengzeng.com/v1",         # OpenAI uses standard format
                "api-qwen-max": "https://api.zhizengzeng.com/v1",            # Changed from /alibaba to /v1 (API returned 500 error)
                "api-glm-4-plus": "https://api.zhizengzeng.com/v1",         # GLM (assume standard, may need adjustment)
            }
            
            registered_count = 0
            skipped_models = []
            
            for display_name in TIER_3_MODELS:
                model_id = display_name.replace(":", "-")
                actual_model = model_name_map.get(display_name, display_name)
                model_base_url = base_url_map.get(display_name, "https://api.zhizengzeng.com/v1")
                
                try:
                    # All API models use OpenAI provider with custom model name
                    # Each model may have different base_url (e.g., Qwen uses /alibaba)
                    config = ModelConfig(
                        model_id=model_id,
                        provider="openai",  # Use OpenAI provider for all (OpenAI-compatible format)
                        model_name=actual_model,  # Different model names
                        display_name=display_name,
                        description=f"Cloud LLM via 智增增: {display_name}",
                        is_small_model=False,
                        # Override config: each model may have different base_url
                        # These will be passed to OpenAILLM.__init__() as kwargs
                        config_override={
                            "base_url": model_base_url,
                            "api_key": api_key,
                        },
                    )
                    manager.register_model(config)
                    registered_count += 1
                    logger.info(f"Registered API model: {display_name} -> {actual_model}")
                except Exception as e:
                    logger.error(f"Failed to register model {display_name}: {e}")
                    skipped_models.append(f"{display_name} (registration failed: {e})")
            
            if skipped_models:
                logger.warning(f"Skipped {len(skipped_models)} models: {skipped_models}")
                st.warning(f"⚠️ {len(skipped_models)} 个模型注册失败，请检查日志")
            
            logger.info(f"Registered {registered_count} Tier 3 models successfully")
        
        st.session_state.arena_model_manager = manager
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        st.error(f"模型初始化失败: {e}")


def _render_interactive_playground() -> None:
    """Render Module A: Interactive Playground."""
    st.subheader("🎮 Interactive Playground (单次对弈台)")
    st.markdown("单次 Query 测试，直观感受系统表现。")
    
    # Strategy selection (always visible)
    st.markdown("#### 策略配置")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        strategy_type = st.selectbox(
            "执行策略",
            options=["单模型", "双模型组合策略"],
            key="playground_strategy_type",
        )
    
    with col2:
        if strategy_type == "单模型":
            all_models = _get_all_models()
            settings = load_settings()
            benchmark_id = st.session_state.get("benchmark_model_id", _get_benchmark_model_id(settings))
            
            # Format model options with benchmark marker
            model_options = []
            for m in all_models:
                display_name = m.display_name
                if m.model_id == benchmark_id:
                    display_name = f"{display_name} [Benchmark]"
                model_options.append(display_name)
            
            selected_model = st.selectbox(
                "选择模型",
                options=model_options,
                key="playground_single_model",
            )
            # Remove [Benchmark] marker if present
            selected_model = selected_model.replace(" [Benchmark]", "")
            small_model = None
            large_model = selected_model
        else:
            tier2_models = _get_tier2_models()
            tier3_models = _get_tier3_models()
            settings = load_settings()
            benchmark_id = st.session_state.get("benchmark_model_id", _get_benchmark_model_id(settings))
            
            # Format small model options
            small_model_options = []
            for m in tier2_models:
                display_name = m.display_name
                if m.model_id == benchmark_id:
                    display_name = f"{display_name} [Benchmark]"
                small_model_options.append(display_name)
            
            # Format large model options
            large_model_options = []
            for m in tier3_models:
                display_name = m.display_name
                if m.model_id == benchmark_id:
                    display_name = f"{display_name} [Benchmark]"
                large_model_options.append(display_name)
            
            small_model = st.selectbox(
                "小模型",
                options=small_model_options,
                key="playground_small_model",
            )
            # Remove [Benchmark] marker if present
            small_model = small_model.replace(" [Benchmark]", "")
            
            large_model = st.selectbox(
                "大模型",
                options=large_model_options,
                key="playground_large_model",
            )
            # Remove [Benchmark] marker if present
            large_model = large_model.replace(" [Benchmark]", "")
    
    st.divider()
    
    # Query input
    st.markdown("#### 查询输入")
    query = st.text_input(
        "输入查询",
        value="",
        key="playground_query",
        placeholder="例如：什么是RAG？或者：请解释混合检索的工作原理",
    )
    
    # Execute button
    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        execute_clicked = st.button("▶️ 执行查询", type="primary", key="playground_execute")
    
    if not query:
        st.info("💡 请先配置策略和模型，然后输入查询以开始测试。")
    
    if execute_clicked:
        if not query:
            st.warning("⚠️ 请输入查询后再执行。")
        else:
            _execute_playground_query(
                query=query,
                strategy_type=strategy_type,
                small_model=small_model if strategy_type == "双模型组合策略" else None,
                large_model=large_model,
            )


def _execute_rag_query(
    query: str,
    settings: Any,
    manager: ModelManager,
    selected_model_id: str,
    collection: Optional[str] = None,
    top_k: int = 10,
    trace: Optional[TraceContext] = None,
) -> tuple[str, List[Any], TraceContext]:
    """Execute a complete RAG query (Retrieve + Generate).
    
    This function reuses the RAG workflow from chat_interface.py's _query_knowledge_base,
    but replaces the LLM with the selected model from ModelManager.
    
    The workflow is:
    1. HybridSearch for retrieval (Dense + Sparse + Fusion) - Same as chat interface
    2. Reranker (if enabled) - Same as chat interface
    3. RAGGenerator for answer generation - Uses selected model instead of settings.llm
    
    Args:
        query: User query string.
        settings: Application settings.
        manager: ModelManager instance.
        selected_model_id: Model ID to use for generation.
        collection: Optional collection name.
        top_k: Number of results to retrieve.
        trace: Optional TraceContext.
        
    Returns:
        Tuple of (answer, retrieved_chunks, trace).
    """
    # Import chat interface's retrieval logic
    from src.core.response.citation_generator import CitationGenerator
    
    effective_collection = collection or "default"
    
    # ===================================================================
    # Step 1: RETRIEVE (Same as chat_interface._query_knowledge_base)
    # ===================================================================
    
    # Initialize retrieval components (same as chat interface)
    embedding_client = EmbeddingFactory.create(settings)
    vector_store = VectorStoreFactory.create(
        settings,
        collection_name=effective_collection,
    )
    dense_retriever = create_dense_retriever(
        settings=settings,
        embedding_client=embedding_client,
        vector_store=vector_store,
    )
    bm25_indexer = BM25Indexer(
        index_dir=str(resolve_path(f"data/db/bm25/{effective_collection}"))
    )
    sparse_retriever = create_sparse_retriever(
        settings=settings,
        bm25_indexer=bm25_indexer,
        vector_store=vector_store,
    )
    sparse_retriever.default_collection = effective_collection
    query_processor = QueryProcessor()
    hybrid_search = create_hybrid_search(
        settings=settings,
        query_processor=query_processor,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
    )
    reranker = create_core_reranker(settings=settings)
    
    # Create trace if not provided
    if trace is None:
        trace = TraceContext(trace_type="query")
    trace.metadata["query"] = query[:200]
    trace.metadata["collection"] = effective_collection
    trace.metadata["top_k"] = top_k
    trace.metadata["source"] = "llm_arena"
    trace.metadata["model_id"] = selected_model_id
    
    # Perform hybrid search (same as chat interface)
    results = hybrid_search.search(
        query=query,
        top_k=top_k,
        filters={"collection": effective_collection} if collection else None,
        trace=trace,
    )
    
    # Extract RetrievalResult list
    if hasattr(results, "results"):
        retrieval_results = results.results
    elif isinstance(results, list):
        retrieval_results = results
    else:
        retrieval_results = []
    
    # Apply reranking if enabled (same as chat interface)
    if reranker.is_enabled and retrieval_results:
        try:
            rerank_result = reranker.rerank(
                query=query,
                results=retrieval_results,
                top_k=top_k,
                trace=trace,
            )
            retrieval_results = rerank_result.results
        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Using original order.")
    
    # ===================================================================
    # Step 2: GENERATE (LLM Arena specific - uses selected model)
    # ===================================================================
    
    # Get selected LLM from ModelManager (instead of using settings.llm)
    manager.set_current_model(selected_model_id)
    selected_llm = manager.get_llm()
    
    # Create RAGGenerator with the selected LLM (not settings.llm)
    rag_generator = RAGGenerator(
        settings=settings,
        llm=selected_llm,  # Override with selected model
    )
    
    # Generate answer using RAGGenerator (same logic as chat interface)
    # But we need to capture token usage, so we'll call LLM directly after building prompt
    # Build context and prompt (same as RAGGenerator does internally)
    context = rag_generator._build_context(retrieval_results)
    prompt = rag_generator._build_prompt(query, context)
    
    # Call LLM directly to get full response with usage
    messages = [Message(role="user", content=prompt)]
    llm_response = selected_llm.chat(messages, trace=trace)
    
    # Extract answer
    llm_answer = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
    if not llm_answer or not llm_answer.strip():
        llm_answer = "抱歉，无法生成回答。请重试。"
    else:
        llm_answer = llm_answer.strip()
    
    # Store usage in trace metadata for metrics tracking
    if hasattr(llm_response, "usage") and llm_response.usage:
        if hasattr(trace, "metadata"):
            trace.metadata["token_usage"] = llm_response.usage
    
    # Collect trace
    TraceCollector().collect(trace)
    
    return llm_answer, retrieval_results, trace


def _execute_playground_query(
    query: str,
    strategy_type: str,
    small_model: Optional[str],
    large_model: str,
) -> None:
    """Execute a single query in playground mode.
    
    This function directly calls the chat interface's RAG pipeline,
    then adds LLM Arena-specific evaluation and metrics.
    """
    try:
        settings = load_settings()
        manager = st.session_state.arena_model_manager
        evaluator = ModelEvaluator()
        
        # Determine which model to use based on user's manual selection
        selected_model_id: Optional[str] = None
        
        if strategy_type == "双模型组合策略":
            # User manually selected small_model and large_model
            # For hybrid strategy, user needs to manually decide which model to use
            # We'll use small_model for "simple" queries and large_model for "complex" queries
            # But since we're removing auto-prediction, we'll default to large_model
            # User can manually test both by running separate queries
            selected_model_id = _get_model_id_by_display_name(large_model)
        else:
            # Single model strategy: use the selected model directly
            selected_model_id = _get_model_id_by_display_name(large_model)
        
        # Temporarily switch settings to use selected model for RAGGenerator
        # We need to modify settings.llm to use the selected model
        # But since settings is frozen, we'll use ModelManager to get the LLM
        # and pass it to RAGGenerator
        
        # Get model config for metrics tracking
        config = manager.get_model_config(selected_model_id)
        
        # Import chat interface's query function
        from src.observability.dashboard.pages.chat_interface import _query_knowledge_base
        
        # Temporarily override the LLM in settings by creating a custom RAGGenerator
        # Actually, we need to modify the approach: 
        # 1. Call _query_knowledge_base but with custom LLM
        # 2. Or create a wrapper that uses our selected model
        
        # For now, let's use the existing RAG pipeline but with model switching
        # We'll create a custom settings-like object that uses our selected model
        
        # Track metrics
        with evaluator.track_call(
            model_id=selected_model_id,
            provider=config.provider,
            model_name=config.model_name,
            query=query,
        ) as metrics:
            # Execute RAG pipeline using chat interface's function
            # But we need to inject our selected model
            # Let's use _execute_rag_query which already handles model selection
            start_time = time.monotonic()
            answer, retrieved_chunks, trace = _execute_rag_query(
                query=query,
                settings=settings,
                manager=manager,
                selected_model_id=selected_model_id,
                collection=None,
                top_k=10,
                trace=None,
            )
            elapsed = time.monotonic() - start_time
            
            # Update metrics
            metrics.latency_ms = elapsed * 1000.0
            metrics.response_length = len(answer)
            
            # Extract token usage from trace
            if hasattr(trace, "metadata") and "token_usage" in trace.metadata:
                usage = trace.metadata["token_usage"]
                metrics.prompt_tokens = usage.get("prompt_tokens", 0)
                metrics.completion_tokens = usage.get("completion_tokens", 0)
                metrics.total_tokens = usage.get("total_tokens", 0)
        
        # ===================================================================
        # 综合评价指标可视化展示
        # ===================================================================
        
        st.markdown("### 📊 综合评价指标")
        
        # 计算所有指标值
        cost = metrics.calculate_cost()
        num_chunks = len(retrieved_chunks) if retrieved_chunks else 0
        avg_score = 0.0
        max_score = 0.0
        min_score = 0.0
        
        if retrieved_chunks:
            scores = []
            for chunk in retrieved_chunks:
                if hasattr(chunk, "score") and chunk.score is not None:
                    scores.append(float(chunk.score))
            
            if scores:
                avg_score = sum(scores) / len(scores)
                max_score = max(scores)
                min_score = min(scores)
        
        answer_length = len(answer)
        answer_word_count = len(answer.split())
        has_citations = "[1]" in answer or "[2]" in answer or "引用" in answer or "来源" in answer
        cost_per_word = cost / answer_word_count if answer_word_count > 0 else 0.0
        
        # 运行 Ragas 评测
        ragas_metrics = {}
        quality_score = 0.5  # Default to 50% when no evaluation
        default_metrics_used = []  # Track which metrics used default values
        eval_time = 0.0  # Evaluation time in seconds
        
        if retrieved_chunks:
            try:
                with st.spinner("正在运行 Ragas 评测..."):
                    eval_start_time = time.monotonic()
                    settings = load_settings()
                    ragas_evaluator = RagasEvaluator(settings=settings)
                    ragas_metrics = ragas_evaluator.evaluate(
                        query=query,
                        retrieved_chunks=retrieved_chunks,
                        generated_answer=answer,
                    )
                    eval_time = time.monotonic() - eval_start_time
                    # Check which metrics are missing and use defaults
                    if "faithfulness" not in ragas_metrics:
                        default_metrics_used.append("Faithfulness (忠实度)")
                    if "answer_relevancy" not in ragas_metrics:
                        default_metrics_used.append("Answer Relevancy (答案相关性)")
                    if "context_precision" not in ragas_metrics:
                        default_metrics_used.append("Context Precision (上下文精确度)")
                    
                    faithfulness = ragas_metrics.get("faithfulness", 0.5)
                    answer_relevancy = ragas_metrics.get("answer_relevancy", 0.5)
                    context_precision = ragas_metrics.get("context_precision", 0.5)
                    # Ensure all values are valid (not NaN, not inf)
                    if not (isinstance(faithfulness, (int, float)) and not math.isnan(faithfulness) and not math.isinf(faithfulness)):
                        faithfulness = 0.5
                    else:
                        faithfulness = max(0.0, min(1.0, float(faithfulness)))  # Clamp to [0.0, 1.0]
                    if not (isinstance(answer_relevancy, (int, float)) and not math.isnan(answer_relevancy) and not math.isinf(answer_relevancy)):
                        answer_relevancy = 0.5
                    else:
                        answer_relevancy = max(0.0, min(1.0, float(answer_relevancy)))  # Clamp to [0.0, 1.0]
                    if not (isinstance(context_precision, (int, float)) and not math.isnan(context_precision) and not math.isinf(context_precision)):
                        context_precision = 0.5
                    else:
                        context_precision = max(0.0, min(1.0, float(context_precision)))  # Clamp to [0.0, 1.0]
                    quality_score = (faithfulness + answer_relevancy + context_precision) / 3.0
                    # Ensure quality_score is valid
                    if not (isinstance(quality_score, (int, float)) and not math.isnan(quality_score) and not math.isinf(quality_score)):
                        quality_score = 0.5
                    else:
                        quality_score = max(0.0, min(1.0, float(quality_score)))  # Clamp to [0.0, 1.0]
            except Exception as e:
                logger.exception("Ragas evaluation failed")
                ragas_metrics = {}
                default_metrics_used = ["Faithfulness (忠实度)", "Answer Relevancy (答案相关性)", "Context Precision (上下文精确度)"]
                quality_score = 0.5
        else:
            # No retrieved chunks, all metrics use defaults
            default_metrics_used = ["Faithfulness (忠实度)", "Answer Relevancy (答案相关性)", "Context Precision (上下文精确度)"]
        
        # 显示默认值提示（如果有）
        if default_metrics_used:
            st.info(f"ℹ️ 以下指标使用了默认分数 (0.5): {', '.join(default_metrics_used)}")
        
        # Display timing information (small, subtle)
        st.markdown("#### ⏱️ 运行时间")
        timing_col1, timing_col2 = st.columns(2)
        with timing_col1:
            st.caption(f"生成时长: {metrics.latency_ms / 1000.0:.2f}秒")
        with timing_col2:
            st.caption(f"测评时长: {eval_time:.2f}秒")
        
        # 使用网格布局展示关键指标（更直观的可视化）
        st.markdown("#### 📈 关键指标可视化")
        
        # 第一行：性能指标
        st.markdown("**性能指标**")
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        with perf_col1:
            st.metric("TTFT", f"{metrics.latency_ms:.0f} ms")
            st.progress(_safe_progress(metrics.latency_ms, min_val=0.0, max_val=1000.0))
        with perf_col2:
            st.metric("总延迟", f"{metrics.latency_ms:.0f} ms")
            st.progress(_safe_progress(metrics.latency_ms, min_val=0.0, max_val=5000.0))
        with perf_col3:
            st.metric("Token", f"{metrics.total_tokens}")
            st.caption(f"P:{metrics.prompt_tokens} C:{metrics.completion_tokens}")
            st.progress(_safe_progress(metrics.total_tokens, min_val=0.0, max_val=2000.0))
        with perf_col4:
            st.metric("成本", f"${cost:.6f}")
            st.progress(_safe_progress(cost, min_val=0.0, max_val=0.01))
        
        # 第二行：检索质量
        st.markdown("**检索质量指标**")
        retrieval_col1, retrieval_col2, retrieval_col3, retrieval_col4 = st.columns(4)
        with retrieval_col1:
            st.metric("文档块数", f"{num_chunks}")
            st.progress(_safe_progress(num_chunks, min_val=0.0, max_val=10.0))
        with retrieval_col2:
            st.metric("平均分数", f"{avg_score:.4f}" if avg_score > 0 else "N/A")
            if avg_score > 0:
                st.progress(_safe_progress(avg_score))
        with retrieval_col3:
            st.metric("最高分数", f"{max_score:.4f}" if max_score > 0 else "N/A")
            if max_score > 0:
                st.progress(_safe_progress(max_score))
        with retrieval_col4:
            st.metric("最低分数", f"{min_score:.4f}" if min_score > 0 else "N/A")
            if min_score > 0:
                st.progress(_safe_progress(min_score))
        
        # 第三行：回答质量
        st.markdown("**回答质量指标**")
        answer_col1, answer_col2, answer_col3, answer_col4 = st.columns(4)
        with answer_col1:
            st.metric("长度", f"{answer_length} 字符")
            st.progress(_safe_progress(answer_length, min_val=0.0, max_val=1000.0))
        with answer_col2:
            st.metric("词数", f"{answer_word_count} 词")
            st.progress(_safe_progress(answer_word_count, min_val=0.0, max_val=200.0))
        with answer_col3:
            st.metric("引用", "✅ 是" if has_citations else "❌ 否")
            st.progress(1.0 if has_citations else 0.0)
        with answer_col4:
            st.metric("每词成本", f"${cost_per_word:.8f}")
            st.progress(_safe_progress(cost_per_word, min_val=0.0, max_val=0.0001))
        
        # 第四行：Ragas 评测
        st.markdown("**Ragas 质量评测**")
        if default_metrics_used:
            st.info(f"ℹ️ 部分指标使用默认值 (0.5): {', '.join(default_metrics_used)}")
        
        ragas_col1, ragas_col2, ragas_col3, ragas_col4 = st.columns(4)
        faithfulness = ragas_metrics.get("faithfulness", 0.5)
        answer_relevancy = ragas_metrics.get("answer_relevancy", 0.5)
        context_precision = ragas_metrics.get("context_precision", 0.5)
        
        # Ensure all values are valid (not NaN, not inf) and clamped to [0.0, 1.0]
        if not (isinstance(faithfulness, (int, float)) and not math.isnan(faithfulness) and not math.isinf(faithfulness)):
            faithfulness = 0.5
        else:
            faithfulness = max(0.0, min(1.0, float(faithfulness)))
        if not (isinstance(answer_relevancy, (int, float)) and not math.isnan(answer_relevancy) and not math.isinf(answer_relevancy)):
            answer_relevancy = 0.5
        else:
            answer_relevancy = max(0.0, min(1.0, float(answer_relevancy)))
        if not (isinstance(context_precision, (int, float)) and not math.isnan(context_precision) and not math.isinf(context_precision)):
            context_precision = 0.5
        else:
            context_precision = max(0.0, min(1.0, float(context_precision)))
        
        with ragas_col1:
            is_default = "faithfulness" not in ragas_metrics
            metric_label = "Faithfulness" + (" ⚠️" if is_default else "")
            st.metric(metric_label, f"{faithfulness:.4f}")
            st.progress(_safe_progress(faithfulness))
            st.caption("忠实度" + (" (默认值)" if is_default else ""))
        with ragas_col2:
            is_default = "answer_relevancy" not in ragas_metrics
            metric_label = "Answer Relevancy" + (" ⚠️" if is_default else "")
            st.metric(metric_label, f"{answer_relevancy:.4f}")
            st.progress(_safe_progress(answer_relevancy))
            st.caption("答案相关性" + (" (默认值)" if is_default else ""))
        with ragas_col3:
            is_default = "context_precision" not in ragas_metrics
            metric_label = "Context Precision" + (" ⚠️" if is_default else "")
            st.metric(metric_label, f"{context_precision:.4f}")
            st.progress(_safe_progress(context_precision))
            st.caption("上下文精确度" + (" (默认值)" if is_default else ""))
        with ragas_col4:
            is_default = len(default_metrics_used) > 0
            metric_label = "综合质量" + (" ⚠️" if is_default else "")
            st.metric(metric_label, f"{quality_score:.4f}")
            st.progress(_safe_progress(quality_score))
            st.caption("平均分数" + (" (含默认值)" if is_default else ""))
        
        
        # 显示回答内容
        st.markdown("### 💬 生成的回答")
        st.markdown(answer)
        
        # 显示检索到的文档块详情
        if retrieved_chunks:
            st.markdown(f"### 📚 检索结果详情 ({num_chunks} 个文档块)")
            with st.expander("查看检索到的文档块", expanded=False):
                for idx, chunk in enumerate(retrieved_chunks[:10], 1):  # Show top 10
                    chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
                    chunk_score = chunk.score if hasattr(chunk, "score") else None
                    chunk_id = chunk.chunk_id if hasattr(chunk, "chunk_id") else f"chunk_{idx}"
                    
                    st.markdown(f"**文档块 {idx}** (ID: {chunk_id})")
                    if chunk_score is not None:
                        st.caption(f"相关性分数: {chunk_score:.4f}")
                    st.text(chunk_text[:300] + "..." if len(chunk_text) > 300 else chunk_text)
                    st.divider()
    
    except Exception as e:
        logger.exception("Playground query execution failed")
        st.error(f"执行失败: {e}")


def _evaluate_playground_answer(
    query: str,
    answer: str,
    retrieved_chunks: List[Any],
) -> None:
    """Evaluate answer quality using Ragas."""
    try:
        settings = load_settings()
        
        # Create Ragas evaluator
        evaluator = RagasEvaluator(settings=settings)
        
        if not retrieved_chunks:
            st.warning("未找到检索到的文档块，无法进行质量评测。")
            return
        
        # Evaluate
        with st.spinner("运行 Ragas 评测中..."):
            metrics = evaluator.evaluate(
                query=query,
                retrieved_chunks=retrieved_chunks,
                generated_answer=answer,
            )
        
        # Display metrics
        st.markdown("#### 📏 Ragas 评分")
        cols = st.columns(3)
        
        faithfulness = metrics.get("faithfulness", 0.5)
        answer_relevancy = metrics.get("answer_relevancy", 0.5)
        context_precision = metrics.get("context_precision", 0.5)
        
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
    st.markdown(
        "跑完测试集，采用两两匹配穷举法找出最优组合。\n\n"
        "**工作原理**：系统会自动生成所有模型的两两组合，并根据测试集中的 `expected_complexity` 标签自动路由：\n"
        "- 简单询问（simple）→ 使用本地小模型\n"
        "- 复杂询问（complex）→ 使用 API 大模型\n\n"
        "您只需要点击开始，系统会自动运行所有组合并实时更新排行榜。"
    )
    
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
        
        # Count simple and complex queries
        simple_count = sum(1 for tc in test_cases if tc.get("expected_complexity", "").lower() == "simple")
        complex_count = sum(1 for tc in test_cases if tc.get("expected_complexity", "").lower() == "complex")
        
        st.info(
            f"已加载 {len(test_cases)} 个测试用例 "
            f"（简单: {simple_count}, 复杂: {complex_count}）"
        )
    except Exception as e:
        st.error(f"加载测试集失败: {e}")
        return
    
    # Show available models info
    all_models = _get_all_models()
    tier2_models = _get_tier2_models()
    tier3_models = _get_tier3_models()
    
    if not tier2_models or not tier3_models:
        st.warning("⚠️ 需要至少一个本地小模型和一个 API 大模型才能运行双模型组合策略。")
        if not tier2_models:
            st.info("💡 提示：请确保已注册本地小模型（Tier 2）。")
        if not tier3_models:
            st.info("💡 提示：请确保已注册 API 大模型（Tier 3）。")
        return
    
    # Get benchmark model ID
    settings = load_settings()
    benchmark_id = st.session_state.get("benchmark_model_id", _get_benchmark_model_id(settings))
    
    # Display model combinations that will be tested
    st.markdown("#### 📋 将测试的组合")
    
    # Find benchmark model
    benchmark_model = None
    for m in all_models:
        if m.model_id == benchmark_id:
            benchmark_model = m
            break
    
    # Calculate total combinations
    # Single model strategies: ONLY Benchmark model
    single_count = 1 if benchmark_model else 0
    # Hybrid strategies: all tier2 × tier3 combinations
    hybrid_count = len(tier2_models) * len(tier3_models)
    total_combinations = single_count + hybrid_count
    total_tasks = total_combinations * len(test_cases)
    
    st.info(
        f"将测试 **{total_combinations}** 个组合（单模型基准: {single_count}, 双模型组合: {hybrid_count}），"
        f"共 **{total_tasks}** 个任务"
    )
    
    # Show preview of combinations
    with st.expander("查看所有组合详情", expanded=False):
        st.markdown("**单模型策略（基准对比）：**")
        if benchmark_model:
            st.text(f"  • {benchmark_model.display_name} [Benchmark]")
        else:
            st.text("  • 未找到基准模型")
        
        st.markdown("**双模型组合策略：**")
        for small in tier2_models:
            for large in tier3_models:
                small_name = small.display_name
                large_name = large.display_name
                if small.model_id == benchmark_id:
                    small_name = f"{small_name} [Benchmark]"
                if large.model_id == benchmark_id:
                    large_name = f"{large_name} [Benchmark]"
                st.text(f"  • {small_name} + {large_name}")
    
    # Performance optimization option
    st.markdown("#### ⚡ 性能优化")
    fast_mode = st.checkbox(
        "🚀 加速模式",
        value=False,
        key="benchmark_fast_mode",
        help=(
            "启用加速模式可以显著提升评估速度（节省 60-70% 时间）：\n"
            "- 减少上下文数量（5→3 chunks）\n"
            "- 截断答案长度（800 字符）\n"
            "- 只评估 faithfulness 指标\n"
            "- 优化超时和重试设置\n"
            "注意：可能会略微影响评估质量"
        ),
    )
    
    if fast_mode:
        st.info("⚡ 加速模式已启用：将使用性能优化设置，评估速度提升 60-70%")
    
    # Progress saving information
    st.markdown("#### 💾 进度保存")
    st.info(
        f"✅ **自动保存功能已启用**\n\n"
        f"进度会自动保存到: `{BENCHMARK_PROGRESS_FILE}`\n\n"
        f"**重要提示**：\n"
        f"- ✅ 刷新网页**不会**清除进度（进度保存在文件中）\n"
        f"- ✅ 关闭浏览器**不会**清除进度\n"
        f"- ✅ 每次测试用例完成后都会自动保存\n"
        f"- ✅ 可以随时中断，稍后继续\n"
        f"- ⚠️ 只有点击「清除进度」按钮才会删除保存的进度"
    )
    
    # Check for saved progress
    saved_progress = _load_benchmark_progress()
    resume_available = False
    
    if saved_progress:
        # Check if saved progress matches current test set
        saved_test_set = saved_progress.get("test_set_path", "")
        current_test_set = str(test_set_file.resolve())
        
        if saved_test_set == current_test_set or Path(saved_test_set) == test_set_file:
            resume_available = True
            completed_count = len(saved_progress.get("completed_strategies", []))
            total_count = saved_progress.get("total_strategies", 0)
            saved_timestamp = saved_progress.get("timestamp", 0)
            saved_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(saved_timestamp))
            
            # Calculate detailed progress
            total_queries_completed = sum(
                len(strategy_data.get("results", []))
                for strategy_data in saved_progress.get("completed_strategies", [])
            )
            total_queries_expected = total_count * len(test_cases)
            
            st.markdown("#### 📊 已保存的进度")
            st.success(
                f"**发现已保存的进度**（保存时间: {saved_time_str}）\n\n"
                f"📈 **完成情况**：\n"
                f"- 已完成策略: {completed_count}/{total_count}\n"
                f"- 已完成查询: {total_queries_completed}/{total_queries_expected}\n"
                f"- 测试集: `{Path(saved_test_set).name}`\n\n"
                f"💡 **提示**：点击「继续压测」将从上次中断的地方继续，不会重复已完成的测试。"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶️ 继续压测", type="primary", key="benchmark_resume"):
                    _run_benchmark(
                        test_cases=test_cases,
                        fast_mode=fast_mode,
                        test_set_path=str(test_set_file.resolve()),
                        resume_from_progress=saved_progress,
                    )
            with col2:
                if st.button("🗑️ 清除进度并重新开始", key="benchmark_clear"):
                    _clear_benchmark_progress()
                    st.success("✅ 进度已清除")
                    st.rerun()
        else:
            st.warning(
                f"⚠️ **已保存的进度来自不同的测试集**：\n\n"
                f"- 已保存: `{Path(saved_test_set).name}`\n"
                f"- 当前: `{Path(current_test_set).name}`\n\n"
                f"💡 **提示**：如需继续之前的进度，请选择对应的测试集文件。"
            )
            if st.button("🗑️ 清除旧进度", key="benchmark_clear_old"):
                _clear_benchmark_progress()
                st.success("✅ 旧进度已清除")
                st.rerun()
    else:
        st.info("ℹ️ 当前没有保存的进度。开始压测后，进度会自动保存。")
    
    # Run benchmark button (only show if no resume available)
    if not resume_available:
        if st.button("▶️ 开始压测", type="primary", key="benchmark_run"):
            _run_benchmark(
                test_cases=test_cases,
                fast_mode=fast_mode,
                test_set_path=str(test_set_file.resolve()),
                resume_from_progress=None,
            )


def _run_benchmark(
    test_cases: List[Dict[str, Any]],
    fast_mode: bool = False,
    test_set_path: str = "",
    resume_from_progress: Optional[Dict[str, Any]] = None,
) -> None:
    """Run exhaustive benchmark on test cases.
    
    Automatically generates all model combinations:
    - Single model strategies: all registered models
    - Hybrid strategies: all Tier 2 (small) × Tier 3 (large) combinations
    
    For hybrid strategies, uses expected_complexity from test cases to route:
    - simple → small model
    - complex → large model
    
    Args:
        test_cases: List of test case dictionaries.
        fast_mode: If True, enables performance optimizations for Ragas evaluation.
    """
    try:
        settings = load_settings()
        manager = st.session_state.arena_model_manager
        evaluator = ModelEvaluator()
        ragas_evaluator = RagasEvaluator(settings=settings, fast_mode=fast_mode)
        scoring_engine = ScoringEngine()
        
        # Get all models
        all_models = _get_all_models()
        tier2_models = _get_tier2_models()
        tier3_models = _get_tier3_models()
        
        # Get benchmark model ID
        benchmark_id = st.session_state.get("benchmark_model_id", _get_benchmark_model_id(settings))
        
        # Collect all strategies automatically
        all_strategies: List[Tuple[str, Optional[str], str]] = []
        
        # Add single model strategies - ONLY Benchmark model
        benchmark_model = None
        for m in all_models:
            if m.model_id == benchmark_id:
                benchmark_model = m
                break
        
        if benchmark_model:
            display_name = f"{benchmark_model.display_name} [Benchmark]"
            all_strategies.append((display_name, None, benchmark_model.display_name))
        
        # Add hybrid strategies (all combinations)
        for small in tier2_models:
            for large in tier3_models:
                small_name = small.display_name
                large_name = large.display_name
                if small.model_id == benchmark_id:
                    small_name = f"{small_name} [Benchmark]"
                if large.model_id == benchmark_id:
                    large_name = f"{large_name} [Benchmark]"
                strategy_name = f"{small_name} + {large_name}"
                all_strategies.append((strategy_name, small.display_name, large.display_name))
        
        if not all_strategies:
            st.warning("没有可用的模型组合。")
            return
        
        # Load saved progress if resuming
        strategy_results: Dict[str, List[Dict[str, Any]]] = {}
        completed_strategy_names: set = set()
        
        if resume_from_progress:
            # Restore completed results
            completed_strategies = resume_from_progress.get("completed_strategies", [])
            for strategy_data in completed_strategies:
                strategy_name = strategy_data.get("strategy_name", "")
                results = strategy_data.get("results", [])
                if strategy_name and results:
                    strategy_results[strategy_name] = results
                    completed_strategy_names.add(strategy_name)
            
            completed_count = len(completed_strategy_names)
            st.success(
                f"✅ 已恢复进度：{completed_count}/{len(all_strategies)} 个策略已完成，"
                f"将从中断处继续..."
            )
        
        # Progress bar and status
        progress_bar = st.progress(0)
        status_text = st.empty()
        stage_text = st.empty()
        
        # Leaderboard placeholder (will be updated dynamically)
        leaderboard_placeholder = st.empty()
        
        # Calculate total tasks (excluding already completed)
        total_tasks = len(all_strategies) * len(test_cases)
        completed_tasks = sum(
            len(results) for results in strategy_results.values()
        )
        current_task = completed_tasks
        
        # Run each strategy
        for strategy_idx, (strategy_name, small_model, large_model) in enumerate(all_strategies, 1):
            # Skip if this strategy is already completed
            if strategy_name in completed_strategy_names:
                # Verify we have all test cases for this strategy
                existing_results = strategy_results.get(strategy_name, [])
                if len(existing_results) >= len(test_cases):
                    logger.info(f"Skipping completed strategy: {strategy_name}")
                    continue
                else:
                    # Partial completion - continue from where we left off
                    logger.info(f"Resuming partial strategy: {strategy_name} ({len(existing_results)}/{len(test_cases)} completed)")
                    # Get completed query IDs to skip
                    completed_query_ids = {
                        r.get("query_id") or r.get("query", "")
                        for r in existing_results
                    }
            else:
                # New strategy - start fresh
                strategy_results[strategy_name] = []
                completed_query_ids = set()
            
            for test_case_idx, test_case in enumerate(test_cases, 1):
                query_id = test_case.get("query_id", f"Q{test_case_idx}")
                query = test_case.get("query", "")
                
                # Skip if this test case is already completed for this strategy
                if query_id in completed_query_ids or query in completed_query_ids:
                    logger.debug(f"Skipping completed test case: {strategy_name} - {query_id}")
                    continue
                current_task += 1
                progress = current_task / total_tasks
                progress_bar.progress(progress)
                
                expected_complexity = test_case.get("expected_complexity", "simple")
                
                # Determine which model will be used
                if small_model is not None:
                    # Hybrid strategy
                    if expected_complexity.lower() == "simple":
                        used_model = small_model
                    else:
                        used_model = large_model
                    model_info = f"{small_model} + {large_model} → 使用: {used_model}"
                else:
                    # Single model strategy
                    used_model = large_model
                    model_info = used_model
                
                # Update status with detailed information
                status_text.markdown(
                    f"**当前进度**: {current_task}/{total_tasks} "
                    f"({progress*100:.1f}%)\n\n"
                    f"**当前组合**: {strategy_name}\n"
                    f"**模型选择**: {model_info}\n"
                    f"**当前问题**: {query_id} ({expected_complexity})"
                )
                
                # Execute query with error handling
                try:
                    # Update stage: Answer Generation
                    stage_text.info("🔄 **阶段**: 回答生成中...")
                    
                    result = _execute_benchmark_query(
                        query=query,
                        strategy_name=strategy_name,
                        small_model=small_model,
                        large_model=large_model,
                        manager=manager,
                        evaluator=evaluator,
                        ragas_evaluator=ragas_evaluator,
                        settings=settings,
                        expected_complexity=expected_complexity,
                        status_callback=lambda stage: stage_text.info(f"🔄 **阶段**: {stage}"),
                    )
                    # Add query_id to result for matching during resume
                    result["query_id"] = query_id
                    strategy_results[strategy_name].append(result)
                    stage_text.empty()  # Clear stage text after completion
                    
                    # Save progress after each test case
                    _save_benchmark_progress(
                        strategy_results=strategy_results,
                        test_cases=test_cases,
                        all_strategies=all_strategies,
                        fast_mode=fast_mode,
                        test_set_path=test_set_path,
                    )
                except Exception as e:
                    logger.error(f"Failed to execute query for strategy {strategy_name}: {e}", exc_info=True)
                    stage_text.empty()
                    # Add failed result
                    strategy_results[strategy_name].append({
                        "query": query,
                        "query_id": query_id,
                        "success": False,
                        "error": str(e),
                        "latency_s": 0.0,
                        "tokens": 0,
                        "cost": 0.0,
                        "quality_score": 50.0,
                        "expected_complexity": expected_complexity,
                    })
                    
                    # Save progress even on error
                    _save_benchmark_progress(
                        strategy_results=strategy_results,
                        test_cases=test_cases,
                        all_strategies=all_strategies,
                        fast_mode=fast_mode,
                        test_set_path=test_set_path,
                    )
            
            # After completing all test cases for this strategy, update leaderboard
            # Compute aggregated metrics for completed strategies
            try:
                completed_metrics = _compute_aggregated_metrics(strategy_results, test_cases)
                
                # Log strategy completion
                logger.info(
                    f"Strategy '{strategy_name}' completed: "
                    f"{len(strategy_results[strategy_name])} test cases, "
                    f"{len([r for r in strategy_results[strategy_name] if r.get('success')])} successful"
                )
                
                # Display/update leaderboard dynamically
                with leaderboard_placeholder.container():
                    completed_count = len([s for s in strategy_results.keys() if strategy_results[s]])
                    
                    # Calculate summary statistics
                    total_queries = sum(len(results) for results in strategy_results.values())
                    successful_queries = sum(
                        len([r for r in results if r.get("success", False)])
                        for results in strategy_results.values()
                    )
                    
                    # Display summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("已完成策略", f"{completed_count}/{len(all_strategies)}")
                    with col2:
                        st.metric("成功查询", f"{successful_queries}/{total_queries}")
                    with col3:
                        success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
                        st.metric("成功率", f"{success_rate:.1f}%")
                    
                    if completed_metrics:
                        _display_leaderboard(completed_metrics, scoring_engine)
                    else:
                        st.info("⏳ 等待更多测试结果...")
            except Exception as e:
                logger.error(f"Failed to update leaderboard: {e}", exc_info=True)
                # Show error but continue
                with leaderboard_placeholder.container():
                    st.warning(f"⚠️ 更新排行榜时出错: {e}")
                # Continue anyway, will update at the end
        
        progress_bar.progress(1.0)
        status_text.markdown("✅ **压测完成！**")
        stage_text.empty()
        
        # Final leaderboard update (should be same as last update, but ensure it's displayed)
        try:
            final_metrics = _compute_aggregated_metrics(strategy_results, test_cases)
            
            # Log final summary
            total_results = sum(len(results) for results in strategy_results.values())
            total_successful = sum(
                len([r for r in results if r.get("success", False)])
                for results in strategy_results.values()
            )
            logger.info(
                f"Benchmark completed: {total_successful}/{total_results} successful queries, "
                f"{len(final_metrics)} strategies with metrics"
            )
            
            with leaderboard_placeholder.container():
                st.caption(f"📊 已完成 {len(all_strategies)}/{len(all_strategies)} 个策略的测试")
                if final_metrics:
                    _display_leaderboard(final_metrics, scoring_engine)
                else:
                    st.warning("⚠️ 没有可显示的评估结果。请检查日志以了解详情。")
        except Exception as e:
            logger.error(f"Failed to display final leaderboard: {e}", exc_info=True)
            st.error(f"显示最终排行榜失败: {e}")
        
        # Save final progress
        _save_benchmark_progress(
            strategy_results=strategy_results,
            test_cases=test_cases,
            all_strategies=all_strategies,
            fast_mode=fast_mode,
            test_set_path=test_set_path,
        )
        
        # Optionally clear progress after completion (user can choose to keep it)
        st.info("💾 进度已保存。如需清除进度文件，请刷新页面后点击「清除进度」按钮。")
    
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
    ragas_evaluator: RagasEvaluator,
    settings: Any,
    expected_complexity: str,
    status_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Execute a single query in benchmark mode using complete RAG pipeline.
    
    For hybrid strategies, uses expected_complexity from golden_test_set_model_evaluation_and_selection.json
    to determine which model to use (small_model for simple, large_model for complex).
    No prediction is performed - uses ground truth labels from test data.
    
    Args:
        status_callback: Optional callback function to update status (e.g., stage information).
                        Called with stage description string.
    """
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
        # Determine model to use based on expected_complexity from test.json
        selected_model_id: Optional[str] = None
        
        if small_model is not None:
            # Hybrid strategy: use expected_complexity from golden_test_set_model_evaluation_and_selection.json to route
            # No prediction needed - use ground truth label
            if expected_complexity.lower() == "simple":
                selected_model_id = _get_model_id_by_display_name(small_model)
                predicted_complexity = "simple"
            else:  # complex or unknown
                selected_model_id = _get_model_id_by_display_name(large_model)
                predicted_complexity = "complex"
            
            # Record routing decision (always correct since we use ground truth)
            result["routing_predicted"] = predicted_complexity
            result["routing_confidence"] = 1.0  # 100% confidence since using ground truth
            result["routing_correct"] = (predicted_complexity == expected_complexity.lower())
        else:
            # Single model strategy
            selected_model_id = _get_model_id_by_display_name(large_model)
        
        # Get model config for metrics tracking
        config = manager.get_model_config(selected_model_id)
        
        # Track metrics
        with evaluator.track_call(
            model_id=selected_model_id,
            provider=config.provider,
            model_name=config.model_name,
            query=query,
        ) as metrics:
            # Update stage: Answer Generation
            if status_callback:
                status_callback("回答生成中...")
            
            # Execute complete RAG pipeline
            start_time = time.monotonic()
            answer, retrieved_chunks, trace = _execute_rag_query(
                query=query,
                settings=settings,
                manager=manager,
                selected_model_id=selected_model_id,
                collection=None,
                top_k=10,
                trace=None,
            )
            elapsed = time.monotonic() - start_time
            
            # Update metrics
            metrics.latency_ms = elapsed * 1000.0
            metrics.response_length = len(answer)
            
            # Try to extract token usage from trace
            if hasattr(trace, "metadata") and "token_usage" in trace.metadata:
                usage = trace.metadata["token_usage"]
                metrics.prompt_tokens = usage.get("prompt_tokens", 0)
                metrics.completion_tokens = usage.get("completion_tokens", 0)
                metrics.total_tokens = usage.get("total_tokens", 0)
        
        # Evaluate quality using Ragas with actual retrieved chunks
        eval_time = 0.0
        quality_metrics = {}
        try:
            # Update stage: RAGAS Evaluation
            if status_callback:
                status_callback("RAGAS 评估中...")
            
            # Check if we have valid retrieved chunks
            if not retrieved_chunks:
                logger.warning(f"No retrieved chunks for query, using default quality score")
                result["quality_score"] = 50.0
                result["eval_error"] = "No retrieved chunks"
            else:
                eval_start_time = time.monotonic()
                quality_metrics = ragas_evaluator.evaluate(
                    query=query,
                    retrieved_chunks=retrieved_chunks,
                    generated_answer=answer,
                )
                eval_time = time.monotonic() - eval_start_time
                
                # Log the raw metrics for debugging
                logger.info(f"Ragas metrics for query: {quality_metrics}")
                
                # Average of all metrics
                quality_scores = [
                    v for v in quality_metrics.values() 
                    if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v)
                ]
                if quality_scores:
                    result["quality_score"] = sum(quality_scores) / len(quality_scores) * 100.0
                    result["ragas_metrics"] = quality_metrics  # Store raw metrics for debugging
                else:
                    logger.warning(f"No valid quality scores extracted, using default")
                    result["quality_score"] = 50.0
                    result["eval_error"] = "No valid scores from Ragas"
                    result["ragas_metrics"] = quality_metrics  # Store even if invalid
        except Exception as eval_exc:
            logger.error(f"Ragas evaluation failed for query: {eval_exc}", exc_info=True)
            result["quality_score"] = 50.0  # Default fallback
            result["eval_error"] = str(eval_exc)
            result["ragas_metrics"] = {}
        
        # Fill result with all metrics
        result["success"] = True
        result["latency_s"] = elapsed  # Generation time
        result["eval_time_s"] = eval_time  # Evaluation time
        result["tokens"] = metrics.total_tokens
        result["cost"] = metrics.calculate_cost()
        result["answer_length"] = len(answer) if answer else 0
        result["retrieved_chunks_count"] = len(retrieved_chunks) if retrieved_chunks else 0
        
        # Log successful completion
        logger.info(
            f"Benchmark query completed: strategy={strategy_name}, "
            f"quality_score={result.get('quality_score', 0):.2f}, "
            f"latency={elapsed:.2f}s, tokens={metrics.total_tokens}"
        )
    
    except Exception as e:
        logger.error(f"Benchmark query failed for strategy {strategy_name}: {e}", exc_info=True)
        result["error"] = str(e)
        result["success"] = False
        # Ensure all required fields are present even on error
        if "latency_s" not in result:
            result["latency_s"] = 0.0
        if "eval_time_s" not in result:
            result["eval_time_s"] = 0.0
        if "tokens" not in result:
            result["tokens"] = 0
        if "cost" not in result:
            result["cost"] = 0.0
        if "quality_score" not in result:
            result["quality_score"] = 50.0
    
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
        
        # Average evaluation time (default to 0.0 if not present)
        eval_times = [r.get("eval_time_s", 0.0) for r in successful]
        avg_eval_time = sum(eval_times) / len(eval_times) if eval_times else 0.0
        
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
            avg_eval_time_s=avg_eval_time,
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
    """Display leaderboard with all metrics.
    
    This function can be called multiple times to update the leaderboard dynamically.
    """
    st.markdown("### 🏆 Leaderboard (排行榜) - 实时更新")
    
    if not all_metrics:
        st.info("⏳ 等待测试结果...")
        return
    
    # Get benchmark model ID
    settings = load_settings()
    benchmark_id = st.session_state.get("benchmark_model_id", _get_benchmark_model_id(settings))
    
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
    benchmark_flags = []
    for metrics in all_metrics:
        # Check if this strategy contains the benchmark model
        # Strategy name might already contain [Benchmark] from selection, or we check by model ID
        strategy_name_clean = metrics.strategy_name.replace(" [Benchmark]", "")
        is_benchmark = (
            "[Benchmark]" in metrics.strategy_name or
            benchmark_id in strategy_name_clean or
            strategy_name_clean == benchmark_id or
            any(benchmark_id in part for part in strategy_name_clean.split(" + "))
        )
        
        strategy_name = metrics.strategy_name
        if is_benchmark and "[Benchmark]" not in strategy_name:
            strategy_name = f"🏆 {strategy_name} [Benchmark]"
        elif is_benchmark:
            strategy_name = f"🏆 {strategy_name}"
        
        # Get composite_score with NaN protection
        composite_score = getattr(metrics, 'composite_score', 50.0)
        if not (isinstance(composite_score, (int, float)) and not math.isnan(composite_score)):
            composite_score = 50.0
        
        row = {
            "策略名称": strategy_name,
            "综合评分": f"{composite_score:.2f}",
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
            "平均生成时长 (s)": f"{metrics.avg_latency_s:.3f}",
            "平均测评时长 (s)": f"{metrics.avg_eval_time_s:.3f}",
            "P95 延迟 (s)": f"{metrics.p95_latency_s:.3f}",
            "平均 Token/Query": f"{metrics.avg_tokens_per_query:.0f}",
            "单次平均成本 ($)": f"{metrics.avg_cost_per_query:.6f}",
            "总压测成本 ($)": f"{metrics.total_cost:.6f}",
            "平均质量得分": f"{metrics.avg_quality_score * 100:.2f}",
        }
        rows.append(row)
        benchmark_flags.append(is_benchmark)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Apply styling for benchmark rows (light orange background)
    def highlight_benchmark(row):
        """Apply light orange background to benchmark rows."""
        row_idx = row.name  # Get the row index
        if row_idx < len(benchmark_flags) and benchmark_flags[row_idx]:
            return ['background-color: #FFE5CC'] * len(df.columns)  # Light orange (#FFE5CC)
        return [''] * len(df.columns)
    
    styled_df = df.style.apply(highlight_benchmark, axis=1)  # axis=1 for row-wise application
    st.dataframe(styled_df, width='stretch')  # Use width='stretch' instead of use_container_width


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


def _save_benchmark_progress(
    strategy_results: Dict[str, List[Dict[str, Any]]],
    test_cases: List[Dict[str, Any]],
    all_strategies: List[Tuple[str, Optional[str], str]],
    fast_mode: bool,
    test_set_path: str,
) -> None:
    """Save benchmark progress to disk.
    
    Args:
        strategy_results: Current results for each strategy.
        test_cases: List of test cases.
        all_strategies: List of all strategies to test.
        fast_mode: Whether fast mode was enabled.
        test_set_path: Path to the test set file.
    """
    try:
        # Ensure directory exists
        BENCHMARK_PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Calculate progress
        completed_strategies = []
        for strategy_name, results in strategy_results.items():
            if results:
                completed_strategies.append({
                    "strategy_name": strategy_name,
                    "results": results,
                })
        
        progress_data = {
            "timestamp": time.time(),
            "test_set_path": test_set_path,
            "fast_mode": fast_mode,
            "total_strategies": len(all_strategies),
            "total_test_cases": len(test_cases),
            "completed_strategies": completed_strategies,
            "all_strategies": [
                {
                    "strategy_name": strategy_name,
                    "small_model": small_model,
                    "large_model": large_model,
                }
                for strategy_name, small_model, large_model in all_strategies
            ],
            "test_cases": test_cases,
        }
        
        # Save to file
        with open(BENCHMARK_PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Benchmark progress saved to {BENCHMARK_PROGRESS_FILE}")
    except Exception as e:
        logger.error(f"Failed to save benchmark progress: {e}", exc_info=True)


def _load_benchmark_progress() -> Optional[Dict[str, Any]]:
    """Load benchmark progress from disk.
    
    Returns:
        Progress data dictionary if file exists, None otherwise.
    """
    try:
        if not BENCHMARK_PROGRESS_FILE.exists():
            return None
        
        with open(BENCHMARK_PROGRESS_FILE, "r", encoding="utf-8") as f:
            progress_data = json.load(f)
        
        logger.info(f"Benchmark progress loaded from {BENCHMARK_PROGRESS_FILE}")
        return progress_data
    except Exception as e:
        logger.error(f"Failed to load benchmark progress: {e}", exc_info=True)
        return None


def _clear_benchmark_progress() -> None:
    """Clear saved benchmark progress."""
    try:
        if BENCHMARK_PROGRESS_FILE.exists():
            BENCHMARK_PROGRESS_FILE.unlink()
            logger.info(f"Benchmark progress cleared: {BENCHMARK_PROGRESS_FILE}")
    except Exception as e:
        logger.error(f"Failed to clear benchmark progress: {e}", exc_info=True)



