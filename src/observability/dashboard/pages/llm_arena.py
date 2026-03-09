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
from src.observability.dashboard.i18n import t
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
BENCHMARK_HISTORY_FILE = BENCHMARK_PROGRESS_DIR / "benchmark_history.jsonl"
PENDING_BENCHMARK_SAVE_KEY = "llm_arena_pending_benchmark_save"

# LLM Arena legacy scoring weights (pre relative-baseline version)
ARENA_COST_WEIGHT = 0.33
ARENA_LATENCY_WEIGHT = 0.33
ARENA_QUALITY_WEIGHT = 0.34

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
    st.header(t("🏟️ LLM Arena", "🏟️ LLM 竞技场"))
    st.markdown(
        t(
            "Use quantitative metrics such as cost, latency, quality, and **fine-grained routing accuracy** to validate the value of a **small-model + large-model hybrid strategy** in end-user scenarios.",
            "通过成本、延迟、质量以及**细粒度路由准确率**等量化指标，验证**小模型 + 大模型混合策略**在终端用户场景中的价值。",
        )
    )
    
    # Initialize session state
    if "arena_models_registered" not in st.session_state:
        _initialize_models()
        st.session_state.arena_models_registered = True
    
    # Tab navigation (Exhaustive Benchmark as default)
    tab1, tab2 = st.tabs([t("📊 Exhaustive Benchmark", "📊 穷举压测"), t("🎮 Interactive Playground", "🎮 交互试验台")])
    
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
            st.warning(t("⚠️ Ollama provider is not implemented. Skipping local model registration.", "⚠️ Ollama provider 尚未实现，跳过本地模型注册。"))
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
            st.warning(t("⚠️ OpenAI provider is not implemented. Skipping API model registration.", "⚠️ OpenAI provider 尚未实现，跳过 API 模型注册。"))
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
                st.warning(t("⚠️ API key was not found. API models may be unavailable.", "⚠️ 未找到 API key，API 模型可能不可用。"))
            
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
                st.warning(t(f"⚠️ {len(skipped_models)} model registrations failed. Check the logs for details.", f"⚠️ 有 {len(skipped_models)} 个模型注册失败，请检查日志。"))
            
            logger.info(f"Registered {registered_count} Tier 3 models successfully")
        
        st.session_state.arena_model_manager = manager
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        st.error(t("Model initialization failed: ", "模型初始化失败：") + str(e))


def _render_interactive_playground() -> None:
    """Render Module A: Interactive Playground."""
    st.subheader(t("🎮 Interactive Playground", "🎮 交互试验台"))
    st.markdown(t("Run a single query to inspect the system behavior interactively.", "运行单条查询，以交互方式观察系统行为。"))
    
    # Strategy selection (always visible)
    st.markdown(t("#### Strategy Configuration", "#### 策略配置"))
    col1, col2 = st.columns([2, 1])
    
    with col1:
        strategy_type = st.selectbox(
            t("Execution strategy", "执行策略"),
            options=[t("Single model", "单模型"), t("Dual-model hybrid strategy", "双模型混合策略")],
            key="playground_strategy_type",
        )
    
    with col2:
        if strategy_type == t("Single model", "单模型"):
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
                t("Select model", "选择模型"),
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
                t("Small model", "小模型"),
                options=small_model_options,
                key="playground_small_model",
            )
            # Remove [Benchmark] marker if present
            small_model = small_model.replace(" [Benchmark]", "")
            
            large_model = st.selectbox(
                t("Large model", "大模型"),
                options=large_model_options,
                key="playground_large_model",
            )
            # Remove [Benchmark] marker if present
            large_model = large_model.replace(" [Benchmark]", "")
    
    st.divider()
    
    # Query input
    st.markdown(t("#### Query Input", "#### 查询输入"))
    query = st.text_input(
        t("Enter query", "输入查询"),
        value="",
        key="playground_query",
        placeholder=t("For example: What is RAG? Or: Explain how hybrid retrieval works.", "例如：什么是 RAG？或者：请解释混合检索是如何工作的。"),
    )
    
    # Execute button
    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        execute_clicked = st.button(t("▶️ Run Query", "▶️ 执行查询"), type="primary", key="playground_execute")
    
    if not query:
        st.info(t("💡 Configure a strategy and model selection, then enter a query to start testing.", "💡 请先配置策略和模型，然后输入查询开始测试。"))
    
    if execute_clicked:
        if not query:
            st.warning(t("⚠️ Please enter a query before running.", "⚠️ 请先输入查询再执行。"))
        else:
            _execute_playground_query(
                query=query,
                strategy_type=strategy_type,
                small_model=small_model if strategy_type == t("Dual-model hybrid strategy", "双模型混合策略") else None,
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
    try:
        manager.set_current_model(selected_model_id)
        selected_llm = manager.get_llm()
        
        if selected_llm is None:
            raise ValueError(f"Failed to get LLM for model_id: {selected_model_id}")
    except Exception as llm_exc:
        raise RuntimeError(f"Failed to initialize LLM for {selected_model_id}: {llm_exc}") from llm_exc
    
    # Create RAGGenerator with the selected LLM (not settings.llm)
    try:
        rag_generator = RAGGenerator(
            settings=settings,
            llm=selected_llm,  # Override with selected model
        )
    except Exception as gen_exc:
        raise RuntimeError(f"Failed to create RAGGenerator: {gen_exc}") from gen_exc
    
    # Generate answer using RAGGenerator (same logic as chat interface)
    # But we need to capture token usage, so we'll call LLM directly after building prompt
    # Build context and prompt (same as RAGGenerator does internally)
    try:
        context = rag_generator._build_context(retrieval_results)
        prompt = rag_generator._build_prompt(query, context)
    except Exception as prompt_exc:
        raise RuntimeError(f"Failed to build prompt: {prompt_exc}") from prompt_exc
    
    # Call LLM directly to get full response with usage
    try:
        messages = [Message(role="user", content=prompt)]
        llm_response = selected_llm.chat(messages, trace=trace)
    except Exception as chat_exc:
        logger.error(f"LLM chat failed for model {selected_model_id}: {chat_exc}", exc_info=True)
        raise RuntimeError(f"LLM generation failed: {chat_exc}") from chat_exc
    
    # Extract answer
    llm_answer = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
    if not llm_answer or not llm_answer.strip():
        llm_answer = "Sorry, no answer could be generated. Please try again."
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
        
        if strategy_type == "Dual-model hybrid strategy":
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
        
        st.markdown("### 📊 Overall Evaluation Metrics")
        
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
                with st.spinner("Running Ragas evaluation..."):
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
                        default_metrics_used.append("Faithfulness")
                    if "answer_relevancy" not in ragas_metrics:
                        default_metrics_used.append("Answer Relevancy")
                    if "context_precision" not in ragas_metrics:
                        default_metrics_used.append("Context Precision")
                    
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
                default_metrics_used = ["Faithfulness", "Answer Relevancy", "Context Precision"]
                quality_score = 0.5
        else:
            # No retrieved chunks, all metrics use defaults
            default_metrics_used = ["Faithfulness", "Answer Relevancy", "Context Precision"]
        
        # 显示默认值提示（如果有）
        if default_metrics_used:
            st.info(f"ℹ️ The following metrics are using the default score (0.5): {', '.join(default_metrics_used)}")
        
        # Display timing information (small, subtle)
        st.markdown("#### ⏱️ Runtime")
        timing_col1, timing_col2 = st.columns(2)
        with timing_col1:
            st.caption(f"Generation time: {metrics.latency_ms / 1000.0:.2f}s")
        with timing_col2:
            st.caption(f"Evaluation time: {eval_time:.2f}s")
        
        # 使用网格布局展示关键指标（更直观的可视化）
        st.markdown("#### 📈 Key Metrics Visualization")
        
        # 第一行：性能指标
        st.markdown("**Performance Metrics**")
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        with perf_col1:
            st.metric("TTFT", f"{metrics.latency_ms:.0f} ms")
            st.progress(_safe_progress(metrics.latency_ms, min_val=0.0, max_val=1000.0))
        with perf_col2:
            st.metric(t("Total latency", "总延迟"), f"{metrics.latency_ms:.0f} ms")
            st.progress(_safe_progress(metrics.latency_ms, min_val=0.0, max_val=5000.0))
        with perf_col3:
            st.metric(t("Token", "Token"), f"{metrics.total_tokens}")
            st.caption(f"P:{metrics.prompt_tokens} C:{metrics.completion_tokens}")
            st.progress(_safe_progress(metrics.total_tokens, min_val=0.0, max_val=2000.0))
        with perf_col4:
            st.metric(t("Cost", "成本"), f"${cost:.6f}")
            st.progress(_safe_progress(cost, min_val=0.0, max_val=0.01))
        
        # 第二行：检索质量
        st.markdown("**Retrieval Quality Metrics**")
        retrieval_col1, retrieval_col2, retrieval_col3, retrieval_col4 = st.columns(4)
        with retrieval_col1:
            st.metric(t("Chunks", "文档块数"), f"{num_chunks}")
            st.progress(_safe_progress(num_chunks, min_val=0.0, max_val=10.0))
        with retrieval_col2:
            st.metric(t("Average score", "平均分数"), f"{avg_score:.4f}" if avg_score > 0 else "N/A")
            if avg_score > 0:
                st.progress(_safe_progress(avg_score))
        with retrieval_col3:
            st.metric(t("Highest score", "最高分数"), f"{max_score:.4f}" if max_score > 0 else "N/A")
            if max_score > 0:
                st.progress(_safe_progress(max_score))
        with retrieval_col4:
            st.metric(t("Lowest score", "最低分数"), f"{min_score:.4f}" if min_score > 0 else "N/A")
            if min_score > 0:
                st.progress(_safe_progress(min_score))
        
        # 第三行：回答质量
        st.markdown("**Answer Quality Metrics**")
        answer_col1, answer_col2, answer_col3, answer_col4 = st.columns(4)
        with answer_col1:
            st.metric(t("Length", "长度"), t(f"{answer_length} chars", f"{answer_length} 字符"))
            st.progress(_safe_progress(answer_length, min_val=0.0, max_val=1000.0))
        with answer_col2:
            st.metric(t("Word count", "词数"), t(f"{answer_word_count} words", f"{answer_word_count} 词"))
            st.progress(_safe_progress(answer_word_count, min_val=0.0, max_val=200.0))
        with answer_col3:
            st.metric(t("Citations", "引用"), t("✅ Yes", "✅ 是") if has_citations else t("❌ No", "❌ 否"))
            st.progress(1.0 if has_citations else 0.0)
        with answer_col4:
            st.metric(t("Cost per word", "每词成本"), f"${cost_per_word:.8f}")
            st.progress(_safe_progress(cost_per_word, min_val=0.0, max_val=0.0001))
        
        # 第四行：Ragas 评测
        st.markdown("**Ragas Quality Evaluation**")
        if default_metrics_used:
            st.info(f"ℹ️ Some metrics are using default values (0.5): {', '.join(default_metrics_used)}")
        
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
            st.caption(t("Faithfulness", "忠实度") + (t(" (default)", "（默认值）") if is_default else ""))
        with ragas_col2:
            is_default = "answer_relevancy" not in ragas_metrics
            metric_label = "Answer Relevancy" + (" ⚠️" if is_default else "")
            st.metric(metric_label, f"{answer_relevancy:.4f}")
            st.progress(_safe_progress(answer_relevancy))
            st.caption(t("Answer relevancy", "答案相关性") + (t(" (default)", "（默认值）") if is_default else ""))
        with ragas_col3:
            is_default = "context_precision" not in ragas_metrics
            metric_label = "Context Precision" + (" ⚠️" if is_default else "")
            st.metric(metric_label, f"{context_precision:.4f}")
            st.progress(_safe_progress(context_precision))
            st.caption(t("Context precision", "上下文精确度") + (t(" (default)", "（默认值）") if is_default else ""))
        with ragas_col4:
            is_default = len(default_metrics_used) > 0
            metric_label = "Overall quality" + (" ⚠️" if is_default else "")
            st.metric(metric_label, f"{quality_score:.4f}")
            st.progress(_safe_progress(quality_score))
            st.caption(t("Average score", "平均分数") + (t(" (includes defaults)", "（含默认值）") if is_default else ""))
        
        
        # 显示回答内容
        st.markdown("### 💬 Generated Answer")
        st.markdown(answer)
        
        # 显示检索到的文档块详情
        if retrieved_chunks:
            st.markdown(f"### 📚 Retrieval Result Details ({num_chunks} chunks)")
            with st.expander(t("View retrieved chunks", "查看检索到的文档块"), expanded=False):
                for idx, chunk in enumerate(retrieved_chunks[:10], 1):  # Show top 10
                    chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
                    chunk_score = chunk.score if hasattr(chunk, "score") else None
                    chunk_id = chunk.chunk_id if hasattr(chunk, "chunk_id") else f"chunk_{idx}"
                    
                    st.markdown(f"**Chunk {idx}** (ID: {chunk_id})")
                    if chunk_score is not None:
                        st.caption(f"Relevance score: {chunk_score:.4f}")
                    st.text(chunk_text[:300] + "..." if len(chunk_text) > 300 else chunk_text)
                    st.divider()
    
    except Exception as e:
        logger.exception("Playground query execution failed")
        st.error(f"Execution failed: {e}")


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
            st.warning(t("No retrieved chunks were found, so quality evaluation cannot be performed.", "未找到检索到的文档块，无法进行质量评测。"))
            return
        
        # Evaluate
        with st.spinner("Running Ragas evaluation..."):
            metrics = evaluator.evaluate(
                query=query,
                retrieved_chunks=retrieved_chunks,
                generated_answer=answer,
            )
        
        # Display metrics
        st.markdown("#### 📏 Ragas Scores")
        cols = st.columns(3)
        
        faithfulness = metrics.get("faithfulness", 0.5)
        answer_relevancy = metrics.get("answer_relevancy", 0.5)
        context_precision = metrics.get("context_precision", 0.5)
        
        with cols[0]:
            st.metric(t("Faithfulness", "忠实度"), f"{faithfulness:.4f}")
        with cols[1]:
            st.metric(t("Answer Relevancy", "答案相关性"), f"{answer_relevancy:.4f}")
        with cols[2]:
            st.metric(t("Context Precision", "上下文精确度"), f"{context_precision:.4f}")
    
    except Exception as e:
        logger.exception("Ragas evaluation failed")
        st.error(f"Quality evaluation failed: {e}")


def _create_arena_scoring_engine() -> ScoringEngine:
    """Create scoring engine using legacy LLM Arena weights."""
    return ScoringEngine(
        cost_weight=ARENA_COST_WEIGHT,
        latency_weight=ARENA_LATENCY_WEIGHT,
        quality_weight=ARENA_QUALITY_WEIGHT,
    )


def _render_exhaustive_benchmark() -> None:
    """Render Module B: Exhaustive Benchmark."""
    st.subheader(t("📊 Exhaustive Benchmark", "📊 穷举压测"))
    
    # Add tab for history view
    view_tab1, view_tab2 = st.tabs(["▶️ Run Benchmark", "📈 History"])
    
    with view_tab1:
        _render_benchmark_run()
    
    with view_tab2:
        _render_benchmark_history()


def _render_benchmark_run() -> None:
    """Render the benchmark run interface."""
    st.markdown(
        "Run the full test set and exhaustively evaluate all pairwise combinations to find the best setup.\n\n"
        "**How it works**: the system automatically generates all model pairs and routes each case using the "
        "`expected_complexity` label from the test set:\n"
        "- Simple queries (`simple`) -> local small model\n"
        "- Complex queries (`complex`) -> API large model\n\n"
        "Click start and the system will run every combination and update the leaderboard in real time."
    )

    pending_save = st.session_state.get(PENDING_BENCHMARK_SAVE_KEY)
    if isinstance(pending_save, dict):
        st.info(t("A leaderboard has been generated for this run. You can choose whether to save it to history.", "本次 leaderboard 已生成，你可以选择是否保存到历史记录。"))
        ps1, ps2 = st.columns(2)
        if ps1.button("💾 Save This Result", key="arena_save_latest_result"):
            _save_benchmark_history(
                strategy_results=pending_save["strategy_results"],
                test_cases=pending_save["test_cases"],
                all_strategies=pending_save["all_strategies"],
                final_metrics=pending_save["final_metrics"],
                fast_mode=pending_save["fast_mode"],
                test_set_path=pending_save["test_set_path"],
                run_name=pending_save.get("run_name"),
                run_note=pending_save.get("run_note"),
            )
            st.session_state.pop(PENDING_BENCHMARK_SAVE_KEY, None)
            st.success(t("This result has been saved to history.", "本次结果已保存到历史记录。"))
        if ps2.button("🗑️ Discard This Result", key="arena_discard_latest_result"):
            st.session_state.pop(PENDING_BENCHMARK_SAVE_KEY, None)
            st.warning(t("The current unsaved result has been discarded.", "当前未保存结果已被丢弃。"))
    
    # Test set file uploader (compact)
    col1, col2 = st.columns([3, 1])
    with col1:
        test_set_path = st.text_input(
            "Test set path",
            value=str(DEFAULT_GOLDEN_SET),
            key="benchmark_test_set_path",
            label_visibility="visible",
        )
    
    test_set_file = Path(test_set_path)
    if not test_set_file.exists():
        st.warning(f"⚠️ Test set file does not exist: {test_set_path}")
        return
    
    # Load test set
    try:
        with open(test_set_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        test_cases = test_data.get("test_cases", [])
        
        # Count simple and complex queries
        simple_count = sum(1 for tc in test_cases if tc.get("expected_complexity", "").lower() == "simple")
        complex_count = sum(1 for tc in test_cases if tc.get("expected_complexity", "").lower() == "complex")
        
        with col2:
            st.caption(f"📊 {len(test_cases)} cases ({simple_count} simple / {complex_count} complex)")
    except Exception as e:
        st.error(f"Failed to load test set: {e}")
        return
    
    # Show available models info
    all_models = _get_all_models()
    tier2_models = _get_tier2_models()
    tier3_models = _get_tier3_models()
    
    if not tier2_models or not tier3_models:
        st.warning(t("⚠️ At least one local small model and one API large model are required for the dual-model hybrid strategy.", "⚠️ 运行双模型混合策略至少需要一个本地小模型和一个 API 大模型。"))
        if not tier2_models:
            st.info(t("💡 Tip: make sure at least one local small model (Tier 2) is registered.", "💡 提示：请确保至少注册了一个本地小模型（Tier 2）。"))
        if not tier3_models:
            st.info(t("💡 Tip: make sure at least one API large model (Tier 3) is registered.", "💡 提示：请确保至少注册了一个 API 大模型（Tier 3）。"))
        return
    
    # Get benchmark model ID
    settings = load_settings()
    benchmark_id = st.session_state.get("benchmark_model_id", _get_benchmark_model_id(settings))
    
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
    
    # Compact display of test combinations
    st.caption(f"📋 Testing {total_combinations} combinations (benchmark: {single_count}, hybrid: {hybrid_count}), for {total_tasks} total tasks")
    
    # Show preview of combinations
    with st.expander(t("View all combination details", "查看所有组合详情"), expanded=False):
        st.markdown("**Single-model strategy (benchmark baseline):**")
        if benchmark_model:
            st.text(f"  • {benchmark_model.display_name} [Benchmark]")
        else:
            st.text("  • Benchmark model not found")
        
        st.markdown("**Dual-model hybrid strategies:**")
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
    st.markdown("#### ⚡ Performance Optimization")
    fast_mode = st.checkbox(
        "🚀 Fast mode",
        value=False,
        key="benchmark_fast_mode",
        help=(
            "Fast mode can significantly speed up evaluation (saving about 60-70% time):\n"
            "- Reduce context count (5 -> 3 chunks)\n"
            "- Truncate answer length (800 characters)\n"
            "- Evaluate only the faithfulness metric\n"
            "- Optimize timeout and retry settings\n"
            "Note: this may slightly reduce evaluation quality"
        ),
    )
    
    if fast_mode:
        st.info(t("⚡ Fast mode is enabled: performance-optimized settings will be used, improving evaluation speed by about 60-70%.", "⚡ 已启用加速模式：将使用性能优化设置，评估速度预计提升约 60-70%。"))
    
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
            
            # Check if benchmark is completed
            is_completed = completed_count >= total_count
            
            # Title with help icon (using expander)
            col_title, col_help = st.columns([20, 1])
            with col_title:
                st.markdown("#### 📊 Saved Progress")
            with col_help:
                with st.expander("ⓘ", expanded=False):
                    st.markdown(
                        "✅ **Auto-save is enabled**\n\n"
                        f"Progress is automatically saved to: `{BENCHMARK_PROGRESS_FILE}`\n\n"
                        "**Important notes**:\n"
                        "- ✅ Refreshing the page will **not** clear progress\n"
                        "- ✅ Closing the browser will **not** clear progress\n"
                        "- ✅ Progress is saved automatically after each completed test case\n"
                        "- ✅ You can stop at any time and continue later\n"
                        "- ⚠️ Progress is deleted only when you click the clear button"
                    )
            
            if is_completed:
                # Test is completed, only show completion message and clear button
                # Results will be shown in "历史结果" tab
                st.success(
                    f"✅ **Benchmark completed** (completed at: {saved_time_str})\n\n"
                    f"📈 **Completion summary**:\n"
                    f"- Completed strategies: {completed_count}/{total_count}\n"
                    f"- Completed queries: {total_queries_completed}/{total_queries_expected}\n"
                    f"- Test set: `{Path(saved_test_set).name}`\n\n"
                    f"💡 **Tip**: You can click `Save This Result` here, or switch to `History` to review past runs."
                )

                sv1, sv2 = st.columns(2)
                if sv1.button("💾 Save This Result", key="arena_save_completed_progress"):
                    _save_completed_progress_to_history(saved_progress)
                    st.success(t("The completed result has been saved to history.", "当前完成结果已保存到历史记录。"))
                
                # Clear progress button
                if sv2.button("🗑️ Clear Progress and Start a New Run", key="benchmark_clear_completed"):
                    _clear_benchmark_progress()
                    st.success(t("✅ Progress cleared", "✅ 进度已清除"))
                    st.rerun()
            else:
                # Test is not completed, show resume option
                st.info(
                    f"**Unfinished progress was found** (saved at: {saved_time_str})\n\n"
                    f"📈 **Completion summary**:\n"
                    f"- Completed strategies: {completed_count}/{total_count}\n"
                    f"- Completed queries: {total_queries_completed}/{total_queries_expected}\n"
                    f"- Test set: `{Path(saved_test_set).name}`\n\n"
                    f"💡 **Tip**: Click `Resume Benchmark` to continue from where the last run stopped without repeating completed tests."
                )
                
                # Buttons in columns, but execute outside column context
                col1, col2 = st.columns(2)
                resume_clicked = False
                clear_clicked = False
                
                with col1:
                    resume_clicked = st.button(t("▶️ Resume Benchmark", "▶️ 继续压测"), type="primary", key="benchmark_resume")
                with col2:
                    clear_clicked = st.button(t("🗑️ Clear Progress and Restart", "🗑️ 清除进度并重新开始"), key="benchmark_clear")
                
                # Handle button clicks outside column context to ensure full-width layout
                if clear_clicked:
                    _clear_benchmark_progress()
                    st.success(t("✅ Progress cleared", "✅ 进度已清除"))
                    st.rerun()
                
                if resume_clicked:
                    # Get name and note from session state if available
                    resume_name = st.session_state.get("benchmark_run_name", time.strftime("%Y-%m-%d %H:%M:%S"))
                    resume_note = st.session_state.get("benchmark_run_note", "")
                    
                    # Execute benchmark outside column context - this ensures full-width layout
                    _run_benchmark(
                        test_cases=test_cases,
                        fast_mode=fast_mode,
                        test_set_path=str(test_set_file.resolve()),
                        resume_from_progress=saved_progress,
                        run_name=resume_name,
                        run_note=resume_note,
                    )
        else:
            st.warning(
                f"⚠️ **Saved progress belongs to a different test set**:\n\n"
                f"- Saved: `{Path(saved_test_set).name}`\n"
                f"- Current: `{Path(current_test_set).name}`\n\n"
                f"💡 **Tip**: Select the matching test set file if you want to continue the previous run."
            )
            if st.button(t("🗑️ Clear Old Progress", "🗑️ 清除旧进度"), key="benchmark_clear_old"):
                _clear_benchmark_progress()
                st.success(t("✅ Old progress cleared", "✅ 旧进度已清除"))
                st.rerun()
    else:
        # Show title with help icon even when no progress exists
        col_title, col_help = st.columns([20, 1])
        with col_title:
            st.markdown("#### 📊 Saved Progress")
        with col_help:
            with st.expander("ⓘ", expanded=False):
                st.markdown(
                    "✅ **Auto-save is enabled**\n\n"
                    f"Progress is automatically saved to: `{BENCHMARK_PROGRESS_FILE}`\n\n"
                    "**Important notes**:\n"
                    "- ✅ Refreshing the page will **not** clear progress\n"
                    "- ✅ Closing the browser will **not** clear progress\n"
                    "- ✅ Progress is saved automatically after each completed test case\n"
                    "- ✅ You can stop at any time and continue later\n"
                    "- ⚠️ Progress is deleted only when you click the clear button"
                )
        st.info(t("ℹ️ No saved progress is available yet. Progress will be auto-saved after the benchmark starts.", "ℹ️ 当前没有已保存进度。开始压测后会自动保存进度。"))
    
    # Run benchmark button (only show if no resume available)
    if not resume_available:
        st.markdown("#### 🚀 Start Benchmark")
        
        # Name and note inputs
        col1, col2 = st.columns([1, 1])
        with col1:
            # Default name is current timestamp
            default_name = time.strftime("%Y-%m-%d %H:%M:%S")
            run_name = st.text_input(
                "Run name",
                value=default_name,
                key="benchmark_run_name",
                help="Name for this benchmark run. The current time is used by default."
            )
        with col2:
            run_note = st.text_area(
                "Run notes",
                value="",
                key="benchmark_run_note",
                height=60,
                help="Optional notes about changes, experiments, or context for this run"
            )
        
        if st.button(t("▶️ Start Benchmark", "▶️ 开始压测"), type="primary", key="benchmark_run"):
            # Store name and note in session state for later use
            st.session_state["benchmark_run_name"] = run_name
            st.session_state["benchmark_run_note"] = run_note
            
            _run_benchmark(
                test_cases=test_cases,
                fast_mode=fast_mode,
                test_set_path=str(test_set_file.resolve()),
                resume_from_progress=None,
                run_name=run_name,
                run_note=run_note,
            )


def _run_benchmark(
    test_cases: List[Dict[str, Any]],
    fast_mode: bool = False,
    test_set_path: str = "",
    resume_from_progress: Optional[Dict[str, Any]] = None,
    run_name: Optional[str] = None,
    run_note: Optional[str] = None,
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
        scoring_engine = _create_arena_scoring_engine()
        
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
            st.warning(t("No model combinations are available.", "没有可用的模型组合。"))
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
                f"✅ Progress restored: {completed_count}/{len(all_strategies)} strategies are already complete, "
                f"and the run will continue from the interruption point..."
            )
        
        # Ensure full-width layout from here - all subsequent content should be full-width
        # Progress bar and status (full-width)
        progress_bar = st.progress(0)
        status_text = st.empty()
        stage_text = st.empty()
        
        # Leaderboard placeholder (will be updated dynamically, full-width)
        # Using container() ensures full-width content
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
                    model_info = f"{small_model} + {large_model} -> using: {used_model}"
                else:
                    # Single model strategy
                    used_model = large_model
                    model_info = used_model
                
                # Update status with detailed information
                status_text.markdown(
                    f"**Current progress**: {current_task}/{total_tasks} "
                    f"({progress*100:.1f}%)\n\n"
                    f"**Current combination**: {strategy_name}\n"
                    f"**Model selection**: {model_info}\n"
                    f"**Current query**: {query_id} ({expected_complexity})"
                )
                
                # Execute query with error handling
                try:
                    # Update stage: Answer Generation
                    stage_text.info("🔄 **Stage**: Generating answer...")
                    
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
                        status_callback=lambda stage: stage_text.info(f"🔄 **Stage**: {stage}"),
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
                    logger.error(
                        f"Failed to execute query for strategy {strategy_name}, "
                        f"query '{query[:50]}...': {e}", 
                        exc_info=True
                    )
                    stage_text.empty()
                    # Add failed result with all required fields
                    failed_result = {
                        "query": query,
                        "query_id": query_id,
                        "success": False,
                        "error": str(e)[:200],  # Truncate error message
                        "answer": "",
                        "retrieved_chunks": [],
                        "latency_s": 0.0,
                        "eval_time_s": 0.0,
                        "tokens": 0,
                        "cost": 0.0,
                        "quality_score": 50.0,
                        "expected_complexity": expected_complexity,
                        "routing_predicted": None,
                        "routing_correct": None,
                        "routing_confidence": 0.0,
                    }
                    strategy_results[strategy_name].append(failed_result)
                    
                    # Save progress even for failed queries
                    try:
                        _save_benchmark_progress(
                            strategy_results=strategy_results,
                            test_cases=test_cases,
                            all_strategies=all_strategies,
                            fast_mode=fast_mode,
                            test_set_path=test_set_path,
                        )
                    except Exception as save_exc:
                        logger.warning(f"Failed to save progress after error: {save_exc}")
                    
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
                # Clear the placeholder and render full-width content
                with leaderboard_placeholder.container():
                    completed_count = len([s for s in strategy_results.keys() if strategy_results[s]])
                    
                    # Calculate summary statistics
                    total_queries = sum(len(results) for results in strategy_results.values())
                    successful_queries = sum(
                        len([r for r in results if r.get("success", False)])
                        for results in strategy_results.values()
                    )
                    
                    # Display summary in columns (these are properly scoped)
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    with summary_col1:
                        st.metric(t("Completed strategies", "已完成策略"), f"{completed_count}/{len(all_strategies)}")
                    with summary_col2:
                        st.metric(t("Successful queries", "成功查询"), f"{successful_queries}/{total_queries}")
                    with summary_col3:
                        success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
                        st.metric(t("Success rate", "成功率"), f"{success_rate:.1f}%")
                    
                    # Columns are closed here, leaderboard will be full-width
                    if completed_metrics:
                        _display_leaderboard(completed_metrics, scoring_engine)
                    else:
                        st.info(t("⏳ Waiting for more benchmark results...", "⏳ 正在等待更多压测结果..."))
            except Exception as e:
                logger.error(f"Failed to update leaderboard: {e}", exc_info=True)
                # Show error but continue
                leaderboard_placeholder.empty()
                with leaderboard_placeholder.container():
                    st.warning(f"⚠️ Failed to update leaderboard: {e}")
                # Continue anyway, will update at the end
        
        progress_bar.progress(1.0)
        status_text.markdown("✅ **Benchmark complete!**")
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
            
            # Clear placeholder and render full-width final leaderboard
            leaderboard_placeholder.empty()
            with leaderboard_placeholder.container():
                st.caption(f"📊 Finished testing {len(all_strategies)}/{len(all_strategies)} strategies")
                if final_metrics:
                    _display_leaderboard(final_metrics, scoring_engine)
                else:
                    st.warning(t("⚠️ No evaluation results are available to display. Check the logs for details.", "⚠️ 当前没有可显示的评估结果，请检查日志。"))
        except Exception as e:
            logger.error(f"Failed to display final leaderboard: {e}", exc_info=True)
            st.error(f"Failed to display the final leaderboard: {e}")
        
        # Save final progress
        _save_benchmark_progress(
            strategy_results=strategy_results,
            test_cases=test_cases,
            all_strategies=all_strategies,
            fast_mode=fast_mode,
            test_set_path=test_set_path,
        )
        
        # Keep result as pending until user explicitly chooses "save this result"
        st.session_state[PENDING_BENCHMARK_SAVE_KEY] = {
            "strategy_results": strategy_results,
            "test_cases": test_cases,
            "all_strategies": all_strategies,
            "final_metrics": final_metrics,
            "fast_mode": fast_mode,
            "test_set_path": test_set_path,
            "run_name": run_name,
            "run_note": run_note,
        }
        
        # Optionally clear progress after completion (user can choose to keep it)
        st.info(t("💾 Progress has been saved. Click `Save This Result` if you want to write it into history.", "💾 进度已保存。如需写入历史记录，请点击“保存此次结果”。"))
    
    except Exception as e:
        logger.exception("Benchmark execution failed")
        st.error(f"Benchmark failed: {e}")


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
            
            # Validate model ID was found
            if selected_model_id is None:
                raise ValueError(
                    f"Model not found: {small_model if expected_complexity.lower() == 'simple' else large_model}. "
                    f"Please ensure the model is registered in ModelManager."
                )
            
            # Record routing decision (always correct since we use ground truth)
            result["routing_predicted"] = predicted_complexity
            result["routing_confidence"] = 1.0  # 100% confidence since using ground truth
            result["routing_correct"] = (predicted_complexity == expected_complexity.lower())
        else:
            # Single model strategy
            selected_model_id = _get_model_id_by_display_name(large_model)
            if selected_model_id is None:
                raise ValueError(
                    f"Model not found: {large_model}. "
                    f"Please ensure the model is registered in ModelManager."
                )
        
        # Get model config for metrics tracking
        try:
            config = manager.get_model_config(selected_model_id)
            if config is None:
                raise ValueError(f"Model config not found for model_id: {selected_model_id}")
        except Exception as config_exc:
            raise ValueError(f"Failed to get model config for {selected_model_id}: {config_exc}") from config_exc
        
        # Track metrics
        with evaluator.track_call(
            model_id=selected_model_id,
            provider=config.provider,
            model_name=config.model_name,
            query=query,
        ) as metrics:
            # Update stage: Answer Generation
            if status_callback:
                status_callback("Generating answer...")
            
            # Execute complete RAG pipeline
            start_time = time.monotonic()
            try:
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
                
                # Validate answer was generated
                if not answer or not answer.strip():
                    logger.warning(f"Empty answer generated for query: {query[:50]}...")
                    answer = "Sorry, no valid answer could be generated."
                
                # Validate retrieved chunks
                if not retrieved_chunks:
                    logger.warning(f"No chunks retrieved for query: {query[:50]}...")
            except Exception as rag_exc:
                elapsed = time.monotonic() - start_time
                logger.error(f"RAG pipeline failed for strategy {strategy_name}, query '{query[:50]}...': {rag_exc}", exc_info=True)
                # Don't raise - return error result instead to allow benchmark to continue
                result["error"] = f"RAG pipeline failed: {str(rag_exc)[:200]}"
                result["success"] = False
                result["answer"] = "Sorry, the RAG pipeline failed to execute."
                result["retrieved_chunks"] = []
                result["latency_s"] = elapsed
                result["eval_time_s"] = 0.0
                result["quality_score"] = 50.0
                result["tokens"] = 0
                result["cost"] = 0.0
                return result
            
            # Update metrics (only if metrics object exists)
            if 'metrics' in locals() and metrics is not None:
                try:
                    metrics.latency_ms = elapsed * 1000.0
                    metrics.response_length = len(answer)
                    
                    # Try to extract token usage from trace
                    try:
                        if hasattr(trace, "metadata") and trace.metadata and "token_usage" in trace.metadata:
                            usage = trace.metadata["token_usage"]
                            if isinstance(usage, dict):
                                metrics.prompt_tokens = usage.get("prompt_tokens", 0) or 0
                                metrics.completion_tokens = usage.get("completion_tokens", 0) or 0
                                metrics.total_tokens = usage.get("total_tokens", 0) or 0
                    except Exception as token_exc:
                        logger.warning(f"Failed to extract token usage from trace: {token_exc}")
                        # Continue with default values (0)
                except Exception as metrics_update_exc:
                    logger.warning(f"Failed to update metrics: {metrics_update_exc}")
        
        # Evaluate quality using Ragas with actual retrieved chunks
        eval_time = 0.0
        quality_metrics = {}
        try:
            # Update stage: RAGAS Evaluation
            if status_callback:
                status_callback("Running RAGAS evaluation...")
            
            # Check if we have valid retrieved chunks
            if not retrieved_chunks:
                logger.warning(f"No retrieved chunks for query '{query[:50]}...', using default quality score")
                result["quality_score"] = 50.0
                result["eval_error"] = "No retrieved chunks"
                result["eval_time_s"] = 0.0
            else:
                # Pre-truncate answer to prevent Ragas max_tokens issues
                # RagasEvaluator will handle truncation, but we do it here too for safety
                max_answer_length = 2000  # Conservative limit
                if len(answer) > max_answer_length:
                    logger.debug(f"Pre-truncating answer from {len(answer)} to {max_answer_length} chars before Ragas evaluation")
                    # Try to truncate at sentence boundary
                    truncated = answer[:max_answer_length]
                    last_period = truncated.rfind('.')
                    last_newline = truncated.rfind('\n')
                    cut_point = max(last_period, last_newline)
                    if cut_point > max_answer_length * 0.8:  # Only use if we keep at least 80%
                        answer = truncated[:cut_point + 1]
                    else:
                        answer = truncated + "..."
                
                eval_start_time = time.monotonic()
                try:
                    quality_metrics = ragas_evaluator.evaluate(
                        query=query,
                        retrieved_chunks=retrieved_chunks,
                        generated_answer=answer,
                    )
                    eval_time = time.monotonic() - eval_start_time
                    
                    # Log the raw metrics for debugging (only if valid)
                    if quality_metrics:
                        logger.debug(f"Ragas metrics for query '{query[:50]}...': {quality_metrics}")
                    
                    # Average of all metrics
                    quality_scores = [
                        v for v in quality_metrics.values() 
                        if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v)
                    ]
                    if quality_scores:
                        result["quality_score"] = sum(quality_scores) / len(quality_scores) * 100.0
                        result["ragas_metrics"] = quality_metrics  # Store raw metrics for debugging
                    else:
                        logger.warning(f"No valid quality scores extracted for query '{query[:50]}...', using default")
                        result["quality_score"] = 50.0
                        result["eval_error"] = "No valid scores from Ragas"
                        result["ragas_metrics"] = quality_metrics  # Store even if invalid
                except Exception as inner_eval_exc:
                    eval_time = time.monotonic() - eval_start_time
                    logger.error(f"Ragas evaluation failed for query '{query[:50]}...': {inner_eval_exc}", exc_info=True)
                    result["quality_score"] = 50.0  # Default fallback
                    result["eval_error"] = str(inner_eval_exc)[:200]  # Truncate error message
                    result["ragas_metrics"] = {}
        except Exception as eval_exc:
            logger.error(f"Ragas evaluation exception for query '{query[:50]}...': {eval_exc}", exc_info=True)
            result["quality_score"] = 50.0  # Default fallback
            result["eval_error"] = str(eval_exc)[:200]  # Truncate error message
            result["ragas_metrics"] = {}
            result["eval_time_s"] = 0.0
        
        # Fill result with all metrics
        result["success"] = True
        result["latency_s"] = elapsed  # Generation time
        result["eval_time_s"] = eval_time  # Evaluation time
        
        # Safely extract metrics (only if metrics object exists and was successfully created)
        result["tokens"] = 0
        result["cost"] = 0.0
        try:
            # Try to get from metrics object (if available)
            if 'metrics' in locals() and metrics is not None:
                try:
                    if hasattr(metrics, 'total_tokens'):
                        result["tokens"] = metrics.total_tokens or 0
                    if hasattr(metrics, 'calculate_cost'):
                        result["cost"] = metrics.calculate_cost() or 0.0
                except Exception as metrics_attr_exc:
                    logger.debug(f"Failed to access metrics attributes: {metrics_attr_exc}")
            
            # Fallback: try to extract from trace metadata if metrics not available
            if result["tokens"] == 0 and 'trace' in locals() and trace is not None:
                try:
                    if hasattr(trace, "metadata") and trace.metadata and "token_usage" in trace.metadata:
                        usage = trace.metadata["token_usage"]
                        if isinstance(usage, dict):
                            result["tokens"] = usage.get("total_tokens", 0) or 0
                            # Estimate cost if available
                            if "cost" in usage:
                                result["cost"] = usage.get("cost", 0.0)
                except Exception as trace_exc:
                    logger.debug(f"Failed to extract from trace metadata: {trace_exc}")
        except Exception as metrics_exc:
            logger.warning(f"Failed to extract metrics: {metrics_exc}")
            # Keep default values (0)
        
        result["answer_length"] = len(answer) if answer else 0
        result["retrieved_chunks_count"] = len(retrieved_chunks) if retrieved_chunks else 0
        
        # Log successful completion (only if no errors)
        if result.get("success", False) and not result.get("error"):
            logger.info(
                f"Benchmark query completed: strategy={strategy_name}, "
                f"query='{query[:50]}...', "
                f"quality_score={result.get('quality_score', 0):.2f}, "
                f"latency={elapsed:.2f}s, tokens={result.get('tokens', 0)}"
            )
        elif result.get("error"):
            logger.warning(
                f"Benchmark query completed with error: strategy={strategy_name}, "
                f"query='{query[:50]}...', error={result.get('error', 'Unknown')[:100]}"
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
        
        # Extract RAGAS detailed metrics from ragas_metrics
        faithfulness_scores = []
        answer_relevancy_scores = []
        context_precision_scores = []
        
        for r in successful:
            ragas_metrics = r.get("ragas_metrics", {})
            if isinstance(ragas_metrics, dict):
                # Extract individual RAGAS metrics (they are already 0-1 range)
                if "faithfulness" in ragas_metrics:
                    val = ragas_metrics["faithfulness"]
                    if isinstance(val, (int, float)) and not (isinstance(val, float) and (val != val)):  # Not NaN
                        faithfulness_scores.append(float(val))
                if "answer_relevancy" in ragas_metrics:
                    val = ragas_metrics["answer_relevancy"]
                    if isinstance(val, (int, float)) and not (isinstance(val, float) and (val != val)):  # Not NaN
                        answer_relevancy_scores.append(float(val))
                if "context_precision" in ragas_metrics:
                    val = ragas_metrics["context_precision"]
                    if isinstance(val, (int, float)) and not (isinstance(val, float) and (val != val)):  # Not NaN
                        context_precision_scores.append(float(val))
        
        # Compute averages (use None if no valid scores)
        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None
        avg_answer_relevancy = sum(answer_relevancy_scores) / len(answer_relevancy_scores) if answer_relevancy_scores else None
        avg_context_precision = sum(context_precision_scores) / len(context_precision_scores) if context_precision_scores else None
        
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
            avg_faithfulness=avg_faithfulness,
            avg_answer_relevancy=avg_answer_relevancy,
            avg_context_precision=avg_context_precision,
            routing_total_accuracy=routing_total_accuracy,
            routing_simple_accuracy=routing_simple_accuracy,
            routing_complex_accuracy=routing_complex_accuracy,
            raw_faithfulness_scores=faithfulness_scores,
            raw_answer_relevancy_scores=answer_relevancy_scores,
            raw_context_precision_scores=context_precision_scores,
            raw_latency_scores=[float(r.get("latency_s", 0.0)) for r in successful],
            raw_cost_scores=[float(r.get("cost", 0.0)) for r in successful],
        )
        
        all_metrics.append(metrics)
    
    return all_metrics


def _display_leaderboard(
    all_metrics: List[StrategyMetrics],
    scoring_engine: ScoringEngine,
) -> None:
    """Display leaderboard with all metrics.
    
    This function can be called multiple times to update the leaderboard dynamically.
    All content is rendered in full-width layout.
    """
    # Ensure full-width layout
    st.markdown("### 🏆 Leaderboard - Live Updates")
    
    if not all_metrics:
        st.info(t("⏳ Waiting for benchmark results...", "⏳ 正在等待压测结果..."))
        return
    
    # Display scoring rules and calculation process
    with st.expander(t("📊 Scoring Rules", "📊 评分规则说明"), expanded=False):
        st.markdown("""
        #### Traditional Composite Score (Min-Max Normalization)

        - Cost, latency, and quality are normalized independently with Min-Max scaling
        - Cost and latency are inverse metrics (lower is better)
        - Quality is a positive metric (higher is better)
        - Composite score = weighted sum (0-100)
        """)
        
        # Show current weights
        st.markdown(f"""
        **Current weight configuration:**
        - Cost weight: {scoring_engine.cost_weight * 100:.1f}%
        - Latency weight: {scoring_engine.latency_weight * 100:.1f}%
        - Quality weight: {scoring_engine.quality_weight * 100:.1f}%
        """)
    
    # Model difference explanation
    with st.expander(t("ℹ️ Model Notes", "ℹ️ 模型说明"), expanded=False):
        st.markdown("""
        **Difference between `api-gpt-4o-mini` and `Openai gpt-4o-mini [Benchmark]`:**
        
        - **api-gpt-4o-mini**: accesses OpenAI GPT-4o-mini through the Zhizengzeng proxy (`https://api.zhizengzeng.com/v1`)
        - **Openai gpt-4o-mini [Benchmark]**: the benchmark model configured in `settings.yaml` (possibly direct OpenAI or Azure OpenAI)
        
        Both use the same underlying model (`GPT-4o-mini`), but different API endpoints may create performance differences because of network path and proxy latency.
        """)
    
    # Get benchmark model ID
    settings = load_settings()
    benchmark_id = st.session_state.get("benchmark_model_id", _get_benchmark_model_id(settings))
    
    for metrics in all_metrics:
        metrics.composite_score = scoring_engine.compute_composite_score(metrics, all_metrics)
    
    # Sort by composite score
    all_metrics.sort(key=lambda m: getattr(m, "composite_score", 0.0), reverse=True)
    
    # Build dataframe
    import pandas as pd
    
    def get_rank_label(rank: int) -> str:
        """Convert rank number to a compact English label."""
        suffix = "th"
        if rank % 10 == 1 and rank % 100 != 11:
            suffix = "st"
        elif rank % 10 == 2 and rank % 100 != 12:
            suffix = "nd"
        elif rank % 10 == 3 and rank % 100 != 13:
            suffix = "rd"
        return f"{rank}{suffix}"
    
    rows = []
    benchmark_flags = []
    for rank, metrics in enumerate(all_metrics, start=1):
        # Check if this is a SINGLE model strategy that uses ONLY the benchmark model
        # Single model strategies don't have " + " in their name
        # Only highlight if it's a single model strategy with [Benchmark]
        strategy_name_clean = metrics.strategy_name.replace(" [Benchmark]", "").replace("🏆 ", "")
        is_benchmark = (
            "[Benchmark]" in metrics.strategy_name and
            " + " not in metrics.strategy_name  # Only single model strategies (no " + " separator)
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
            "Rank": get_rank_label(rank),
            "Strategy": strategy_name,
            "Composite Score": f"{composite_score:.2f}",
            "Success Rate (%)": f"{metrics.success_rate * 100:.2f}",
            "Avg Generation Time (s)": f"{metrics.avg_latency_s:.3f}",
            "Avg Evaluation Time (s)": f"{metrics.avg_eval_time_s:.3f}",
            "P95 Latency (s)": f"{metrics.p95_latency_s:.3f}",
            "Avg Tokens / Query": f"{metrics.avg_tokens_per_query:.0f}",
            "Avg Cost / Query ($)": f"{metrics.avg_cost_per_query:.6f}",
            "Total Benchmark Cost ($)": f"{metrics.total_cost:.6f}",
            "Avg Quality Score": f"{metrics.avg_quality_score * 100:.2f}",
            "Faithfulness": (
                f"{metrics.avg_faithfulness * 100:.2f}"
                if metrics.avg_faithfulness is not None
                else "N/A"
            ),
            "Answer Relevancy": (
                f"{metrics.avg_answer_relevancy * 100:.2f}"
                if metrics.avg_answer_relevancy is not None
                else "N/A"
            ),
            "Context Precision": (
                f"{metrics.avg_context_precision * 100:.2f}"
                if metrics.avg_context_precision is not None
                else "N/A"
            ),
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
    
    # Display comparison: First place vs Benchmark
    if len(all_metrics) > 0:
        first_place = all_metrics[0]  # First place (highest composite score)
        
        # Find benchmark strategy (single model strategy with [Benchmark])
        benchmark_strategy = None
        for m in all_metrics:
            if "[Benchmark]" in m.strategy_name and " + " not in m.strategy_name:
                benchmark_strategy = m
                break
        
        if benchmark_strategy and first_place.strategy_name != benchmark_strategy.strategy_name:
            st.markdown("---")
            st.markdown("### 🎯 Top Strategy vs Benchmark")
            
            # Calculate improvements
            def calculate_improvement(new_val: float, old_val: float, higher_is_better: bool = True) -> tuple:
                """Calculate improvement percentage and direction."""
                if old_val == 0:
                    return (float('inf') if new_val > 0 else 0.0, "N/A")
                
                if higher_is_better:
                    improvement = ((new_val - old_val) / old_val) * 100
                    direction = "improved" if improvement > 0 else "declined"
                else:
                    improvement = ((old_val - new_val) / old_val) * 100
                    direction = "improved" if improvement > 0 else "declined"
                
                return (improvement, direction)
            
            # Composite score improvement
            comp_improvement, comp_dir = calculate_improvement(
                first_place.composite_score, 
                benchmark_strategy.composite_score, 
                higher_is_better=True
            )
            
            # Cost improvement (lower is better)
            cost_improvement, cost_dir = calculate_improvement(
                first_place.avg_cost_per_query,
                benchmark_strategy.avg_cost_per_query,
                higher_is_better=False
            )
            
            # Latency improvement (lower is better)
            latency_improvement, latency_dir = calculate_improvement(
                first_place.avg_latency_s,
                benchmark_strategy.avg_latency_s,
                higher_is_better=False
            )
            
            # Quality improvement (higher is better)
            quality_improvement, quality_dir = calculate_improvement(
                first_place.avg_quality_score,
                benchmark_strategy.avg_quality_score,
                higher_is_better=True
            )
            
            # Display comparison cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                comp_delta = f"{comp_improvement:+.2f}%" if comp_improvement != float('inf') else "N/A"
                st.metric(
                    "Composite Score",
                    f"{first_place.composite_score:.2f}",
                    delta=comp_delta,
                    delta_color="normal" if comp_improvement > 0 else "inverse",
                    help=f"Benchmark: {benchmark_strategy.composite_score:.2f}"
                )
            
            with col2:
                cost_delta = f"{cost_improvement:+.2f}%" if cost_improvement != float('inf') else "N/A"
                st.metric(
                    "Average Cost",
                    f"${first_place.avg_cost_per_query:.6f}",
                    delta=cost_delta,
                    delta_color="normal" if cost_improvement > 0 else "inverse",
                    help=f"Benchmark: ${benchmark_strategy.avg_cost_per_query:.6f}"
                )
            
            with col3:
                latency_delta = f"{latency_improvement:+.2f}%" if latency_improvement != float('inf') else "N/A"
                st.metric(
                    "Average Latency",
                    f"{first_place.avg_latency_s:.3f}s",
                    delta=latency_delta,
                    delta_color="normal" if latency_improvement > 0 else "inverse",
                    help=f"Benchmark: {benchmark_strategy.avg_latency_s:.3f}s"
                )
            
            with col4:
                quality_delta = f"{quality_improvement:+.2f}%" if quality_improvement != float('inf') else "N/A"
                st.metric(
                    "Average Quality",
                    f"{first_place.avg_quality_score * 100:.2f}%",
                    delta=quality_delta,
                    delta_color="normal" if quality_improvement > 0 else "inverse",
                    help=f"Benchmark: {benchmark_strategy.avg_quality_score * 100:.2f}%"
                )
            
            # Detailed comparison table
            with st.expander(t("📊 Detailed Comparison Table", "📊 详细对比表"), expanded=False):
                comparison_data = {
                    "Metric": [
                        "Composite Score",
                        "Average Cost ($)",
                        "Average Latency (s)",
                        "P95 Latency (s)",
                        "Average Quality Score (%)",
                        "Faithfulness (%)",
                        "Answer Relevancy (%)",
                        "Context Precision (%)",
                        "Average Token Count",
                        "Total Cost ($)",
                    ],
                    f"🥇 {first_place.strategy_name}": [
                        f"{first_place.composite_score:.2f}",
                        f"{first_place.avg_cost_per_query:.6f}",
                        f"{first_place.avg_latency_s:.3f}",
                        f"{first_place.p95_latency_s:.3f}",
                        f"{first_place.avg_quality_score * 100:.2f}",
                        f"{first_place.avg_faithfulness * 100:.2f}" if first_place.avg_faithfulness is not None else "N/A",
                        f"{first_place.avg_answer_relevancy * 100:.2f}" if first_place.avg_answer_relevancy is not None else "N/A",
                        f"{first_place.avg_context_precision * 100:.2f}" if first_place.avg_context_precision is not None else "N/A",
                        f"{first_place.avg_tokens_per_query:.0f}",
                        f"{first_place.total_cost:.6f}",
                    ],
                    f"🏆 {benchmark_strategy.strategy_name}": [
                        f"{benchmark_strategy.composite_score:.2f}",
                        f"{benchmark_strategy.avg_cost_per_query:.6f}",
                        f"{benchmark_strategy.avg_latency_s:.3f}",
                        f"{benchmark_strategy.p95_latency_s:.3f}",
                        f"{benchmark_strategy.avg_quality_score * 100:.2f}",
                        f"{benchmark_strategy.avg_faithfulness * 100:.2f}" if benchmark_strategy.avg_faithfulness is not None else "N/A",
                        f"{benchmark_strategy.avg_answer_relevancy * 100:.2f}" if benchmark_strategy.avg_answer_relevancy is not None else "N/A",
                        f"{benchmark_strategy.avg_context_precision * 100:.2f}" if benchmark_strategy.avg_context_precision is not None else "N/A",
                        f"{benchmark_strategy.avg_tokens_per_query:.0f}",
                        f"{benchmark_strategy.total_cost:.6f}",
                    ],
                    "Improvement / Decline": [
                        f"{comp_improvement:+.2f}%" if comp_improvement != float('inf') else "N/A",
                        f"{cost_improvement:+.2f}%" if cost_improvement != float('inf') else "N/A",
                        f"{latency_improvement:+.2f}%" if latency_improvement != float('inf') else "N/A",
                        f"{calculate_improvement(first_place.p95_latency_s, benchmark_strategy.p95_latency_s, False)[0]:+.2f}%" if benchmark_strategy.p95_latency_s > 0 else "N/A",
                        f"{quality_improvement:+.2f}%" if quality_improvement != float('inf') else "N/A",
                        f"{calculate_improvement(first_place.avg_faithfulness or 0, benchmark_strategy.avg_faithfulness or 0, True)[0]:+.2f}%" if (first_place.avg_faithfulness and benchmark_strategy.avg_faithfulness) else "N/A",
                        f"{calculate_improvement(first_place.avg_answer_relevancy or 0, benchmark_strategy.avg_answer_relevancy or 0, True)[0]:+.2f}%" if (first_place.avg_answer_relevancy and benchmark_strategy.avg_answer_relevancy) else "N/A",
                        f"{calculate_improvement(first_place.avg_context_precision or 0, benchmark_strategy.avg_context_precision or 0, True)[0]:+.2f}%" if (first_place.avg_context_precision and benchmark_strategy.avg_context_precision) else "N/A",
                        f"{calculate_improvement(first_place.avg_tokens_per_query, benchmark_strategy.avg_tokens_per_query, False)[0]:+.2f}%" if benchmark_strategy.avg_tokens_per_query > 0 else "N/A",
                        f"{calculate_improvement(first_place.total_cost, benchmark_strategy.total_cost, False)[0]:+.2f}%" if benchmark_strategy.total_cost > 0 else "N/A",
                    ],
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, width='stretch', use_container_width=True)
                
                st.caption(t("💡 Positive percentages indicate improvement, while negative percentages indicate a decline.", "💡 正数表示改进，负数表示下降。"))
    
    # Visualization section
    st.markdown("#### 📊 Visual Analysis")
    
    # Create visualization tabs
    viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5 = st.tabs(["Composite Score", "Cost Analysis", "Latency Analysis", "Quality Analysis", "Runtime Analysis"])
    
    with viz_tab1:
        # Composite score comparison
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            import numpy as np
            matplotlib_available = True
            
            # Configure Chinese font support
            try:
                # Try to use system fonts that support Chinese
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
            except Exception:
                pass  # Fallback to default font if Chinese fonts not available
        except ImportError:
            matplotlib_available = False
            st.warning(t("`matplotlib` is not installed, so charts cannot be displayed. Run: `pip install matplotlib`", "未安装 `matplotlib`，无法显示图表。请运行：`pip install matplotlib`"))
        
        if matplotlib_available:
            # Sort by composite score (descending - highest first)
            sorted_data = sorted(zip(all_metrics, range(len(all_metrics))), 
                               key=lambda x: getattr(x[0], 'composite_score', 50.0), 
                               reverse=True)
            sorted_metrics, _ = zip(*sorted_data)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            strategy_names = [m.strategy_name.replace(" [Benchmark]", "") for m in sorted_metrics]
            composite_scores = [getattr(m, 'composite_score', 50.0) for m in sorted_metrics]
            
            # Color bars: orange for benchmark, blue for others
            # Only highlight single model benchmark strategies (no " + " separator)
            colors = ['#FF8C00' if '[Benchmark]' in m.strategy_name and ' + ' not in m.strategy_name else '#4A90E2' 
                     for m in sorted_metrics]
            
            # Highlight first place with gold color
            if len(colors) > 0:
                colors[0] = '#FFD700'  # Gold color for first place
            
            bars = ax.barh(strategy_names, composite_scores, color=colors)
            ax.set_xlabel('Composite Score', fontsize=12, fontweight='bold')
            ax.set_ylabel('Strategy', fontsize=12, fontweight='bold')
            ax.set_title('Composite Score Comparison (higher is better)', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlim(0, 100)
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels on bars and mark first place
            for i, (bar, score) in enumerate(zip(bars, composite_scores)):
                width = bar.get_width()
                y_pos = bar.get_y() + bar.get_height()/2
                
                # Add rank indicator for first place
                if i == 0:
                    ax.text(width + 1, y_pos, 
                           f'🥇 {score:.2f}', ha='left', va='center', 
                           fontweight='bold', fontsize=11, color='#FF6B00')
                else:
                    ax.text(width + 1, y_pos, 
                           f'{score:.2f}', ha='left', va='center', 
                           fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            # Fallback to simple text display
            st.info(t("Chart rendering requires `matplotlib`. Please install it with: `pip install matplotlib`", "图表渲染需要 `matplotlib`。请安装：`pip install matplotlib`"))
    
    with viz_tab2:
        # Cost analysis
        if matplotlib_available:
            # Ensure Chinese font is configured
            try:
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
            except Exception:
                pass
            # Sort by cost (ascending - lowest first)
            sorted_data = sorted(zip(all_metrics, range(len(all_metrics))), 
                               key=lambda x: x[0].avg_cost_per_query)
            sorted_metrics, _ = zip(*sorted_data)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            strategy_names = [m.strategy_name.replace(" [Benchmark]", "") for m in sorted_metrics]
            # Cost per query
            costs = [m.avg_cost_per_query for m in sorted_metrics]
            # Only highlight single model benchmark strategies (no " + " separator)
            colors = ['#FF8C00' if '[Benchmark]' in m.strategy_name and ' + ' not in m.strategy_name else '#4A90E2' 
                     for m in sorted_metrics]
            
            ax1.barh(strategy_names, costs, color=colors)
            ax1.set_xlabel('Average Cost per Query ($)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Strategy', fontsize=12, fontweight='bold')
            ax1.set_title('Per-Query Cost Comparison (lower is better)', fontsize=14, fontweight='bold', pad=20)
            ax1.grid(axis='x', alpha=0.3)
            
            # Total cost
            total_costs = [m.total_cost for m in sorted_metrics]
            ax2.barh(strategy_names, total_costs, color=colors)
            ax2.set_xlabel('Total Benchmark Cost ($)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Strategy', fontsize=12, fontweight='bold')
            ax2.set_title('Total Cost Comparison (lower is better)', fontsize=14, fontweight='bold', pad=20)
            ax2.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info(t("Chart rendering requires `matplotlib`.", "图表渲染需要 `matplotlib`。"))
    
    with viz_tab3:
        # Latency analysis
        if matplotlib_available:
            # Ensure Chinese font is configured
            try:
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
            except Exception:
                pass
            # Sort by average latency (ascending - lowest first)
            sorted_data = sorted(zip(all_metrics, range(len(all_metrics))), 
                               key=lambda x: x[0].avg_latency_s)
            sorted_metrics, _ = zip(*sorted_data)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            strategy_names = [m.strategy_name.replace(" [Benchmark]", "") for m in sorted_metrics]
            # Only highlight single model benchmark strategies (no " + " separator)
            colors = ['#FF8C00' if '[Benchmark]' in m.strategy_name and ' + ' not in m.strategy_name else '#4A90E2' 
                     for m in sorted_metrics]
            
            # Average latency
            avg_lats = [m.avg_latency_s for m in sorted_metrics]
            ax1.barh(strategy_names, avg_lats, color=colors)
            ax1.set_xlabel('Average Generation Time (s)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Strategy', fontsize=12, fontweight='bold')
            ax1.set_title('Average Latency Comparison (lower is better)', fontsize=14, fontweight='bold', pad=20)
            ax1.grid(axis='x', alpha=0.3)
            
            # P95 latency
            p95_lats = [m.p95_latency_s for m in sorted_metrics]
            ax2.barh(strategy_names, p95_lats, color=colors)
            ax2.set_xlabel('P95 Latency (s)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Strategy', fontsize=12, fontweight='bold')
            ax2.set_title('P95 Latency Comparison (lower is better)', fontsize=14, fontweight='bold', pad=20)
            ax2.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info(t("Chart rendering requires `matplotlib`.", "图表渲染需要 `matplotlib`。"))
    
    with viz_tab4:
        # Quality analysis
        if matplotlib_available:
            # Ensure Chinese font is configured
            try:
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
            except Exception:
                pass
            # Sort by quality score (descending - highest first)
            sorted_data = sorted(zip(all_metrics, range(len(all_metrics))), 
                               key=lambda x: x[0].avg_quality_score, 
                               reverse=True)
            sorted_metrics, _ = zip(*sorted_data)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            strategy_names = [m.strategy_name.replace(" [Benchmark]", "") for m in sorted_metrics]
            # Only highlight single model benchmark strategies (no " + " separator)
            colors = ['#FF8C00' if '[Benchmark]' in m.strategy_name and ' + ' not in m.strategy_name else '#4A90E2' 
                     for m in sorted_metrics]
            
            qualities = [m.avg_quality_score * 100 for m in sorted_metrics]  # Convert to percentage
            
            bars = ax.barh(strategy_names, qualities, color=colors)
            ax.set_xlabel('Average Quality Score (%)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Strategy', fontsize=12, fontweight='bold')
            ax.set_title('Average Quality Score Comparison (higher is better)\nNote: average quality score = (Faithfulness + Answer Relevancy + Context Precision) / 3', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlim(0, 100)
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for bar, qual in zip(bars, qualities):
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                       f'{qual:.2f}%', ha='left', va='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            
            # RAGAS detailed metrics comparison (grouped bar chart)
            st.markdown("#### 📊 Detailed RAGAS Metric Comparison")
            
            # Filter metrics that have RAGAS data
            metrics_with_ragas = [
                m for m in all_metrics 
                if m.avg_faithfulness is not None or m.avg_answer_relevancy is not None or m.avg_context_precision is not None
            ]
            
            if metrics_with_ragas:
                # Sort by average quality score
                sorted_ragas = sorted(metrics_with_ragas, key=lambda x: x.avg_quality_score, reverse=True)
                
                fig, ax = plt.subplots(figsize=(14, 8))
                
                strategy_names_ragas = [m.strategy_name.replace(" [Benchmark]", "") for m in sorted_ragas]
                x = np.arange(len(strategy_names_ragas))
                width = 0.25
                
                # Extract RAGAS metrics (convert to percentage, use 0 if None)
                faithfulness_vals = [(m.avg_faithfulness * 100) if m.avg_faithfulness is not None else 0 
                                    for m in sorted_ragas]
                answer_relevancy_vals = [(m.avg_answer_relevancy * 100) if m.avg_answer_relevancy is not None else 0 
                                        for m in sorted_ragas]
                context_precision_vals = [(m.avg_context_precision * 100) if m.avg_context_precision is not None else 0 
                                          for m in sorted_ragas]
                
                bars1 = ax.bar(x - width, faithfulness_vals, width, label='Faithfulness', color='#FF6B6B')
                bars2 = ax.bar(x, answer_relevancy_vals, width, label='Answer Relevancy', color='#4ECDC4')
                bars3 = ax.bar(x + width, context_precision_vals, width, label='Context Precision', color='#95E1D3')
                
                ax.set_xlabel('Strategy', fontsize=12, fontweight='bold')
                ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
                ax.set_title('Detailed RAGAS Metric Comparison (higher is better)', fontsize=14, fontweight='bold', pad=20)
                ax.set_xticks(x)
                ax.set_xticklabels(strategy_names_ragas, rotation=45, ha='right')
                ax.set_ylim(0, 100)
                ax.legend(loc='upper left')
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bars in [bars1, bars2, bars3]:
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info(t("No detailed RAGAS metric data is available yet.", "暂无详细 RAGAS 指标数据。"))
            
            # Route-accuracy plots intentionally removed from leaderboard analytics.
    
    with viz_tab5:
        # Runtime analysis
        if matplotlib_available:
            # Ensure Chinese font is configured
            try:
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
            except Exception:
                pass
            # Calculate total runtime per strategy (avg_latency * number of queries)
            # We'll use avg_latency_s as proxy, but ideally we'd have actual runtime
            # Sort by average latency (ascending - fastest first)
            sorted_data = sorted(zip(all_metrics, range(len(all_metrics))), 
                               key=lambda x: x[0].avg_latency_s)
            sorted_metrics, _ = zip(*sorted_data)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            strategy_names = [m.strategy_name.replace(" [Benchmark]", "") for m in sorted_metrics]
            # Only highlight single model benchmark strategies (no " + " separator)
            colors = ['#FF8C00' if '[Benchmark]' in m.strategy_name and ' + ' not in m.strategy_name else '#4A90E2' 
                     for m in sorted_metrics]
            
            # Average latency (generation time)
            avg_lats = [m.avg_latency_s for m in sorted_metrics]
            ax1.barh(strategy_names, avg_lats, color=colors)
            ax1.set_xlabel('Average Generation Time (s)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Strategy', fontsize=12, fontweight='bold')
            ax1.set_title('Average Generation Time Comparison (shorter is better)', fontsize=14, fontweight='bold', pad=20)
            ax1.grid(axis='x', alpha=0.3)
            
            # Evaluation time
            eval_times = [m.avg_eval_time_s for m in sorted_metrics]
            ax2.barh(strategy_names, eval_times, color=colors)
            ax2.set_xlabel('Average Evaluation Time (s)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Strategy', fontsize=12, fontweight='bold')
            ax2.set_title('Average Evaluation Time Comparison (shorter is better)', fontsize=14, fontweight='bold', pad=20)
            ax2.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info(t("Chart rendering requires `matplotlib`.", "图表渲染需要 `matplotlib`。"))


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


def _save_benchmark_history(
    strategy_results: Dict[str, List[Dict[str, Any]]],
    test_cases: List[Dict[str, Any]],
    all_strategies: List[Tuple[str, Optional[str], str]],
    final_metrics: Optional[List[StrategyMetrics]],
    fast_mode: bool,
    test_set_path: str,
    run_name: Optional[str] = None,
    run_note: Optional[str] = None,
) -> None:
    """Save benchmark results to history file.
    
    Args:
        strategy_results: Final results for each strategy.
        test_cases: List of test cases.
        all_strategies: List of all strategies tested.
        final_metrics: Final aggregated metrics for all strategies.
        fast_mode: Whether fast mode was enabled.
        test_set_path: Path to the test set file.
    """
    try:
        # Ensure directory exists
        BENCHMARK_PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Prepare history entry
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_name": run_name or time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_note": run_note or "",
            "test_set_path": test_set_path,
            "test_set_name": Path(test_set_path).name,
            "fast_mode": fast_mode,
            "total_strategies": len(all_strategies),
            "total_test_cases": len(test_cases),
            "strategy_results": strategy_results,
            "test_cases": test_cases,
            "all_strategies": [
                {
                    "strategy_name": strategy_name,
                    "small_model": small_model,
                    "large_model": large_model,
                }
                for strategy_name, small_model, large_model in all_strategies
            ],
            "final_metrics": [
                {
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
                    "routing_simple_accuracy": m.routing_simple_accuracy,
                    "routing_complex_accuracy": m.routing_complex_accuracy,
                    "raw_faithfulness_scores": m.raw_faithfulness_scores,
                    "raw_answer_relevancy_scores": m.raw_answer_relevancy_scores,
                    "raw_context_precision_scores": m.raw_context_precision_scores,
                    "raw_latency_scores": m.raw_latency_scores,
                    "raw_cost_scores": m.raw_cost_scores,
                    "composite_score": getattr(m, 'composite_score', 50.0),
                }
                for m in (final_metrics or [])
            ],
        }
        
        # Append to JSONL file
        with open(BENCHMARK_HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        logger.info(f"Benchmark history saved to {BENCHMARK_HISTORY_FILE}")
    except Exception as e:
        logger.error(f"Failed to save benchmark history: {e}", exc_info=True)


def _save_completed_progress_to_history(progress_data: Dict[str, Any]) -> None:
    """Save a completed progress snapshot into benchmark history."""
    completed_strategies = progress_data.get("completed_strategies", [])
    strategy_results: Dict[str, List[Dict[str, Any]]] = {}
    for item in completed_strategies:
        strategy_name = item.get("strategy_name")
        if strategy_name:
            strategy_results[strategy_name] = item.get("results", [])

    test_cases = progress_data.get("test_cases", [])
    all_strategies_raw = progress_data.get("all_strategies", [])
    all_strategies: List[Tuple[str, Optional[str], str]] = []
    for s in all_strategies_raw:
        name = s.get("strategy_name")
        large = s.get("large_model")
        if not name or not large:
            continue
        all_strategies.append((name, s.get("small_model"), large))

    final_metrics = _compute_aggregated_metrics(strategy_results, test_cases)
    scoring_engine = _create_arena_scoring_engine()
    for metrics in final_metrics:
        metrics.composite_score = scoring_engine.compute_composite_score(metrics, final_metrics)  # type: ignore[attr-defined]

    _save_benchmark_history(
        strategy_results=strategy_results,
        test_cases=test_cases,
        all_strategies=all_strategies,
        final_metrics=final_metrics,
        fast_mode=bool(progress_data.get("fast_mode", False)),
        test_set_path=str(progress_data.get("test_set_path", "")),
    )


def _load_benchmark_history() -> List[Dict[str, Any]]:
    """Load benchmark history from JSONL file.
    
    Returns:
        List of history entries, most recent first.
    """
    if not BENCHMARK_HISTORY_FILE.exists():
        return []
    
    entries: List[Dict[str, Any]] = []
    try:
        with open(BENCHMARK_HISTORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse history line: {line[:100]}")
                        continue
    except Exception as e:
        logger.error(f"Failed to load benchmark history: {e}", exc_info=True)
    
    # Return most recent first
    return list(reversed(entries))


def _render_benchmark_history() -> None:
    """Render benchmark history view."""
    st.markdown("#### 📈 Benchmark History")
    st.markdown("Review previously completed benchmark runs, including leaderboards and visual analysis.")
    
    history = _load_benchmark_history()
    if not history:
        st.info(t("📭 No benchmark history is available yet. Completed runs will appear here automatically.", "📭 暂无压测历史。完成的运行会自动显示在这里。"))
        return
    
    # Show history list
    st.markdown(f"**{len(history)} history entries found** (newest first)")
    
    # Create selection dropdown
    history_options = [
        f"{entry.get('run_name', entry.get('timestamp', 'Unknown'))} - {entry.get('test_set_name', 'Unknown')} "
        f"({entry.get('total_strategies', 0)} strategies, {entry.get('total_test_cases', 0)} test cases)"
        for entry in history
    ]
    
    selected_idx = st.selectbox(
        "Select a history entry",
        options=range(len(history)),
        format_func=lambda x: history_options[x],
        key="benchmark_history_select",
    )
    
    if selected_idx is not None:
        selected_entry = history[selected_idx]
        
        # Display entry info
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(t("Run Name", "运行名称"), selected_entry.get("run_name", selected_entry.get("timestamp", "Unknown")))
        with col2:
            st.metric(t("Strategies", "策略数"), selected_entry.get("total_strategies", 0))
        with col3:
            st.metric(t("Test Cases", "测试用例数"), selected_entry.get("total_test_cases", 0))
        
        st.caption(f"Test set: `{selected_entry.get('test_set_name', 'Unknown')}` | Time: {selected_entry.get('timestamp', 'Unknown')}")
        if selected_entry.get("fast_mode"):
            st.caption(t("⚡ Fast mode was enabled", "⚡ 已启用加速模式"))
        
        # Show note if available
        run_note = selected_entry.get("run_note", "")
        if run_note:
            with st.expander(t("📝 Run Notes", "📝 运行备注"), expanded=False):
                st.markdown(run_note)
        
        # Reconstruct StrategyMetrics from saved data
        from src.observability.dashboard.services.scoring_engine import StrategyMetrics
        
        final_metrics_data = selected_entry.get("final_metrics", [])
        reconstructed_metrics = []
        for m_data in final_metrics_data:
            metrics = StrategyMetrics(
                strategy_name=m_data.get("strategy_name", "Unknown"),
                success_rate=m_data.get("success_rate", 0.0),
                avg_latency_s=m_data.get("avg_latency_s", 0.0),
                p95_latency_s=m_data.get("p95_latency_s", 0.0),
                avg_eval_time_s=m_data.get("avg_eval_time_s", 0.0),
                avg_tokens_per_query=m_data.get("avg_tokens_per_query", 0.0),
                avg_cost_per_query=m_data.get("avg_cost_per_query", 0.0),
                total_cost=m_data.get("total_cost", 0.0),
                avg_quality_score=m_data.get("avg_quality_score", 0.0),
                avg_faithfulness=m_data.get("avg_faithfulness"),
                avg_answer_relevancy=m_data.get("avg_answer_relevancy"),
                avg_context_precision=m_data.get("avg_context_precision"),
                routing_total_accuracy=m_data.get("routing_total_accuracy"),
                routing_simple_accuracy=m_data.get("routing_simple_accuracy"),
                routing_complex_accuracy=m_data.get("routing_complex_accuracy"),
                raw_faithfulness_scores=m_data.get("raw_faithfulness_scores", []) or [],
                raw_answer_relevancy_scores=m_data.get("raw_answer_relevancy_scores", []) or [],
                raw_context_precision_scores=m_data.get("raw_context_precision_scores", []) or [],
                raw_latency_scores=m_data.get("raw_latency_scores", []) or [],
                raw_cost_scores=m_data.get("raw_cost_scores", []) or [],
            )
            # Set composite_score as attribute
            metrics.composite_score = m_data.get("composite_score", 50.0)
            reconstructed_metrics.append(metrics)
        
        # Display leaderboard and visualizations
        if reconstructed_metrics:
            scoring_engine = _create_arena_scoring_engine()
            _display_leaderboard(reconstructed_metrics, scoring_engine)
        else:
            st.warning(t("⚠️ This history entry does not contain displayable evaluation results.", "⚠️ 该历史记录不包含可显示的评估结果。"))



