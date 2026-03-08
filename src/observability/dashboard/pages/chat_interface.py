"""Chat Interface page – Client-side RAG chat interface.

This page implements a complete RAG workflow at the CLIENT level:
1. Retrieve: Calls HybridSearch + Reranker (same as MCP tool)
2. Generate: Uses RAGGenerator to call local LLM (from settings.yaml)
3. Format: Combines LLM answer with citations

Architecture:
- MCP Server: Only does retrieval (query_knowledge_hub tool)
- UI Client (this): Does retrieval + generation (complete RAG loop)
- Both use the same retrieval components, but generation happens here.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import json
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)


# ── Chat history persistence ──────────────────────────────────────────────

_CHAT_STORAGE_PATH = Path("data/chat/conversations.json")


def _ensure_storage_dir() -> None:
    """Ensure chat history directory exists."""
    try:
        _CHAT_STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.warning("Failed to create chat storage directory: %s", exc)


def _load_conversations_from_disk() -> List[Dict[str, Any]]:
    """Load conversations from disk, returning an empty list on failure."""
    try:
        if _CHAT_STORAGE_PATH.exists():
            with _CHAT_STORAGE_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            logger.warning("Chat history file has unexpected format, resetting.")
    except Exception as exc:
        logger.warning("Failed to load chat history: %s", exc)
    return []


def _save_conversations_to_disk() -> None:
    """Persist current conversations to disk (best-effort)."""
    try:
        _ensure_storage_dir()
        with _CHAT_STORAGE_PATH.open("w", encoding="utf-8") as f:
            json.dump(st.session_state.conversations, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Failed to save chat history: %s", exc)


# ── Custom CSS for Gemini-style UI ──────────────────────────────────────

GEMINI_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --sidebar-bg: #F0F2F5;
        --main-bg: #FFFFFF;
        --text-primary: #1F1F1F;
        --text-secondary: #5F6368;
        --border-color: #E8EAED;
        --hover-bg: #F8F9FA;
        --selected-bg: #E8F0FE;
        --selected-text: #1A73E8;
        --input-bg: #FFFFFF;
        --input-border: #DADCE0;
        --shadow-sm: 0 1px 2px 0 rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
        --shadow-md: 0 2px 6px 2px rgba(60, 64, 67, 0.15), 0 1px 2px 0 rgba(60, 64, 67, 0.3);
    }
    
    .main .block-container {
        padding: 0;
        max-width: 100%;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    section[data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
    }
    
    section[data-testid="stSidebar"] > div {
        background-color: var(--sidebar-bg);
    }
    
    .stChatMessage {
        padding: 16px 0;
    }
    
    .stChatInput > div > div > input {
        border-radius: 24px;
        border: 1px solid var(--input-border);
        box-shadow: var(--shadow-md);
        padding: 16px 20px;
        font-family: 'Inter', sans-serif;
        font-size: 15px;
    }
    
    .stChatInput > div > div > input:focus {
        border-color: var(--selected-text);
        box-shadow: 0 2px 8px 2px rgba(26, 115, 232, 0.15);
    }

    /* Citation expander title smaller */
    details[citation-summary="true"] > summary {
        font-size: 13px;
        color: var(--text-secondary);
    }
</style>
"""


# ── Helper Functions ─────────────────────────────────────────────────────

def _initialize_session_state() -> None:
    """Initialize session state for chat interface."""
    if "conversations" not in st.session_state:
        # Load persisted conversations if available.
        st.session_state.conversations = _load_conversations_from_disk()

    if "current_conversation_id" not in st.session_state:
        # Default to the first conversation if any exist.
        if st.session_state.conversations:
            st.session_state.current_conversation_id = st.session_state.conversations[0]["id"]
        else:
            st.session_state.current_conversation_id = None

    if "messages" not in st.session_state:
        # Initialize messages from the current conversation, if any.
        current = _get_current_conversation()
        st.session_state.messages = current.get("messages", []).copy() if current else []
    
    # Initialize model manager and register models
    if "chat_model_manager" not in st.session_state:
        _initialize_chat_models()
    
    # Initialize selected model (default to benchmark model)
    if "selected_chat_model" not in st.session_state:
        from src.core.settings import load_settings
        settings = load_settings()
        benchmark_id = _get_benchmark_model_id(settings)
        st.session_state.selected_chat_model = benchmark_id


def _get_current_conversation() -> Optional[Dict[str, Any]]:
    """Get the current conversation dict."""
    if st.session_state.current_conversation_id is None:
        return None
    for conv in st.session_state.conversations:
        if conv["id"] == st.session_state.current_conversation_id:
            return conv
    return None


def _get_benchmark_model_id(settings: Any) -> str:
    """Get the benchmark model ID from settings."""
    provider = settings.llm.provider
    model_name = settings.llm.model
    return f"{provider}-{model_name}".replace("/", "-").replace(":", "-")


def _initialize_chat_models() -> None:
    """Initialize and register all models for chat interface."""
    try:
        from src.core.settings import load_settings
        from src.libs.llm.model_manager import ModelConfig, ModelManager
        from src.libs.llm.llm_factory import LLMFactory
        
        settings = load_settings()
        manager = ModelManager(settings)
        
        # Store benchmark model ID
        benchmark_id = _get_benchmark_model_id(settings)
        st.session_state.benchmark_model_id = benchmark_id
        
        # Available providers
        available_providers = set(LLMFactory.list_providers())
        
        # Register benchmark model (from settings.yaml)
        provider_display = settings.llm.provider.title()
        benchmark_config = ModelConfig(
            model_id=benchmark_id,
            provider=settings.llm.provider,
            model_name=settings.llm.model,
            display_name=f"{provider_display} {settings.llm.model} [Benchmark]",
            description="Benchmark model from settings.yaml",
            is_small_model=False,
        )
        manager.register_model(benchmark_config)
        
        # Register Ollama models
        if "ollama" in available_providers:
            ollama_models = [
                ("ollama-qwen2.5:0.5b", "qwen2.5:0.5b", "Ollama Qwen2.5 0.5B (极速)"),
                ("ollama-qwen2.5:1.5b", "qwen2.5:1.5b", "Ollama Qwen2.5 1.5B (推荐)"),
            ]
            
            for display_name, model_name, description in ollama_models:
                model_id = display_name.replace(":", "-")
                config = ModelConfig(
                    model_id=model_id,
                    provider="ollama",
                    model_name=model_name,
                    display_name=display_name,
                    description=description,
                    is_small_model=True,
                )
                manager.register_model(config)
        
        # Register API models (via OpenAI-compatible format)
        if "openai" in available_providers:
            api_key = getattr(settings.llm, "api_key", None)
            
            # Model name mapping and base URL mapping
            # Note: api-qwen-max base_url changed from /alibaba to /v1 due to API 500 error
            api_models = [
                ("api-deepseek-chat", "deepseek-chat", "https://api.zhizengzeng.com/v1", "DeepSeek Chat"),
                ("api-gpt-4o-mini", "gpt-4o-mini", "https://api.zhizengzeng.com/v1", "OpenAI GPT-4o Mini"),
                ("api-qwen-max", "qwen-max", "https://api.zhizengzeng.com/v1", "Qwen Max"),  # Changed from /alibaba to /v1
                ("api-glm-4-plus", "glm-4-plus", "https://api.zhizengzeng.com/v1", "GLM-4 Plus"),
            ]
            
            for display_name, model_name, base_url, description in api_models:
                model_id = display_name.replace(":", "-")
                config = ModelConfig(
                    model_id=model_id,
                    provider="openai",
                    model_name=model_name,
                    display_name=display_name,
                    description=description,
                    is_small_model=False,
                    config_override={
                        "base_url": base_url,
                        "api_key": api_key,
                    },
                )
                manager.register_model(config)
        
        st.session_state.chat_model_manager = manager
        logger.info("Chat models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize chat models: {e}")
        st.error(f"模型初始化失败: {e}")


def _create_new_conversation() -> str:
    """Create a new conversation and return its ID."""
    import uuid
    conv_id = str(uuid.uuid4())
    conv = {
        "id": conv_id,
        "title": "新对话",
        "messages": [],
        "created_at": "Now",
    }
    st.session_state.conversations.insert(0, conv)
    st.session_state.current_conversation_id = conv_id
    st.session_state.messages = []
    _save_conversations_to_disk()
    return conv_id


def _load_conversation(conv_id: str) -> None:
    """Load a conversation into the current session."""
    for conv in st.session_state.conversations:
        if conv["id"] == conv_id:
            st.session_state.current_conversation_id = conv_id
            st.session_state.messages = conv.get("messages", []).copy()
            break


def _update_conversation_title(conv_id: str, first_message: str) -> None:
    """Update conversation title based on first message."""
    changed = False
    for conv in st.session_state.conversations:
        if conv["id"] == conv_id:
            title = first_message[:30] + ("..." if len(first_message) > 30 else "")
            conv["title"] = title
            changed = True
            break
    if changed:
        _save_conversations_to_disk()


def _query_knowledge_base(query: str, collection: Optional[str] = None, top_k: int = 10, model_id: Optional[str] = None) -> Dict[str, Any]:
    """Query the knowledge base using complete RAG pipeline (Client-side).
    
    This implements the COMPLETE RAG workflow at the CLIENT level:
    
    Step 1 (Retrieve): 
        - HybridSearch.search() - Dense + Sparse + Fusion
        - Reranker.rerank() - Optional reranking
    
    Step 2 (Generate - UI Client only):
        - RAGGenerator.create(settings) - Uses LLM from settings.yaml (e.g., Ollama)
        - RAGGenerator.generate() - Generates answer from retrieved chunks
    
    Step 3 (Format):
        - CitationGenerator - Generates citation references
        - Combine LLM answer with citations
    
    Architecture Note:
    - MCP Server (query_knowledge_hub): Only does Step 1 (retrieval)
    - UI Client (this function): Does Step 1 + Step 2 + Step 3 (complete RAG)
    - Both use the same retrieval components, but generation happens at client side.
    
    Args:
        query: User query string.
        collection: Optional collection name to query.
        top_k: Maximum number of results to return.
        
    Returns:
        Dictionary with:
            - content: LLM-generated answer with citation references
            - citations: List of citation objects
            - is_empty: Whether results were found
    """
    try:
        from src.core.settings import load_settings, resolve_path
        from src.core.query_engine.hybrid_search import create_hybrid_search
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.dense_retriever import create_dense_retriever
        from src.core.query_engine.sparse_retriever import create_sparse_retriever
        from src.core.query_engine.reranker import create_core_reranker
        from src.core.response.citation_generator import CitationGenerator
        from src.core.response.rag_generator import RAGGenerator
        from src.core.trace.trace_context import TraceContext
        from src.core.trace.trace_collector import TraceCollector
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        
        # Load settings
        settings = load_settings()
        effective_collection = collection or "default"
        
        # ===================================================================
        # Step 1: RETRIEVE (Same as MCP tool - pure retrieval)
        # ===================================================================
        
        # Initialize retrieval components
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
        
        # Create trace
        trace = TraceContext(trace_type="query")
        trace.metadata["query"] = query[:200]
        trace.metadata["collection"] = effective_collection
        trace.metadata["top_k"] = top_k
        trace.metadata["source"] = "dashboard_chat"
        
        # Perform hybrid search
        results = hybrid_search.search(
            query=query,
            top_k=top_k,
            filters={"collection": effective_collection} if collection else None,
            trace=trace,
        )
        
        # Apply reranking if enabled
        if reranker.is_enabled and results:
            try:
                rerank_result = reranker.rerank(
                    query=query,
                    results=results,
                    top_k=top_k,
                    trace=trace,
                )
                results = rerank_result.results
            except Exception as e:
                logger.warning(f"Reranking failed: {e}. Using original order.")
        
        # ===================================================================
        # Step 2: GENERATE (Client-side only - UI layer)
        # ===================================================================
        
        # Get selected model or use default
        effective_model_id = model_id or st.session_state.get("selected_chat_model")
        
        # Get LLM instance from ModelManager if available
        llm_instance = None
        if effective_model_id and "chat_model_manager" in st.session_state:
            try:
                manager = st.session_state.chat_model_manager
                llm_instance = manager.get_llm(effective_model_id)
                logger.info(f"Using selected model: {effective_model_id}")
            except Exception as e:
                logger.warning(f"Failed to get LLM from ModelManager: {e}, falling back to default")
        
        # Generate LLM answer using RAGGenerator
        # Use selected model if available, otherwise use default from settings
        rag_generator = RAGGenerator.create(settings=settings, llm=llm_instance)
        llm_answer = rag_generator.generate(
            query=query,
            results=results,
            trace=trace,
        )
        
        # ===================================================================
        # Step 3: FORMAT (Client-side)
        # ===================================================================
        
        # Generate citations
        citation_generator = CitationGenerator()
        citations = citation_generator.generate(results)
        
        # Collect trace
        TraceCollector().collect(trace)
        
        return {
            "content": llm_answer,
            "citations": citations,
            "is_empty": len(results) == 0,
        }
        
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        return {
            "content": f"## 查询错误\n\n查询时发生错误：{str(e)}\n\n请检查配置和日志。",
            "citations": [],
            "is_empty": True,
        }


# ── Main Render Function ────────────────────────────────────────────────

def render() -> None:
    """Render the chat interface page."""
    # Inject custom CSS
    st.markdown(GEMINI_CSS, unsafe_allow_html=True)
    
    # Initialize session state
    _initialize_session_state()
    
    # Sidebar: New conversation button
    if st.sidebar.button("➕ 发起新对话", use_container_width=True):
        _create_new_conversation()
        st.rerun()
    
    # Sidebar: Conversation history
    st.sidebar.title("对话历史")
    
    for conv in st.session_state.conversations:
        is_selected = conv["id"] == st.session_state.current_conversation_id
        if st.sidebar.button(
            conv["title"],
            key=f"sidebar_conv_{conv['id']}",
            use_container_width=True,
            type="primary" if is_selected else "secondary",
        ):
            _load_conversation(conv["id"])
            st.rerun()
    
    # Sidebar: Model selection
    st.sidebar.markdown("---")
    st.sidebar.markdown("**模型选择**")
    
    if "chat_model_manager" in st.session_state:
        manager = st.session_state.chat_model_manager
        all_models = manager.list_models()
        benchmark_id = st.session_state.get("benchmark_model_id")
        
        # Format model options with benchmark marker
        model_options = []
        model_id_map = {}
        for m in all_models:
            display_name = m.display_name
            if m.model_id == benchmark_id:
                # Ensure benchmark marker is present
                if "[Benchmark]" not in display_name:
                    display_name = f"{display_name} [Benchmark]"
            model_options.append(display_name)
            model_id_map[display_name] = m.model_id
        
        # Get current selected model
        current_model_id = st.session_state.get("selected_chat_model")
        current_display = None
        for m in all_models:
            if m.model_id == current_model_id:
                current_display = m.display_name
                if m.model_id == benchmark_id and "[Benchmark]" not in current_display:
                    current_display = f"{current_display} [Benchmark]"
                break
        
        if current_display is None or current_display not in model_options:
            current_display = model_options[0] if model_options else None
        
        # Model selector
        try:
            default_index = model_options.index(current_display) if current_display and model_options else 0
        except ValueError:
            default_index = 0
        
        selected_display = st.sidebar.selectbox(
            "选择生成模型",
            options=model_options,
            index=default_index,
            key="chat_model_selector",
            help="选择用于RAG最终生成环节的大模型",
        )
        
        # Update selected model ID
        if selected_display in model_id_map:
            selected_model_id = model_id_map[selected_display]
            if st.session_state.get("selected_chat_model") != selected_model_id:
                st.session_state.selected_chat_model = selected_model_id
                logger.info(f"Model changed to: {selected_model_id}")
    else:
        st.sidebar.warning("模型管理器未初始化")
    
    # Sidebar: User info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**用户**")
    st.sidebar.caption("设置")
    
    # Main chat area: Title
    current_conv = _get_current_conversation()
    if current_conv:
        st.title(current_conv["title"])
    else:
        st.title("新对话")
    
    # Main chat area: Display messages
    for idx, msg in enumerate(st.session_state.messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        citations = msg.get("citations", [])
        
        if role == "user":
            with st.chat_message("user"):
                st.write(content)
        else:
            with st.chat_message("assistant"):
                st.markdown(content)
                if citations:
                    with st.expander(
                        f"📚 引用来源（{len(citations)}）",
                        expanded=False,
                    ):
                        for citation in citations:
                            if isinstance(citation, dict):
                                c_idx = citation.get("index", 0)
                                source = citation.get("source", "unknown")
                                page = citation.get("page")
                                snippet = citation.get("text_snippet", "")
                            else:
                                c_idx = getattr(citation, "index", 0)
                                source = getattr(citation, "source", "unknown")
                                page = getattr(citation, "page", None)
                                snippet = getattr(citation, "text_snippet", "")

                            source_label = f"[{c_idx}] {source}"
                            if page is not None:
                                source_label += f" (第 {page} 页)"

                            if isinstance(source, str) and (
                                source.startswith("http://") or source.startswith("https://")
                            ):
                                st.markdown(f"[{source_label}]({source})")
                            else:
                                st.markdown(source_label)

                            if snippet:
                                st.caption(f"*{snippet}*")
                            st.markdown("---")
    
    # Main chat area: Input
    if prompt := st.chat_input("输入消息..."):
        # Record start time
        import time
        start_time = time.monotonic()
        
        # Add user message
        user_msg = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_msg)
        
        # Create conversation if needed
        if st.session_state.current_conversation_id is None:
            _create_new_conversation()
            _update_conversation_title(st.session_state.current_conversation_id, prompt)
        
        # Update conversation messages
        current_conv = _get_current_conversation()
        if current_conv:
            current_conv["messages"] = st.session_state.messages.copy()
            _save_conversations_to_disk()
        
        # Query knowledge base (complete RAG: Retrieve + Generate + Format)
        with st.chat_message("assistant"):
            # Get selected model ID
            selected_model_id = st.session_state.get("selected_chat_model")
            with st.spinner("🔍 正在检索知识库并生成回答..."):
                result = _query_knowledge_base(prompt, top_k=10, model_id=selected_model_id)
            
            # Calculate elapsed time
            elapsed_time = time.monotonic() - start_time
            
            # Display elapsed time (small, subtle)
            st.caption(f"⏱️ 运行时间: {elapsed_time:.2f}秒")
            
            # Display LLM-generated answer
            st.markdown(result["content"])
            if result["citations"]:
                with st.expander(
                    f"📚 引用来源（{len(result['citations'])}）",
                    expanded=False,
                ):
                        for citation in result["citations"]:
                            if isinstance(citation, dict):
                                c_idx = citation.get("index", 0)
                                source = citation.get("source", "unknown")
                                page = citation.get("page")
                                snippet = citation.get("text_snippet", "")
                            else:
                                c_idx = getattr(citation, "index", 0)
                                source = getattr(citation, "source", "unknown")
                                page = getattr(citation, "page", None)
                                snippet = getattr(citation, "text_snippet", "")

                            source_label = f"[{c_idx}] {source}"
                            if page is not None:
                                source_label += f" (第 {page} 页)"

                            if isinstance(source, str) and (
                                source.startswith("http://") or source.startswith("https://")
                            ):
                                st.markdown(f"[{source_label}]({source})")
                            else:
                                st.markdown(source_label)

                            if snippet:
                                st.caption(f"*{snippet}*")
                            st.markdown("---")
        
        # Add assistant message with citations
        assistant_msg = {
            "role": "assistant",
            "content": result["content"],
            "citations": [c.to_dict() if hasattr(c, "to_dict") else c for c in result["citations"]],
        }
        st.session_state.messages.append(assistant_msg)
        
        # Update conversation
        if current_conv:
            current_conv["messages"] = st.session_state.messages.copy()
            _save_conversations_to_disk()
        
        st.rerun()

