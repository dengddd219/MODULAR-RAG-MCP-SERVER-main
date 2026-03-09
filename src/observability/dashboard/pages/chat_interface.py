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
import re
from typing import Any, Dict, List, Optional

import json
from pathlib import Path

import streamlit as st

from src.observability.dashboard.i18n import localized_mode, mode_options, t

logger = logging.getLogger(__name__)

CHAT_MODE_OPTIONS = mode_options()


# ── Chat history persistence ──────────────────────────────────────────────

_CHAT_STORAGE_PATH = Path("data/chat/conversations.json")


def _resolve_ui_language(query: str) -> str:
    """Resolve UI language from explicit directive or query language.

    Returns:
        "English" or "Chinese".
    """
    q = (query or "").strip()
    if not q:
        return "Chinese"

    # Chinese directives, e.g. "请用英文回答"
    zh_directive = re.search(
        r"(?:请|请你)?(?:用|使用|以)\s*([A-Za-z\u4e00-\u9fff\-\s]{1,24})\s*(?:回答|回复|输出|作答|说明)",
        q,
        flags=re.IGNORECASE,
    )
    if zh_directive:
        token = zh_directive.group(1).strip().lower()
        if token in {"英文", "英语", "english", "en"}:
            return "English"
        if token in {"中文", "汉语", "chinese", "zh", "简体中文", "繁体中文"}:
            return "Chinese"

    # English directives, e.g. "answer in English"
    en_directive = re.search(
        r"(?:answer|respond|reply|write|output)\s+(?:in|using)\s+([A-Za-z\-\s]{2,24})",
        q,
        flags=re.IGNORECASE,
    )
    if en_directive:
        token = en_directive.group(1).strip().lower()
        if token in {"english", "en"}:
            return "English"
        if token in {"chinese", "zh"}:
            return "Chinese"

    lower_q = q.lower()
    if any(p in lower_q for p in ("in english", "use english", "answer in english")):
        return "English"
    if any(p in lower_q for p in ("in chinese", "use chinese", "answer in chinese")):
        return "Chinese"

    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", q))
    latin_count = len(re.findall(r"[A-Za-z]", q))
    return "Chinese" if cjk_count >= latin_count else "English"


def _citation_label(count: int, ui_language: str) -> str:
    """Build citations expander label in target language."""
    return t(f"📚 Sources ({count})", f"📚 引用来源（{count}）")


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
                for conv in data:
                    if isinstance(conv, dict) and conv.get("title") in {"新对话", "New Chat"}:
                        conv["title"] = t("New Chat", "新对话")
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
    if "selected_chat_mode" not in st.session_state:
        st.session_state.selected_chat_mode = "Fast"


def _get_current_conversation() -> Optional[Dict[str, Any]]:
    """Get the current conversation dict."""
    if st.session_state.current_conversation_id is None:
        return None
    for conv in st.session_state.conversations:
        if conv["id"] == st.session_state.current_conversation_id:
            return conv
    return None


def _display_conversation_title(title: str) -> str:
    """Localize the default conversation title while leaving custom titles unchanged."""
    if title in {"New Chat", "新对话"}:
        return t("New Chat", "新对话")
    return title


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
                ("ollama-qwen2.5:0.5b", "qwen2.5:0.5b", "Ollama Qwen2.5 0.5B (Ultra Fast)"),
                ("ollama-qwen2.5:1.5b", "qwen2.5:1.5b", "Ollama Qwen2.5 1.5B (Recommended)"),
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
        st.error(t("Model initialization failed: ", "模型初始化失败：") + str(e))


def _pick_model_id_for_mode(query: str) -> Optional[str]:
    """Resolve concrete model ID from UI mode and query complexity."""
    manager = st.session_state.get("chat_model_manager")
    if manager is None:
        return None

    all_models = manager.list_models()
    if not all_models:
        return None

    selected_mode = st.session_state.get("selected_chat_mode", "Fast")
    benchmark_id = st.session_state.get("benchmark_model_id")

    def _match_id(keywords: List[str]) -> Optional[str]:
        for m in all_models:
            haystack = f"{m.model_id} {m.display_name} {m.model_name}".lower()
            if all(k.lower() in haystack for k in keywords):
                return m.model_id
        return None

    gpt4o_mini_id = _match_id(["gpt", "4o", "mini"]) or benchmark_id or all_models[0].model_id
    qwen_05b_id = _match_id(["qwen2.5", "0.5b"]) or _match_id(["qwen", "0.5b"])

    if selected_mode in ("Fast", "Pro"):
        return gpt4o_mini_id

    # Think mode: simple query -> qwen 0.5b, complex query -> gpt-4o-mini
    query_text = (query or "").strip().lower()
    complex_signals = [
        "why", "compare", "tradeoff", "architecture", "design", "deep",
        "原理", "对比", "架构", "设计", "原因", "深入", "分析",
    ]
    looks_complex = len(query_text) >= 40 or any(sig in query_text for sig in complex_signals)
    if (not looks_complex) and qwen_05b_id:
        return qwen_05b_id
    return gpt4o_mini_id


def _create_new_conversation() -> str:
    """Create a new conversation and return its ID."""
    import uuid
    conv_id = str(uuid.uuid4())
    conv = {
        "id": conv_id,
        "title": t("New Chat", "新对话"),
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
            "content": (
                f"## {t('Query Error', '查询错误')}\n\n"
                + t("An error occurred while processing the query: ", "查询时发生错误：")
                + f"{str(e)}\n\n"
                + t("Please check the configuration and logs.", "请检查配置和日志。")
            ),
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
    if st.sidebar.button(t("➕ New Chat", "➕ 新对话"), use_container_width=True):
        _create_new_conversation()
        st.rerun()
    
    # Sidebar: Conversation history
    st.sidebar.title(t("Conversation History", "对话历史"))
    
    for conv in st.session_state.conversations:
        is_selected = conv["id"] == st.session_state.current_conversation_id
        if st.sidebar.button(
            _display_conversation_title(conv["title"]),
            key=f"sidebar_conv_{conv['id']}",
            use_container_width=True,
            type="primary" if is_selected else "secondary",
        ):
            _load_conversation(conv["id"])
            st.rerun()
    
    # Sidebar: Mode selection
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**{t('Response Mode', '回答模式')}**")
    default_mode = st.session_state.get("selected_chat_mode", "Fast")
    mode_index = CHAT_MODE_OPTIONS.index(default_mode) if default_mode in CHAT_MODE_OPTIONS else 0
    selected_mode = st.sidebar.selectbox(
        t("Choose mode", "选择模式"),
        options=CHAT_MODE_OPTIONS,
        index=mode_index,
        key="chat_mode_selector",
        format_func=localized_mode,
        help=t("Expose only the interaction mode, not the underlying model.", "只展示交互模式，不展示底层模型。"),
    )
    st.session_state.selected_chat_mode = selected_mode
    st.sidebar.caption(
        t(
            "Fast: speed/cost first. Think: better for complex questions. Pro: balanced overall experience.",
            "快速：优先速度和成本。思考：更适合复杂问题。Pro：综合体验更均衡。",
        )
    )
    
    # Sidebar: User info
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**{t('User', '用户')}**")
    st.sidebar.caption(t("Settings", "设置"))
    
    # Main chat area: Title
    current_conv = _get_current_conversation()
    if current_conv:
        st.title(_display_conversation_title(current_conv["title"]))
    else:
        st.title(t("New Chat", "新对话"))
    
    # Main chat area: Display messages
    for idx, msg in enumerate(st.session_state.messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        citations = msg.get("citations", [])
        ui_language = msg.get("ui_language", "Chinese")
        
        if role == "user":
            with st.chat_message("user"):
                st.write(content)
        else:
            with st.chat_message("assistant"):
                st.markdown(content)
                if citations:
                    with st.expander(
                        _citation_label(len(citations), ui_language),
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
                                source_label += (
                                    f" (page {page})"
                                    if ui_language == "English"
                                    else f" (page {page})"
                                )

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
    if prompt := st.chat_input(t("Type your message...", "输入消息...")):
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
            # Resolve model ID from selected mode
            selected_model_id = _pick_model_id_for_mode(prompt)
            selected_mode = st.session_state.get("selected_chat_mode", "Fast")
            ui_language = _resolve_ui_language(prompt)
            spinner_text = "🔍 Retrieving knowledge base and generating answer..."
            with st.spinner(spinner_text):
                result = _query_knowledge_base(prompt, top_k=10, model_id=selected_model_id)
            
            # Calculate elapsed time
            elapsed_time = time.monotonic() - start_time
            
            # Display elapsed time (small, subtle)
            st.caption(t(f"⏱️ Runtime: {elapsed_time:.2f}s", f"⏱️ 运行时间：{elapsed_time:.2f}秒"))
            if selected_model_id:
                st.caption(
                    t(
                        f"🧭 Mode: {localized_mode(selected_mode)} | Routed model: {selected_model_id}",
                        f"🧭 模式：{localized_mode(selected_mode)} | 路由模型：{selected_model_id}",
                    )
                )
            
            # Display LLM-generated answer
            st.markdown(result["content"])
            if result["citations"]:
                with st.expander(
                    _citation_label(len(result["citations"]), ui_language),
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
                                source_label += (
                                    f" (page {page})"
                                    if ui_language == "English"
                                    else f" (page {page})"
                                )

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
            "ui_language": ui_language,
        }
        st.session_state.messages.append(assistant_msg)
        
        # Update conversation
        if current_conv:
            current_conv["messages"] = st.session_state.messages.copy()
            _save_conversations_to_disk()
        
        st.rerun()

