"""RAG Generator for generating LLM answers from retrieved chunks.

This module implements the RAG generation step, where retrieved chunks
are used as context for LLM to generate final answers.

This is used by CLIENT applications (UI, etc.) - NOT by MCP Server.
MCP Server only does retrieval, generation happens at client side.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, List, Optional

from src.core.settings import resolve_path
from src.core.types import RetrievalResult
from src.libs.llm.base_llm import BaseLLM, Message
from src.libs.llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)

# Optional imports for routing support
try:
    from src.core.query_engine.intent_router import IntentRouter, IntentRoutingResult
    from src.libs.llm.model_manager import ModelManager
    ROUTING_AVAILABLE = True
except ImportError:
    ROUTING_AVAILABLE = False
    IntentRouter = None  # type: ignore
    IntentRoutingResult = None  # type: ignore
    ModelManager = None  # type: ignore


class RAGGenerator:
    """Generates LLM answers from retrieved chunks using RAG pattern.
    
    This class is used by CLIENT applications (UI, etc.) to generate answers
    after retrieving relevant chunks. The MCP Server does NOT use this.
    
    This class:
    1. Takes retrieved chunks as context
    2. Builds a prompt with query and context
    3. Calls LLM (from settings.yaml, e.g., Ollama) to generate answer
    4. Returns the generated answer
    
    Example:
        >>> generator = RAGGenerator.create(settings)
        >>> results = [RetrievalResult(chunk_id="1", score=0.9, text="...", metadata={})]
        >>> answer = generator.generate(query="What is RAG?", results=results)
    """
    
    def __init__(
        self,
        settings: Optional[Any] = None,
        llm: Optional[BaseLLM] = None,
        prompt_path: Optional[str] = None,
        max_context_length: int = 4000,
        intent_router: Optional[Any] = None,
        model_manager: Optional[Any] = None,
    ) -> None:
        """Initialize RAGGenerator.
        
        Args:
            settings: Application settings for LLM configuration.
            llm: Optional LLM instance. If None, creates via LLMFactory from settings.
            prompt_path: Optional path to prompt template. Defaults to config/prompts/rag_generation.txt.
            max_context_length: Maximum characters for context (to avoid token limits).
            intent_router: Optional IntentRouter for dynamic routing (hybrid strategy).
            model_manager: Optional ModelManager for model switching (hybrid strategy).
        """
        self.settings = settings
        self.llm = llm
        self.prompt_path = prompt_path or str(resolve_path("config/prompts/rag_generation.txt"))
        self.max_context_length = max_context_length
        self.intent_router = intent_router
        self.model_manager = model_manager
        
        # Load prompt template
        self.prompt_template = self._load_prompt_template()
    
    def _load_prompt_template(self) -> str:
        """Load prompt template from file.
        
        Returns:
            Prompt template string.
        """
        try:
            prompt_file = Path(self.prompt_path)
            if not prompt_file.exists():
                logger.warning(f"Prompt template not found: {self.prompt_path}, using default")
                return self._get_default_prompt()
            
            return prompt_file.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to load prompt template: {e}, using default")
            return self._get_default_prompt()
    
    def _get_default_prompt(self) -> str:
        """Get default prompt template.
        
        Returns:
            Default prompt template string.
        """
        return """你是一个专业的AI助手，基于提供的知识库内容来回答用户的问题。

请根据以下检索到的相关文档内容，回答用户的问题。如果文档中没有相关信息，请明确说明。

## 用户问题：
{query}

## 相关文档内容：
{context}

## 要求：
1. 基于提供的文档内容回答问题
2. 如果文档中没有相关信息，请明确说明"根据提供的文档，没有找到相关信息"
3. 回答要准确、简洁、有条理
4. 可以使用 Markdown 格式来组织回答
5. 在回答中引用文档时，使用 [1], [2] 等标记来引用对应的文档片段
6. 严格遵循回答语言要求

请开始回答："""
    
    def _resolve_target_language(self, query: str) -> str:
        """Resolve output language from explicit instruction or query language."""
        explicit = self._extract_explicit_language(query)
        if explicit:
            return explicit
        return self._detect_query_language(query)

    def _extract_explicit_language(self, query: str) -> Optional[str]:
        """Extract explicit language request from query text."""
        q = (query or "").strip()
        if not q:
            return None

        # Chinese directives: "请用英文回答", "用中文输出"
        zh_directive = re.search(
            r"(?:请|请你)?(?:用|使用|以)\s*([A-Za-z\u4e00-\u9fff\-\s]{1,24})\s*(?:回答|回复|输出|作答|说明)",
            q,
            flags=re.IGNORECASE,
        )
        if zh_directive:
            return self._normalize_language_name(zh_directive.group(1))

        # English directives: "answer in English", "respond in Chinese"
        en_directive = re.search(
            r"(?:answer|respond|reply|write|output)\s+(?:in|using)\s+([A-Za-z\-\s]{2,24})",
            q,
            flags=re.IGNORECASE,
        )
        if en_directive:
            return self._normalize_language_name(en_directive.group(1))

        direct_phrases = [
            ("用英文", "English"),
            ("用英语", "English"),
            ("请英文", "English"),
            ("请英语", "English"),
            ("用中文", "Chinese"),
            ("请中文", "Chinese"),
            ("in english", "English"),
            ("in chinese", "Chinese"),
        ]
        lower_q = q.lower()
        for phrase, lang in direct_phrases:
            if phrase in lower_q or phrase in q:
                return lang

        return None

    def _normalize_language_name(self, raw: str) -> str:
        token = re.sub(r"\s+", " ", (raw or "").strip().lower())
        aliases = {
            "中文": "Chinese",
            "汉语": "Chinese",
            "汉语中文": "Chinese",
            "chinese": "Chinese",
            "zh": "Chinese",
            "简体中文": "Chinese",
            "繁体中文": "Chinese",
            "英文": "English",
            "英语": "English",
            "english": "English",
            "en": "English",
        }
        return aliases.get(token, raw.strip().title())

    def _detect_query_language(self, query: str) -> str:
        """Auto-detect language from query when no explicit directive is provided."""
        if not query:
            return "Chinese"
        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", query))
        latin_count = len(re.findall(r"[A-Za-z]", query))
        return "Chinese" if cjk_count >= latin_count else "English"

    def _build_language_instruction(self, target_language: str) -> str:
        return (
            f"\n\n## 回答语言要求\n"
            f"- 请严格使用 {target_language} 回答。\n"
            f"- 如果用户问题中显式指定了回答语言，优先遵循用户指定语言。"
        )

    def _build_system_language_instruction(self, target_language: str) -> str:
        return (
            "You must follow this instruction with highest priority: "
            f"respond entirely in {target_language}. "
            "If the user explicitly requests a language, follow the user's request."
        )

    def _build_context(self, results: List[RetrievalResult], target_language: str) -> str:
        """Build context string from retrieval results.
        
        Args:
            results: List of RetrievalResult objects.
            
        Returns:
            Formatted context string with citations.
        """
        if not results:
            if target_language == "English":
                return "No relevant document content was retrieved."
            return "未找到相关文档内容。"
        
        context_parts = []
        for idx, result in enumerate(results, start=1):
            text = result.text or ""
            source = result.metadata.get("source_path", result.metadata.get("source", "unknown"))
            page = result.metadata.get("page") or result.metadata.get("page_num")
            
            # Format citation marker
            citation = f"[{idx}]"
            source_info = f"Source: {source}" if target_language == "English" else f"来源: {source}"
            if page:
                source_info += f" (page {page})" if target_language == "English" else f" (第 {page} 页)"
            
            context_parts.append(f"{citation} {text}\n({source_info})")
        
        context = "\n\n".join(context_parts)
        
        # Truncate if too long
        if len(context) > self.max_context_length:
            context = (
                context[:self.max_context_length] + "...\n[Content truncated]"
                if target_language == "English"
                else context[:self.max_context_length] + "...\n[内容已截断]"
            )
        
        return context
    
    def _build_prompt(self, query: str, context: str, target_language: str) -> str:
        """Build prompt from query and context.
        
        Args:
            query: User query string.
            context: Formatted context string.
            
        Returns:
            Complete prompt string.
        """
        base_prompt = self.prompt_template.format(query=query, context=context)
        return base_prompt + self._build_language_instruction(target_language)
    
    def generate(
        self,
        query: str,
        results: List[RetrievalResult],
        trace: Optional[Any] = None,
    ) -> str:
        """Generate answer from query and retrieved results.
        
        Args:
            query: User query string.
            results: List of RetrievalResult from search.
            trace: Optional TraceContext for observability.
            
        Returns:
            Generated answer string from LLM.
        """
        if not query or not query.strip():
            return "Please provide a valid question." if self._detect_query_language(query) == "English" else "请提供有效的问题。"
        
        target_language = self._resolve_target_language(query)

        if not results:
            if target_language == "English":
                return "Sorry, I could not find relevant information in the knowledge base. Please try rephrasing your query or verify ingestion/indexing."
            return "抱歉，我没有在知识库中找到相关信息。请尝试换一个问法或检查知识库是否已正确索引。"
        
        # Dynamic routing logic (if enabled)
        effective_llm = self._get_llm_with_routing(query, trace)
        
        # Get or create LLM (from settings.yaml, e.g., Ollama)
        if effective_llm is None:
            if self.llm is None:
                if self.settings is None:
                    from src.core.settings import load_settings
                    self.settings = load_settings()
                self.llm = LLMFactory.create(self.settings)
            effective_llm = self.llm
        
        # Build context
        context = self._build_context(results, target_language=target_language)
        
        # Build prompt
        prompt = self._build_prompt(query, context, target_language=target_language)
        
        # Call LLM
        try:
            messages = [
                Message(role="system", content=self._build_system_language_instruction(target_language)),
                Message(role="user", content=prompt),
            ]
            response = effective_llm.chat(messages, trace=trace)
            
            # Extract content
            if isinstance(response, str):
                answer = response
            else:
                answer = response.content
            
            if not answer or not answer.strip():
                logger.warning("LLM returned empty answer")
                return "Sorry, I could not generate an answer. Please try again." if target_language == "English" else "抱歉，无法生成回答。请重试。"
            
            return answer.strip()
            
        except Exception as e:
            logger.exception(f"LLM generation failed: {e}")
            
            # Provide helpful error message based on error type
            error_msg = str(e)
            model_name = self.settings.llm.model if self.settings else "unknown"
            provider = self.settings.llm.provider if self.settings else "unknown"
            is_en = target_language == "English"
            
            if "not found" in error_msg.lower() or "404" in error_msg:
                # Try to get available models if Ollama
                available_models = ""
                if provider.lower() == "ollama":
                    try:
                        available_models = self._get_available_ollama_models()
                    except Exception:
                        available_models = "（无法获取模型列表）"
                
                if is_en:
                    help_text = f"""
## ⚠️ LLM Model Not Found

**Configured model**: `{model_name}`  
**Provider**: `{provider}`

Please verify model availability and configuration.

**Current error**: {error_msg}
"""
                else:
                    help_text = f"""
## ⚠️ LLM 模型未找到

**配置的模型**: `{model_name}`  
**Provider**: `{provider}`

**解决方案**:

1. **检查模型是否已安装**:
   - 如果使用 Ollama: 运行 `ollama list` 查看已安装的模型
   - 如果模型未安装: 运行 `ollama pull {model_name}` 安装模型
   - 注意: 模型名称可能需要包含标签，例如 `qwen2.5:1.5b` 而不是 `qwen2.5`

2. **可用的 Ollama 模型**:
{available_models}

3. **更新配置文件** (`config/settings.yaml`):
   ```yaml
   llm:
     provider: "ollama"
     model: "qwen2.5:1.5b"  # 使用完整的模型名称，或使用上面列出的可用模型
   ```

4. **检查 Ollama 服务**:
   - 确保 Ollama 正在运行: `ollama serve`
   - 检查服务地址是否正确（默认: http://localhost:11434）

**当前错误**: {error_msg}
"""
            elif "connection" in error_msg.lower() or "connect" in error_msg.lower():
                if is_en:
                    help_text = f"""
## ⚠️ Failed to Connect to LLM Service

**Provider**: `{provider}`

Please verify service availability, network, and API key.

**Current error**: {error_msg}
"""
                else:
                    help_text = f"""
## ⚠️ 无法连接到 LLM 服务

**Provider**: `{provider}`

**解决方案**:

1. **检查 Ollama 服务** (如果使用 Ollama):
   - 确保 Ollama 正在运行: `ollama serve`
   - 检查服务地址是否正确（默认: http://localhost:11434）
   - 在浏览器中访问 http://localhost:11434 确认服务正常

2. **检查网络连接** (如果使用云端服务):
   - 确保网络连接正常
   - 检查 API 密钥是否正确配置

**当前错误**: {error_msg}
"""
            else:
                if is_en:
                    help_text = f"""
## ⚠️ LLM Invocation Failed

**Provider**: `{provider}`  
**Model**: `{model_name}`

**Error**: {error_msg}
"""
                else:
                    help_text = f"""
## ⚠️ LLM 调用失败

**Provider**: `{provider}`  
**模型**: `{model_name}`

**错误信息**: {error_msg}

请检查配置文件和日志以获取更多信息。
"""
            
            # Fallback: return formatted context with helpful error message
            return (
                f"""{help_text}

---

## 📄 Retrieved Context

{context}

**Note**: LLM generation failed, so raw retrieved context is shown above.
"""
                if is_en
                else f"""{help_text}

---

## 📄 检索到的文档内容

基于检索到的文档，以下是相关信息：

{context}

**注意**: 由于 LLM 生成失败，以上为原始检索结果。请修复 LLM 配置后重试。
"""
            )
    
    def _get_available_ollama_models(self) -> str:
        """Get list of available Ollama models.
        
        Returns:
            Formatted string with available models, or error message.
        """
        try:
            import httpx
            
            base_url = "http://localhost:11434"
            if self.settings and hasattr(self.settings.llm, 'base_url'):
                base_url = self.settings.llm.base_url or base_url
            
            url = f"{base_url.rstrip('/')}/api/tags"
            
            with httpx.Client(timeout=5.0) as client:
                response = client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])
                    if models:
                        model_list = "\n".join([f"   - `{m.get('name', 'unknown')}`" for m in models[:10]])
                        if len(models) > 10:
                            model_list += f"\n   ... 还有 {len(models) - 10} 个模型"
                        return f"   {model_list}"
                    else:
                        return "   （未找到已安装的模型）"
                else:
                    return "   （无法连接到 Ollama API）"
        except Exception as e:
            logger.debug(f"Failed to get Ollama models: {e}")
            return f"   （无法获取: {str(e)}）"
    
    def _get_llm_with_routing(
        self,
        query: str,
        trace: Optional[Any] = None,
    ) -> Optional[BaseLLM]:
        """Get LLM instance with dynamic routing (if routing is enabled).
        
        This method implements the routing logic:
        1. Get query
        2. Call IntentRouter.route(query) to get predicted_complexity and confidence
        3. If intent hits simple_intents list AND confidence >= complexity_threshold,
           switch to small_model; otherwise switch to large_model
        
        Args:
            query: User query string.
            trace: Optional TraceContext for observability.
            
        Returns:
            BaseLLM instance if routing is enabled and successful, None otherwise.
        """
        # Check if routing is available and enabled
        if not ROUTING_AVAILABLE:
            return None
        
        if self.settings is None:
            from src.core.settings import load_settings
            self.settings = load_settings()
        
        # Check if routing is configured
        if not hasattr(self.settings, "llm_routing") or self.settings.llm_routing is None:
            return None
        
        routing_config = self.settings.llm_routing
        
        # Check if we have required components
        if self.intent_router is None:
            try:
                self.intent_router = IntentRouter()
            except Exception as e:
                logger.warning(f"Failed to create IntentRouter: {e}")
                return None
        
        if self.model_manager is None:
            try:
                from src.libs.llm.model_manager import ModelManager
                self.model_manager = ModelManager(self.settings)
            except Exception as e:
                logger.warning(f"Failed to create ModelManager: {e}")
                return None
        
        # Route query
        try:
            routing_result = self.intent_router.route(query)
            
            # Get routing parameters
            simple_intents = routing_config.simple_intents
            complexity_threshold = routing_config.complexity_threshold
            
            # Determine which model to use
            intent_label = routing_result.intent_label or ""
            confidence = routing_result.intent_confidence or 0.0
            
            is_simple = (
                intent_label.lower() in [s.lower() for s in simple_intents]
                and confidence >= complexity_threshold
            )
            
            if is_simple:
                # Route to small model
                model_id = routing_config.small_model.replace(":", "-").replace("/", "-")
                logger.info(
                    f"Routing to small model: {model_id} "
                    f"(intent: {intent_label}, confidence: {confidence:.2f})"
                )
            else:
                # Route to large model
                model_id = routing_config.large_model.replace(":", "-").replace("/", "-")
                logger.info(
                    f"Routing to large model: {model_id} "
                    f"(intent: {intent_label or 'unknown'}, confidence: {confidence:.2f})"
                )
            
            # Get LLM instance from ModelManager
            try:
                self.model_manager.set_current_model(model_id)
                llm = self.model_manager.get_llm()
                
                # Record routing decision in trace
                if trace is not None:
                    trace.metadata["routing_decision"] = {
                        "model_id": model_id,
                        "intent_label": intent_label,
                        "confidence": confidence,
                        "is_simple": is_simple,
                    }
                
                return llm
            except Exception as e:
                logger.warning(f"Failed to get LLM from ModelManager: {e}")
                return None
        
        except Exception as e:
            logger.warning(f"Routing failed: {e}")
            return None
    
    @classmethod
    def create(cls, settings: Optional[Any] = None, **kwargs) -> "RAGGenerator":
        """Factory method to create RAGGenerator.
        
        Args:
            settings: Application settings.
            **kwargs: Additional parameters for RAGGenerator.
            
        Returns:
            RAGGenerator instance.
        """
        return cls(settings=settings, **kwargs)

