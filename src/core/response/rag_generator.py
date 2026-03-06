"""RAG Generator for generating LLM answers from retrieved chunks.

This module implements the RAG generation step, where retrieved chunks
are used as context for LLM to generate final answers.

This is used by CLIENT applications (UI, etc.) - NOT by MCP Server.
MCP Server only does retrieval, generation happens at client side.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

from src.core.settings import resolve_path
from src.core.types import RetrievalResult
from src.libs.llm.base_llm import BaseLLM, Message
from src.libs.llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


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
    ) -> None:
        """Initialize RAGGenerator.
        
        Args:
            settings: Application settings for LLM configuration.
            llm: Optional LLM instance. If None, creates via LLMFactory from settings.
            prompt_path: Optional path to prompt template. Defaults to config/prompts/rag_generation.txt.
            max_context_length: Maximum characters for context (to avoid token limits).
        """
        self.settings = settings
        self.llm = llm
        self.prompt_path = prompt_path or str(resolve_path("config/prompts/rag_generation.txt"))
        self.max_context_length = max_context_length
        
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

请开始回答："""
    
    def _build_context(self, results: List[RetrievalResult]) -> str:
        """Build context string from retrieval results.
        
        Args:
            results: List of RetrievalResult objects.
            
        Returns:
            Formatted context string with citations.
        """
        if not results:
            return "未找到相关文档内容。"
        
        context_parts = []
        for idx, result in enumerate(results, start=1):
            text = result.text or ""
            source = result.metadata.get("source_path", result.metadata.get("source", "unknown"))
            page = result.metadata.get("page") or result.metadata.get("page_num")
            
            # Format citation marker
            citation = f"[{idx}]"
            source_info = f"来源: {source}"
            if page:
                source_info += f" (第 {page} 页)"
            
            context_parts.append(f"{citation} {text}\n({source_info})")
        
        context = "\n\n".join(context_parts)
        
        # Truncate if too long
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "...\n[内容已截断]"
        
        return context
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt from query and context.
        
        Args:
            query: User query string.
            context: Formatted context string.
            
        Returns:
            Complete prompt string.
        """
        return self.prompt_template.format(query=query, context=context)
    
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
            return "请提供有效的问题。"
        
        if not results:
            return "抱歉，我没有在知识库中找到相关信息。请尝试换一个问法或检查知识库是否已正确索引。"
        
        # Get or create LLM (from settings.yaml, e.g., Ollama)
        if self.llm is None:
            if self.settings is None:
                from src.core.settings import load_settings
                self.settings = load_settings()
            self.llm = LLMFactory.create(self.settings)
        
        # Build context
        context = self._build_context(results)
        
        # Build prompt
        prompt = self._build_prompt(query, context)
        
        # Call LLM
        try:
            messages = [Message(role="user", content=prompt)]
            response = self.llm.chat(messages, trace=trace)
            
            # Extract content
            if isinstance(response, str):
                answer = response
            else:
                answer = response.content
            
            if not answer or not answer.strip():
                logger.warning("LLM returned empty answer")
                return "抱歉，无法生成回答。请重试。"
            
            return answer.strip()
            
        except Exception as e:
            logger.exception(f"LLM generation failed: {e}")
            
            # Provide helpful error message based on error type
            error_msg = str(e)
            model_name = self.settings.llm.model if self.settings else "unknown"
            provider = self.settings.llm.provider if self.settings else "unknown"
            
            if "not found" in error_msg.lower() or "404" in error_msg:
                # Try to get available models if Ollama
                available_models = ""
                if provider.lower() == "ollama":
                    try:
                        available_models = self._get_available_ollama_models()
                    except Exception:
                        available_models = "（无法获取模型列表）"
                
                help_text = f"""
## ⚠️ LLM 模型未找到

**配置的模型**: `{model_name}`  
**Provider**: `{provider}`

**解决方案**:

1. **检查模型是否已安装**:
   - 如果使用 Ollama: 运行 `ollama list` 查看已安装的模型
   - 如果模型未安装: 运行 `ollama pull {model_name}` 安装模型
   - 注意: 模型名称可能需要包含标签，例如 `llama3.1:8b` 而不是 `llama3.1`

2. **可用的 Ollama 模型**:
{available_models}

3. **更新配置文件** (`config/settings.yaml`):
   ```yaml
   llm:
     provider: "ollama"
     model: "llama3.1:8b"  # 使用完整的模型名称，或使用上面列出的可用模型
   ```

4. **检查 Ollama 服务**:
   - 确保 Ollama 正在运行: `ollama serve`
   - 检查服务地址是否正确（默认: http://localhost:11434）

**当前错误**: {error_msg}
"""
            elif "connection" in error_msg.lower() or "connect" in error_msg.lower():
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
                help_text = f"""
## ⚠️ LLM 调用失败

**Provider**: `{provider}`  
**模型**: `{model_name}`

**错误信息**: {error_msg}

请检查配置文件和日志以获取更多信息。
"""
            
            # Fallback: return formatted context with helpful error message
            return f"""{help_text}

---

## 📄 检索到的文档内容

基于检索到的文档，以下是相关信息：

{context}

**注意**: 由于 LLM 生成失败，以上为原始检索结果。请修复 LLM 配置后重试。
"""
    
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

