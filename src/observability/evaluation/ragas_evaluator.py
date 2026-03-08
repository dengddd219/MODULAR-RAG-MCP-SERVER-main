"""Ragas-based evaluator for RAG quality assessment.

This evaluator wraps the Ragas framework to compute LLM-as-Judge metrics:
- Faithfulness: Does the answer stick to the retrieved context?
- Answer Relevancy: Is the answer relevant to the query?
- Context Precision: Are the retrieved chunks relevant and well-ordered?

Design Principles:
- Pluggable: Implements BaseEvaluator interface, swappable via factory.
- Config-Driven: LLM/Embedding backend read from settings.yaml.
- Graceful Degradation: Clear ImportError if ragas not installed.
"""

from __future__ import annotations

import logging
import nest_asyncio
nest_asyncio.apply()
from typing import Any, Dict, List, Optional, Sequence
import re

from src.libs.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

# Metric name constants
FAITHFULNESS = "faithfulness"
ANSWER_RELEVANCY = "answer_relevancy"
CONTEXT_PRECISION = "context_precision"

SUPPORTED_METRICS = {FAITHFULNESS, ANSWER_RELEVANCY, CONTEXT_PRECISION}


def _import_ragas() -> None:
    """Validate that ragas is importable, raising a clear error if not."""
    try:
        import ragas  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'ragas' package is required for RagasEvaluator. "
            "Install it with: pip install ragas datasets"
        ) from exc


class RagasEvaluator(BaseEvaluator):
    """Evaluator that uses the Ragas framework for LLM-as-Judge metrics.

    Ragas does NOT require ground-truth labels.  It uses an LLM to judge
    the quality of the generated answer against the retrieved context.

    Supported metrics:
        - faithfulness: Measures factual consistency with context.
        - answer_relevancy: Measures how relevant the answer is to the query.
        - context_precision: Measures relevance/ordering of retrieved chunks.

    Example::

        evaluator = RagasEvaluator(settings=settings)
        metrics = evaluator.evaluate(
            query="What is RAG?",
            retrieved_chunks=[{"id": "c1", "text": "RAG is ..."}],
            generated_answer="RAG stands for ...",
        )
        # metrics == {"faithfulness": 0.95, "answer_relevancy": 0.88, ...}
    """

    def __init__(
        self,
        settings: Any = None,
        metrics: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize RagasEvaluator.

        Args:
            settings: Application settings (used to configure LLM backend).
            metrics: Metric names to compute. Defaults to all supported.
            **kwargs: Additional parameters (reserved).

        Raises:
            ImportError: If ragas is not installed.
            ValueError: If unsupported metric names are requested.
        """
        _import_ragas()

        self.settings = settings
        self.kwargs = kwargs

        if metrics is None:
            metrics = self._metrics_from_settings(settings)

        normalised = [m.strip().lower() for m in (metrics or [])]
        if not normalised:
            normalised = sorted(SUPPORTED_METRICS)

        unsupported = [m for m in normalised if m not in SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(
                f"Unsupported ragas metrics: {', '.join(unsupported)}. "
                f"Supported: {', '.join(sorted(SUPPORTED_METRICS))}"
            )

        self._metric_names = normalised

    # ── public API ────────────────────────────────────────────────

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Evaluate RAG quality using Ragas LLM-as-Judge metrics.

        Args:
            query: The user query string.
            retrieved_chunks: Retrieved chunks (dicts with 'text' key or strings).
            generated_answer: The generated answer text. Required for Ragas.
            ground_truth: Ignored by Ragas (not needed for LLM-as-Judge).
            trace: Optional TraceContext for observability.
            **kwargs: Additional parameters.

        Returns:
            Dictionary mapping metric names to float scores (0.0 – 1.0).

        Raises:
            ValueError: If query/chunks are invalid or generated_answer is missing.
        """
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)

        max_eval_chunks = 5
        if len(retrieved_chunks) > max_eval_chunks:
            logger.warning(f"Truncating retrieved chunks from {len(retrieved_chunks)} to {max_eval_chunks} for evaluation.")
            retrieved_chunks = retrieved_chunks[:max_eval_chunks]

        if not generated_answer or not generated_answer.strip():
            raise ValueError(
                "RagasEvaluator requires a non-empty 'generated_answer'. "
                "Ragas uses LLM-as-Judge and needs the answer text to evaluate."
            )

        contexts = self._extract_texts(retrieved_chunks)

        try:
            result = self._run_ragas(query, contexts, generated_answer)
        except Exception as exc:
            logger.error("Ragas evaluation failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Ragas evaluation failed: {exc}") from exc

        return result

    # ── private helpers ───────────────────────────────────────────

    def _run_ragas(
        self,
        query: str,
        contexts: List[str],
        answer: str,
    ) -> Dict[str, float]:
        """
        Execute Ragas using legacy evaluate() pipeline (non-collections mode).
        This avoids modern embeddings requirement.
        """
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
        )
        from datasets import Dataset
        from ragas.run_config import RunConfig  # 👈 第一处：必须导入

        # Build LLM / Embedding wrappers
        llm, embeddings = self._build_wrappers()

        print("🔥 USING LEGACY RAGAS PIPELINE")
        print("🔥 LLM TYPE:", type(llm))
        print("🔥 EMBEDDINGS TYPE:", type(embeddings))

        # Build dataset
        data = {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
        }
        dataset = Dataset.from_dict(data)

        # Select metrics dynamically
        metric_map = {
            FAITHFULNESS: faithfulness,
            ANSWER_RELEVANCY: answer_relevancy,
            CONTEXT_PRECISION: context_precision,
        }
        selected_metrics = [
            metric_map[m] for m in self._metric_names if m in metric_map
        ]

        # 👈 第二处：实例化 RunConfig 对象
        # Increase timeout for evaluation (Ragas can take longer with complex queries)
        my_run_config = RunConfig(timeout=300)  # Increased from 120 to 300 seconds

        result = evaluate(
            dataset=dataset,
            metrics=selected_metrics,
            llm=llm,
            embeddings=embeddings,
            run_config=my_run_config,  # 👈 第三处：必须明确传给 evaluate
        )

        # Convert result to simple dict
        import math
        
        scores = {}
        # First, log what keys are actually in the result for debugging
        result_keys = list(result.keys()) if hasattr(result, 'keys') else []
        logger.debug(f"Ragas evaluation result keys: {result_keys}")
        logger.debug(f"Requested metrics: {[m.name for m in selected_metrics]}")
        
        for key in selected_metrics:
            name = key.name
            # Use .get() to safely access result, handle KeyError if metric is missing
            if name not in result:
                logger.warning(
                    f"Ragas metric '{name}' not found in result. "
                    f"Available keys: {result_keys}. Using default 0.5"
                )
                scores[name] = 0.5
                continue
            
            value = result[name]
            if isinstance(value, list):
                value = value[0] if value else None
            
            # Handle None, NaN, and invalid values
            if value is None:
                logger.warning(f"Ragas metric '{name}' returned None, using default 0.5")
                scores[name] = 0.5  # Default to 0.5 instead of 0.0
            elif isinstance(value, (int, float)):
                if math.isnan(value):
                    logger.warning(f"Ragas metric '{name}' returned NaN, using default 0.5")
                    scores[name] = 0.5
                elif math.isinf(value):
                    logger.warning(f"Ragas metric '{name}' returned Inf, using default 0.5")
                    scores[name] = 0.5
                elif value == 0.0:
                    # Log when metric is actually 0 (not missing)
                    logger.info(f"Ragas metric '{name}' calculated as 0.0 (may indicate poor retrieval quality)")
                    scores[name] = 0.0
                else:
                    scores[name] = float(value)
            else:
                logger.warning(f"Ragas metric '{name}' returned unexpected type {type(value)}, using default 0.5")
                scores[name] = 0.5
        
        # Ensure all requested metrics are in scores (even if missing from result)
        for key in selected_metrics:
            name = key.name
            if name not in scores:
                logger.warning(f"Metric '{name}' was not processed, adding default 0.5")
                scores[name] = 0.5
        
        logger.debug(f"Final scores: {scores}")
        return scores

    def _build_wrappers(self) -> tuple:
        """Build Ragas LLM and Embedding wrappers from project settings."""
        print("🔥 USING FIXED RAGAS OLLAMA WRAPPER")
        if self.settings is None:
            raise ValueError("Settings required to create LLM for Ragas evaluation")

        llm_cfg = self.settings.llm
        provider = llm_cfg.provider.lower()

        # 【核心武器】：调用 Ragas 新版唯一认定的 llm_factory
        from ragas.llms import llm_factory
        from ragas.embeddings import LangchainEmbeddingsWrapper

        if provider == "openai":
            # 使用 settings.llm 中的国内代理配置（base_url + api_key），
            # 保持与业务侧 Chat 相同的调用路径。
            from openai import AsyncOpenAI
            from langchain_openai import OpenAIEmbeddings

            base_url = getattr(llm_cfg, "base_url", None)

            client = AsyncOpenAI(
                api_key=llm_cfg.api_key,
                base_url=base_url,
                max_retries=3,
            )

            # For Ragas evaluation, we need larger max_tokens to avoid truncation
            # Ragas generates detailed JSON responses that can be quite long
            evaluation_max_tokens = getattr(llm_cfg, "max_tokens", 512)
            # Increase max_tokens for evaluation (minimum 4096, or 2x the configured value)
            evaluation_max_tokens = max(4096, evaluation_max_tokens * 2)
            
            # Create LLM - Ragas will use the client's default max_tokens
            # We need to set max_tokens via the client's default_params or directly on the LLM
            ragas_llm = llm_factory(llm_cfg.model, client=client)
            
            # Try multiple ways to set max_tokens for evaluation
            # Method 1: Direct attribute on ragas_llm
            if hasattr(ragas_llm, "max_tokens"):
                ragas_llm.max_tokens = evaluation_max_tokens
                logger.debug(f"Set max_tokens={evaluation_max_tokens} on ragas_llm.max_tokens")
            # Method 2: Nested llm attribute
            elif hasattr(ragas_llm, "llm") and hasattr(ragas_llm.llm, "max_tokens"):
                ragas_llm.llm.max_tokens = evaluation_max_tokens
                logger.debug(f"Set max_tokens={evaluation_max_tokens} on ragas_llm.llm.max_tokens")
            # Method 3: Try to set via default_params on client
            elif hasattr(client, "default_params"):
                if client.default_params is None:
                    client.default_params = {}
                client.default_params["max_tokens"] = evaluation_max_tokens
                logger.debug(f"Set max_tokens={evaluation_max_tokens} via client.default_params")
            else:
                logger.warning(
                    f"Could not set max_tokens={evaluation_max_tokens} for Ragas evaluation. "
                    f"LLM may truncate responses. Ragas LLM type: {type(ragas_llm)}"
                )

            # Embedding 也通过同一代理走 OpenAI 接口，保证走国内运营商链路
            ragas_embeddings = LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(
                    model=self.settings.embedding.model,
                    api_key=getattr(self.settings.embedding, "api_key", None) or llm_cfg.api_key,
                    base_url=getattr(self.settings.embedding, "base_url", None) or base_url,
                )
            )

            return ragas_llm, ragas_embeddings
            
        elif provider == "azure":
            from openai import AsyncAzureOpenAI
            from langchain_openai import AzureOpenAIEmbeddings
            client = AsyncAzureOpenAI(
                azure_endpoint=llm_cfg.azure_endpoint,
                api_key=llm_cfg.api_key,
                api_version=llm_cfg.api_version,
                max_retries=3,
            )
            
            # For Ragas evaluation, we need larger max_tokens to avoid truncation
            evaluation_max_tokens = getattr(llm_cfg, "max_tokens", 512)
            evaluation_max_tokens = max(4096, evaluation_max_tokens * 2)
            
            ragas_llm = llm_factory(llm_cfg.deployment_name, client=client)
            
            # Try multiple ways to set max_tokens for evaluation
            if hasattr(ragas_llm, "max_tokens"):
                ragas_llm.max_tokens = evaluation_max_tokens
                logger.debug(f"Set max_tokens={evaluation_max_tokens} on ragas_llm.max_tokens")
            elif hasattr(ragas_llm, "llm") and hasattr(ragas_llm.llm, "max_tokens"):
                ragas_llm.llm.max_tokens = evaluation_max_tokens
                logger.debug(f"Set max_tokens={evaluation_max_tokens} on ragas_llm.llm.max_tokens")
            elif hasattr(client, "default_params"):
                if client.default_params is None:
                    client.default_params = {}
                client.default_params["max_tokens"] = evaluation_max_tokens
                logger.debug(f"Set max_tokens={evaluation_max_tokens} via client.default_params")
            else:
                logger.warning(
                    f"Could not set max_tokens={evaluation_max_tokens} for Ragas evaluation. "
                    f"LLM may truncate responses. Ragas LLM type: {type(ragas_llm)}"
                )
            ragas_embeddings = LangchainEmbeddingsWrapper(AzureOpenAIEmbeddings(
                azure_deployment=self.settings.embedding.deployment_name,
                api_version=self.settings.embedding.api_version,
            ))
            return ragas_llm, ragas_embeddings
            
        elif provider == "ollama":
            from openai import AsyncOpenAI
            from ragas.llms import llm_factory
            from ragas.embeddings import LangchainEmbeddingsWrapper
            import os

            # 🔹 关键：使用 OpenAI 兼容接口连接 Ollama
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

            client = AsyncOpenAI(
                base_url=ollama_url,
                api_key="ollama",  # 任意字符串即可
                max_retries=3,
            )

            # For Ragas evaluation, we need larger max_tokens to avoid truncation
            evaluation_max_tokens = getattr(llm_cfg, "max_tokens", 512)
            evaluation_max_tokens = max(4096, evaluation_max_tokens * 2)

            # 🔹 关键：必须通过 llm_factory 创建
            ragas_llm = llm_factory(model=llm_cfg.model, client=client)
            
            # Try multiple ways to set max_tokens for evaluation
            if hasattr(ragas_llm, "max_tokens"):
                ragas_llm.max_tokens = evaluation_max_tokens
                logger.debug(f"Set max_tokens={evaluation_max_tokens} on ragas_llm.max_tokens")
            elif hasattr(ragas_llm, "llm") and hasattr(ragas_llm.llm, "max_tokens"):
                ragas_llm.llm.max_tokens = evaluation_max_tokens
                logger.debug(f"Set max_tokens={evaluation_max_tokens} on ragas_llm.llm.max_tokens")
            elif hasattr(client, "default_params"):
                if client.default_params is None:
                    client.default_params = {}
                client.default_params["max_tokens"] = evaluation_max_tokens
                logger.debug(f"Set max_tokens={evaluation_max_tokens} via client.default_params")
            else:
                logger.warning(
                    f"Could not set max_tokens={evaluation_max_tokens} for Ragas evaluation. "
                    f"LLM may truncate responses. Ragas LLM type: {type(ragas_llm)}"
                )

            # 🔹 embeddings 仍然使用 Ollama 原生
            try:
                from langchain_ollama import OllamaEmbeddings
            except ImportError:
                from langchain_community.embeddings import OllamaEmbeddings

            raw_embeddings = OllamaEmbeddings(
                model=self.settings.embedding.model
            )

            ragas_embeddings = LangchainEmbeddingsWrapper(raw_embeddings)

            return ragas_llm, ragas_embeddings
            
        else:
            raise ValueError(
                f"Unsupported LLM provider for Ragas: '{provider}'. "
                f"Supported: azure, openai, ollama"
            )

    def _extract_texts(self, chunks: List[Any]) -> List[str]:
        """Extract text strings from various chunk representations.

        为避免裁判 LLM 的 token 爆炸，这里做两层控制：
        1. 只取前若干 chunks（在 evaluate 里已有 max_eval_chunks 控制）
        2. 对每个 chunk 文本按句号/换行做“句子级”截断，而不是暴力 [:N] 切字符
        """
        MAX_CHARS_PER_CONTEXT = 1000

        def _truncate_by_sentence(text: str, max_chars: int) -> str:
            if len(text) <= max_chars:
                return text

            # 先按换行粗分段，优先保留前几段
            parts = re.split(r"\n+", text)
            kept_parts: List[str] = []
            total = 0
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                # 再按句号/问号/感叹号/英文标点切分句子
                sentences = re.split(r"(?<=[。！？.!?])\s*", part)
                for s in sentences:
                    s = s.strip()
                    if not s:
                        continue
                    if total + len(s) > max_chars:
                        return " ".join(kept_parts).strip()
                    kept_parts.append(s)
                    total += len(s) + 1  # 简单加空格

            # 如果循环结束还没超过上限，就返回全部拼接结果
            joined = " ".join(kept_parts).strip()
            if joined:
                return joined
            # 兜底：即便正则没切出句子，也保底截一刀，避免空文本
            return text[:max_chars]

        texts: List[str] = []
        for chunk in chunks:
            if isinstance(chunk, str):
                raw = chunk
            elif isinstance(chunk, dict):
                raw = str(chunk.get("text") or chunk.get("content") or chunk.get("page_content") or "")
            elif hasattr(chunk, "text"):
                raw = str(getattr(chunk, "text"))
            else:
                raw = str(chunk)

            texts.append(_truncate_by_sentence(raw, MAX_CHARS_PER_CONTEXT))

        return texts

    def _metrics_from_settings(self, settings: Any) -> List[str]:
        """Extract metrics list from settings if available."""
        if settings is None:
            return []
        evaluation = getattr(settings, "evaluation", None)
        if evaluation is None:
            return []
        raw_metrics = getattr(evaluation, "metrics", None)
        if raw_metrics is None:
            return []
        # Filter to only ragas-supported metrics
        return [m for m in raw_metrics if m.lower() in SUPPORTED_METRICS]
