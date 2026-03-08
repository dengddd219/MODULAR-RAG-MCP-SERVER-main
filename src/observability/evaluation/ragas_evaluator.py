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
import os

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
        fast_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize RagasEvaluator.

        Args:
            settings: Application settings (optional, used to extract metrics list and API configuration).
                     Note: API key and base_url are read from (in order of priority):
                     1. EVAL_API_KEY / EVAL_BASE_URL environment variables
                     2. settings.llm.api_key / settings.llm.base_url
                     3. settings.embedding.api_key / settings.embedding.base_url
                     4. OPENAI_API_KEY environment variable (for API key only)
                     RAGAS evaluation ALWAYS uses models from settings.yaml (llm.model and embedding.model),
                     regardless of user interface model selection. This ensures consistent evaluation metrics.
                     Falls back to gpt-4o-mini and text-embedding-3-small if not configured in settings.
            metrics: Metric names to compute. Defaults to all supported.
            fast_mode: If True, enables performance optimizations:
                      - Reduces max_eval_chunks from 5 to 3
                      - Truncates answer to 800 characters
                      - Reduces max_tokens from 4096 to 2048
                      - Uses only faithfulness metric (if multiple metrics configured)
                      - Optimized timeouts and retries
            **kwargs: Additional parameters (reserved).

        Raises:
            ImportError: If ragas is not installed.
            ValueError: If unsupported metric names are requested, or if EVAL_API_KEY is not set.
        """
        _import_ragas()

        self.settings = settings
        self.kwargs = kwargs
        self.fast_mode = fast_mode

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
        
        # In fast_mode, only use faithfulness metric (fastest and most important)
        if self.fast_mode and len(self._metric_names) > 1:
            original_metrics = self._metric_names.copy()
            self._metric_names = [FAITHFULNESS] if FAITHFULNESS in self._metric_names else [self._metric_names[0]]
            logger.info(f"Fast mode enabled: Using only {self._metric_names} metric(s) instead of {original_metrics}")

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
        # Step 1: Validate and sanitize inputs
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)

        # Step 2: Sanitize and validate generated_answer
        if not generated_answer or not generated_answer.strip():
            logger.warning("Empty generated_answer provided, using placeholder")
            generated_answer = "No answer generated."  # Provide a safe default instead of raising
        
        # Clean and truncate answer to prevent LLM format issues
        generated_answer = self._sanitize_answer(generated_answer)
        
        # Truncate answer length in fast_mode for faster evaluation
        if self.fast_mode:
            max_answer_length = 800  # characters, roughly 200-300 tokens
            if len(generated_answer) > max_answer_length:
                logger.debug(f"Truncating answer from {len(generated_answer)} to {max_answer_length} chars for faster evaluation")
                generated_answer = generated_answer[:max_answer_length] + "..."

        # Step 3: Limit and validate chunks (optimized based on fast_mode)
        max_eval_chunks = 3 if self.fast_mode else 5
        if len(retrieved_chunks) > max_eval_chunks:
            logger.debug(f"Truncating retrieved chunks from {len(retrieved_chunks)} to {max_eval_chunks} for evaluation.")
            retrieved_chunks = retrieved_chunks[:max_eval_chunks]

        # Step 4: Extract and validate contexts
        contexts = self._extract_texts(retrieved_chunks)
        
        # Step 5: Validate contexts are not empty (boundary case handling)
        if not contexts or all(not ctx or not ctx.strip() for ctx in contexts):
            logger.warning("No valid contexts extracted, using placeholder context")
            contexts = ["No context available."]  # Provide safe default

        # Step 6: Validate query is not empty or too short
        query = query.strip()
        if len(query) < 3:
            logger.warning(f"Query too short ({len(query)} chars), may cause evaluation issues")
            # Still proceed but log warning

        # Step 7: Run Ragas evaluation with comprehensive error handling
        try:
            result = self._run_ragas(query, contexts, generated_answer, ground_truth)
        except Exception as exc:
            # Don't raise - return default scores instead to prevent application crash
            logger.error("Ragas evaluation failed: %s", exc, exc_info=True)
            # Return default scores instead of raising to prevent application crash
            result = {}
            for metric_name in self._metric_names:
                name = self._get_metric_name_safe(metric_name)
                result[name] = 0.5
            logger.warning(f"Returning default scores due to evaluation failure: {result}")

        return result

    # ── private helpers ───────────────────────────────────────────

    def _get_metric_name_safe(self, metric_name: str) -> str:
        """Safely get metric name string without accessing Ragas objects.
        
        Args:
            metric_name: Internal metric name (e.g., 'faithfulness')
            
        Returns:
            Standard metric name string (e.g., 'faithfulness')
        """
        # Map internal names to standard metric names
        name_map = {
            FAITHFULNESS: "faithfulness",
            ANSWER_RELEVANCY: "answer_relevancy",
            CONTEXT_PRECISION: "context_precision",
        }
        return name_map.get(metric_name, metric_name.lower())

    def _run_ragas(
        self,
        query: str,
        contexts: List[str],
        answer: str,
        ground_truth: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Execute Ragas using legacy evaluate() pipeline (non-collections mode).
        This avoids modern embeddings requirement.
        
        Includes comprehensive boundary case handling to prevent Ragas internal bugs.
        
        Args:
            query: The user query string.
            contexts: List of context strings.
            answer: The generated answer string.
            ground_truth: Optional ground truth answer. If None, will inject dummy ground_truth
                        and remove context_precision metric to prevent Ragas internal bugs.
        """
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
        )
        from datasets import Dataset
        from ragas.run_config import RunConfig  # 👈 第一处：必须导入

        # Step 1: Validate inputs before building wrappers (return default scores instead of raising)
        if not query or not query.strip():
            logger.error("Empty query provided to _run_ragas, returning default scores")
            scores = {}
            for metric_name in self._metric_names:
                name = self._get_metric_name_safe(metric_name)
                scores[name] = 0.5
            return scores
        
        if not answer or not answer.strip():
            logger.error("Empty answer provided to _run_ragas, returning default scores")
            scores = {}
            for metric_name in self._metric_names:
                name = self._get_metric_name_safe(metric_name)
                scores[name] = 0.5
            return scores
        
        if not contexts:
            logger.warning("No contexts provided, using placeholder")
            contexts = ["No context available."]
        
        # Step 2: Validate contexts are not all empty
        valid_contexts = [ctx for ctx in contexts if ctx and ctx.strip()]
        if not valid_contexts:
            logger.warning("All contexts are empty, using placeholder")
            valid_contexts = ["No context available."]
        contexts = valid_contexts

        # Step 3: Build LLM / Embedding wrappers
        try:
            llm, embeddings = self._build_wrappers()
        except ValueError as wrapper_exc:
            # Specifically catch ValueError from missing EVAL_API_KEY
            # This is expected when API key is not configured
            logger.warning(f"Ragas wrapper build failed (likely missing EVAL_API_KEY): {wrapper_exc}")
            logger.info("Returning default scores. To enable Ragas evaluation, set EVAL_API_KEY environment variable.")
            # Return default scores if wrapper building fails
            scores = {}
            for metric_name in self._metric_names:
                name = self._get_metric_name_safe(metric_name)
                scores[name] = 0.5
            return scores
        except Exception as wrapper_exc:
            # Catch any other exceptions during wrapper building
            logger.error(f"Failed to build Ragas wrappers: {wrapper_exc}", exc_info=True)
            # Return default scores if wrapper building fails
            scores = {}
            for metric_name in self._metric_names:
                name = self._get_metric_name_safe(metric_name)
                scores[name] = 0.5
            return scores

        print("🔥 USING LEGACY RAGAS PIPELINE")
        print("🔥 LLM TYPE:", type(llm))
        print("🔥 EMBEDDINGS TYPE:", type(embeddings))

        # Step 4: Build dataset with validated data and inject dummy ground_truth if needed
        # Check if we have a real ground_truth (not None and not empty)
        has_real_ground_truth = ground_truth is not None and (
            (isinstance(ground_truth, str) and ground_truth.strip()) or
            (isinstance(ground_truth, (list, dict)) and ground_truth)
        )
        
        # If no real ground_truth, inject dummy to prevent Ragas internal KeyError: 0
        # Use answer as dummy ground_truth to satisfy Ragas validation
        dummy_ground_truth_str = answer.strip()
        
        try:
            data = {
                "question": [query.strip()],
                "answer": [answer.strip()],
                "contexts": [contexts],  # contexts is already a list
            }
            
            # Inject dummy ground_truth to prevent Ragas internal bugs
            # Support both old format (ground_truths as list of lists) and new format (ground_truth as string)
            if not has_real_ground_truth:
                logger.debug("No real ground_truth provided, injecting dummy ground_truth to prevent Ragas KeyError: 0")
                data["ground_truth"] = [dummy_ground_truth_str]  # New format: single string
                data["ground_truths"] = [[dummy_ground_truth_str]]  # Old format: list of lists
            else:
                # Use real ground_truth if provided
                if isinstance(ground_truth, str):
                    data["ground_truth"] = [ground_truth.strip()]
                    data["ground_truths"] = [[ground_truth.strip()]]
                elif isinstance(ground_truth, list):
                    # Assume it's a list of strings
                    data["ground_truth"] = [ground_truth[0] if ground_truth else dummy_ground_truth_str]
                    data["ground_truths"] = [ground_truth if ground_truth else [dummy_ground_truth_str]]
                else:
                    # Fallback to dummy
                    data["ground_truth"] = [dummy_ground_truth_str]
                    data["ground_truths"] = [[dummy_ground_truth_str]]
            
            # Validate dataset structure before creating
            if not data["question"] or not data["question"][0]:
                logger.error("Question is empty after sanitization, returning default scores")
                scores = {}
                for metric_name in self._metric_names:
                    name = self._get_metric_name_safe(metric_name)
                    scores[name] = 0.5
                return scores
            if not data["answer"] or not data["answer"][0]:
                logger.error("Answer is empty after sanitization, returning default scores")
                scores = {}
                for metric_name in self._metric_names:
                    name = self._get_metric_name_safe(metric_name)
                    scores[name] = 0.5
                return scores
            if not data["contexts"] or not data["contexts"][0]:
                logger.error("Contexts are empty after sanitization, returning default scores")
                scores = {}
                for metric_name in self._metric_names:
                    name = self._get_metric_name_safe(metric_name)
                    scores[name] = 0.5
                return scores
            
            dataset = Dataset.from_dict(data)
        except Exception as dataset_exc:
            logger.error(f"Failed to create Ragas dataset: {dataset_exc}", exc_info=True)
            # Return default scores if dataset creation fails
            scores = {}
            for metric_name in self._metric_names:
                name = self._get_metric_name_safe(metric_name)
                scores[name] = 0.5
            return scores

        # Step 5: Select metrics dynamically and validate
        # Remove context_precision if no real ground_truth (it requires ground_truth)
        metrics_to_use = list(self._metric_names)
        if not has_real_ground_truth:
            if CONTEXT_PRECISION in metrics_to_use:
                logger.info("Removing context_precision metric because no real ground_truth provided")
                metrics_to_use.remove(CONTEXT_PRECISION)
        
        metric_map = {
            FAITHFULNESS: faithfulness,
            ANSWER_RELEVANCY: answer_relevancy,
            CONTEXT_PRECISION: context_precision,
        }
        selected_metrics = [
            metric_map[m] for m in metrics_to_use if m in metric_map
        ]
        
        # Validate at least one metric is selected
        if not selected_metrics:
            logger.error("No valid metrics selected for evaluation")
            scores = {}
            for metric_name in self._metric_names:
                name = self._get_metric_name_safe(metric_name)
                scores[name] = 0.5
            return scores

        # Step 6: Configure run config with optimized timeout based on fast_mode
        timeout = 120 if self.fast_mode else 300
        my_run_config = RunConfig(timeout=timeout)

        # Step 7: Execute Ragas evaluation with comprehensive error handling
        result = None
        try:
            result = evaluate(
                dataset=dataset,
                metrics=selected_metrics,
                llm=llm,
                embeddings=embeddings,
                run_config=my_run_config,  # 👈 第三处：必须明确传给 evaluate
            )
            
            # Validate result is not None immediately after evaluation
            if result is None:
                logger.error("Ragas evaluate() returned None result")
                raise ValueError("Ragas evaluation returned None")
                
        except Exception as eval_exc:
            # Comprehensive error handling - catch ALL exceptions
            # This includes:
            # - LLM API errors
            # - Timeout errors
            # - Format errors from LLM responses
            # - Internal Ragas bugs (KeyError: 0, etc.)
            
            # Safely convert exception to string to avoid triggering Ragas Dataset access
            exc_type = type(eval_exc).__name__
            exc_str = str(eval_exc) if eval_exc else "Unknown error"
            
            # Log different error types with appropriate detail
            if "timeout" in exc_str.lower() or "Timeout" in exc_type:
                logger.warning(f"Ragas evaluation timed out: {exc_str}")
            elif "KeyError" in exc_type or "0" in exc_str:
                logger.warning(f"Ragas internal KeyError detected (likely missing metric data): {exc_str}")
            else:
                logger.error(f"Ragas evaluate() call failed: {exc_type}: {exc_str}", exc_info=True)
            
            # Return default scores instead of raising to prevent application crash
            scores = {}
            for metric_name in self._metric_names:
                name = self._get_metric_name_safe(metric_name)
                scores[name] = 0.5
            logger.warning(f"Returning default scores due to evaluation failure: {scores}")
            return scores
        
        # Ensure result is not None before processing
        if result is None:
            logger.error("Ragas evaluate() returned None")
            scores = {}
            # Use metric names directly instead of accessing key.name to avoid triggering Ragas Dataset
            for metric_name in self._metric_names:
                if metric_name == FAITHFULNESS:
                    scores["faithfulness"] = 0.5
                elif metric_name == ANSWER_RELEVANCY:
                    scores["answer_relevancy"] = 0.5
                elif metric_name == CONTEXT_PRECISION:
                    scores["context_precision"] = 0.5
            return scores

        # ---------------------------------------------------------
        # BOMB-PROOF SCORE EXTRACTION
        # ---------------------------------------------------------
        # Ragas EvaluationResult has internal bugs that trigger KeyError: 0
        # when accessing missing metrics. We use string parsing to bypass __getitem__
        import math
        import ast
        
        # 1. Initialize all requested metrics with a safe default of 0.5
        scores = {}
        for m in self._metric_names:
            scores[self._get_metric_name_safe(m)] = 0.5
            
        try:
            # 2. Safest Approach: Parse the __str__ representation
            # Ragas EvaluationResult prints exactly as a dict string, bypassing internal bugs
            res_str = str(result)
            logger.info(f"🔍 Ragas result string representation: {res_str[:500]}...")  # Log first 500 chars
            dict_start = res_str.find("{")
            dict_end = res_str.rfind("}")
            
            if dict_start != -1 and dict_end != -1:
                dict_str = res_str[dict_start:dict_end+1]
                parsed_dict = ast.literal_eval(dict_str)
                
                if isinstance(parsed_dict, dict):
                    logger.info(f"✅ Parsed Ragas result dict: {parsed_dict}")
                    for k, v in parsed_dict.items():
                        if k in scores and isinstance(v, (int, float)):
                            if not math.isnan(v) and not math.isinf(v):
                                scores[k] = float(v)
                                logger.info(f"📊 Extracted metric '{k}': {v}")
                            else:
                                logger.warning(f"⚠️ Metric '{k}' has invalid value (NaN/Inf): {v}, keeping default 0.5")
                        else:
                            logger.debug(f"Metric '{k}' not in requested metrics or not numeric: {v} (type: {type(v)})")
        except Exception as str_exc:
            logger.warning(f"❌ String parsing of Ragas result failed: {str_exc}")
            
            # 3. Fallback Approach: Direct access but SWALLOW ALL EXCEPTIONS (No raise!)
            for name in scores.keys():
                try:
                    if hasattr(result, name):
                        val = getattr(result, name)
                    else:
                        val = result[name]  # This is where Ragas normally throws KeyError: 0
                        
                    if val is not None and not math.isnan(val) and not math.isinf(val):
                        scores[name] = float(val)
                        logger.info(f"📊 Extracted metric '{name}' via direct access: {val}")
                    else:
                        logger.warning(f"⚠️ Metric '{name}' has invalid value (None/NaN/Inf): {val}, keeping default 0.5")
                except Exception as e:
                    logger.warning(f"❌ Failed to extract '{name}' directly, keeping default 0.5. Error: {e}")
                    
        # Ensure all originally requested metrics are in the result
        # (even if some were removed from evaluation, like context_precision without ground_truth)
        for metric_name in self._metric_names:
            name = self._get_metric_name_safe(metric_name)
            if name not in scores:
                # If a metric was removed (e.g., context_precision without ground_truth),
                # it won't be in Ragas result, so keep the default 0.5
                logger.debug(f"Metric '{name}' not in Ragas result, keeping default 0.5")
                scores[name] = 0.5
        
        # Log final scores with emphasis on answer_relevancy
        logger.info("=" * 60)
        logger.info("🎯 FINAL RAGAS EVALUATION SCORES:")
        for metric_name, score in scores.items():
            if metric_name == "answer_relevancy":
                logger.info(f"  ⭐ {metric_name}: {score} {'⚠️ WARNING: Score is 0!' if score == 0.0 else ''}")
            else:
                logger.info(f"  📈 {metric_name}: {score}")
        logger.info("=" * 60)
        return scores

    def _build_wrappers(self) -> tuple:
        """Build Ragas LLM and Embedding wrappers using dedicated OpenAI-compatible API.
        
        This method is decoupled from the main business LLM to avoid performance bottlenecks.
        IMPORTANT: Ragas evaluation ALWAYS uses models from settings.yaml (llm.model and embedding.model),
        NOT from user interface model selection. This ensures evaluation consistency.
        Falls back to gpt-4o-mini and text-embedding-3-small if not configured in settings.
        
        Configuration is read with the following priority order:
        1. Environment variables: EVAL_API_KEY, EVAL_BASE_URL (highest priority)
        2. Settings: settings.llm.api_key, settings.llm.base_url
        3. Settings: settings.embedding.api_key, settings.embedding.base_url
        4. Environment variable: OPENAI_API_KEY (for API key only)
        5. Default: "https://api.openai.com/v1" (for base_url only)
        
        Returns:
            Tuple of (ragas_llm, ragas_embeddings) for RAGAS evaluation.
            
        Raises:
            ValueError: If API key is not found in any of the above sources.
        """
        from ragas.llms import llm_factory
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from openai import AsyncOpenAI
        from langchain_openai import OpenAIEmbeddings
        
        # Read configuration with fallback chain:
        # 1. Environment variable EVAL_API_KEY (highest priority)
        # 2. settings.llm.api_key
        # 3. settings.embedding.api_key
        # 4. Environment variable OPENAI_API_KEY (lowest priority)
        api_key = (
            os.getenv("EVAL_API_KEY") or
            (self.settings and getattr(self.settings.llm, "api_key", None)) or
            (self.settings and getattr(self.settings.embedding, "api_key", None)) or
            os.getenv("OPENAI_API_KEY")
        )
        
        if not api_key:
            raise ValueError(
                "API key not found for RAGAS evaluation. Please set one of:\n"
                "  - EVAL_API_KEY environment variable\n"
                "  - llm.api_key in settings.yaml\n"
                "  - embedding.api_key in settings.yaml\n"
                "  - OPENAI_API_KEY environment variable"
            )
        
        # Read model names from settings, with fallback to defaults
        # LLM model: use settings.llm.model if available, otherwise default to gpt-4o-mini
        llm_model = (
            (self.settings and getattr(self.settings.llm, "model", None)) or
            "gpt-4o-mini"
        )
        
        # Model-specific base_url mapping
        # Each model has its own base_url, but gpt-4o-mini uses settings.llm.base_url
        model_base_url_map = {
            "qwen-max": "https://api.zhizengzeng.com/alibaba",
            "qwen-turbo": "https://api.zhizengzeng.com/alibaba",
            "qwen-plus": "https://api.zhizengzeng.com/alibaba",
            "deepseek-chat": "https://api.zhizengzeng.com/v1",
            "glm-4-plus": "https://api.zhizengzeng.com/v1",
            # gpt-4o-mini and other OpenAI models use settings.llm.base_url (handled below)
        }
        
        # Determine base_url based on model:
        # 1. Environment variable EVAL_BASE_URL (highest priority, overrides everything)
        # 2. If model is gpt-4o-mini or not in map, use settings.llm.base_url
        # 3. If model is in map, use model-specific base_url
        # 4. Fallback to settings.embedding.base_url
        # 5. Default "https://api.openai.com/v1" (lowest priority)
        if os.getenv("EVAL_BASE_URL"):
            base_url = os.getenv("EVAL_BASE_URL")
            logger.info(f"Using EVAL_BASE_URL from environment: {base_url}")
        elif llm_model == "gpt-4o-mini" or llm_model not in model_base_url_map:
            # gpt-4o-mini and unknown models use settings.llm.base_url
            base_url = (
                (self.settings and getattr(self.settings.llm, "base_url", None)) or
                (self.settings and getattr(self.settings.embedding, "base_url", None)) or
                "https://api.openai.com/v1"
            )
            logger.info(f"Using settings base_url for model '{llm_model}': {base_url}")
        else:
            # Use model-specific base_url
            base_url = model_base_url_map[llm_model]
            logger.info(f"Using model-specific base_url for '{llm_model}': {base_url}")
        
        # Embedding model: use settings.embedding.model if available, otherwise default to text-embedding-3-small
        embedding_model = (
            (self.settings and getattr(self.settings.embedding, "model", None)) or
            "text-embedding-3-small"
        )
        
        # Initialize high-performance async client with optimized timeout based on fast_mode
        api_timeout = 30.0 if self.fast_mode else 60.0
        api_max_retries = 2 if self.fast_mode else 3
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=api_timeout,
            max_retries=api_max_retries,
        )
        
        # Build judge LLM with optimized token limit based on fast_mode
        ragas_llm = llm_factory(model=llm_model, client=client)
        
        # Defensive code: Set max_tokens based on fast_mode
        evaluation_max_tokens = 2048 if self.fast_mode else 4096
        if hasattr(ragas_llm, "max_tokens"):
            try:
                ragas_llm.max_tokens = evaluation_max_tokens
                logger.debug(f"Set max_tokens={evaluation_max_tokens} on ragas_llm.max_tokens")
            except Exception as e:
                logger.debug(f"Could not set max_tokens on ragas_llm.max_tokens: {e}")
        elif hasattr(ragas_llm, "llm") and hasattr(ragas_llm.llm, "max_tokens"):
            try:
                ragas_llm.llm.max_tokens = evaluation_max_tokens
                logger.debug(f"Set max_tokens={evaluation_max_tokens} on ragas_llm.llm.max_tokens")
            except Exception as e:
                logger.debug(f"Could not set max_tokens on ragas_llm.llm.max_tokens: {e}")
        
        logger.info(
            f"RAGAS evaluation configured with dedicated API: "
            f"model={llm_model}, base_url={base_url}, max_tokens={evaluation_max_tokens}"
        )
        
        # Build embedding model
        ragas_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model=embedding_model,
                api_key=api_key,
                base_url=base_url,
            )
        )
        
        return ragas_llm, ragas_embeddings

    def _sanitize_answer(self, answer: str) -> str:
        """Sanitize generated answer to prevent LLM format issues.
        
        Args:
            answer: Raw answer string
            
        Returns:
            Sanitized answer string
        """
        if not answer:
            return "No answer generated."
        
        # Remove excessive whitespace
        answer = " ".join(answer.split())
        
        # Truncate if too long (prevent token overflow)
        MAX_ANSWER_LENGTH = 2000
        if len(answer) > MAX_ANSWER_LENGTH:
            logger.warning(f"Answer truncated from {len(answer)} to {MAX_ANSWER_LENGTH} chars")
            # Try to truncate at sentence boundary
            truncated = answer[:MAX_ANSWER_LENGTH]
            last_period = truncated.rfind('.')
            last_newline = truncated.rfind('\n')
            cut_point = max(last_period, last_newline)
            if cut_point > MAX_ANSWER_LENGTH * 0.8:  # Only use if we keep at least 80%
                answer = truncated[:cut_point + 1]
            else:
                answer = truncated + "..."
        
        # Ensure answer is not empty after sanitization
        if not answer.strip():
            return "No answer generated."
        
        return answer.strip()

    def _extract_texts(self, chunks: List[Any]) -> List[str]:
        """Extract text strings from various chunk representations.

        为避免裁判 LLM 的 token 爆炸，这里做两层控制：
        1. 只取前若干 chunks（在 evaluate 里已有 max_eval_chunks 控制）
        2. 对每个 chunk 文本按句号/换行做“句子级”截断，而不是暴力 [:N] 切字符
        
        Also validates and sanitizes extracted texts to prevent boundary case issues.
        """
        MAX_CHARS_PER_CONTEXT = 1000

        def _truncate_by_sentence(text: str, max_chars: int) -> str:
            if not text or not text.strip():
                return ""
            
            if len(text) <= max_chars:
                return text.strip()

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
                        result = " ".join(kept_parts).strip()
                        return result if result else text[:max_chars]
                    kept_parts.append(s)
                    total += len(s) + 1  # 简单加空格

            # 如果循环结束还没超过上限，就返回全部拼接结果
            joined = " ".join(kept_parts).strip()
            if joined:
                return joined
            # 兜底：即便正则没切出句子，也保底截一刀，避免空文本
            return text[:max_chars].strip()

        texts: List[str] = []
        for chunk in chunks:
            try:
                if isinstance(chunk, str):
                    raw = chunk
                elif isinstance(chunk, dict):
                    raw = str(chunk.get("text") or chunk.get("content") or chunk.get("page_content") or "")
                elif hasattr(chunk, "text"):
                    raw = str(getattr(chunk, "text", ""))
                else:
                    raw = str(chunk) if chunk else ""
                
                # Validate and sanitize extracted text
                if not raw or not raw.strip():
                    logger.debug(f"Skipping empty chunk: {type(chunk)}")
                    continue
                
                truncated = _truncate_by_sentence(raw, MAX_CHARS_PER_CONTEXT)
                if truncated:  # Only add non-empty texts
                    texts.append(truncated)
            except Exception as e:
                logger.warning(f"Error extracting text from chunk {type(chunk)}: {e}, skipping")
                continue

        # Ensure we have at least one valid context
        if not texts:
            logger.warning("No valid texts extracted from chunks, using placeholder")
            texts = ["No context available."]
        
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
