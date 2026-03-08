"""Model Evaluator for tracking LLM performance metrics.

This module provides functionality to track and evaluate LLM performance
including latency, token usage, costs, and quality metrics.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Metrics for a single LLM call.
    
    Attributes:
        model_id: Unique identifier for the model (e.g., "openai-gpt-4o-mini")
        provider: Provider name (e.g., "openai", "ollama")
        model_name: Model name (e.g., "gpt-4o-mini")
        latency_ms: Total latency in milliseconds
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        total_tokens: Total tokens used
        cost_per_1k_input: Cost per 1000 input tokens (USD)
        cost_per_1k_output: Cost per 1000 output tokens (USD)
        timestamp: ISO timestamp of the call
        query_length: Length of the input query
        response_length: Length of the output response
        success: Whether the call succeeded
        error: Error message if failed
    """
    model_id: str
    provider: str
    model_name: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    query_length: int = 0
    response_length: int = 0
    success: bool = True
    error: Optional[str] = None
    
    def calculate_cost(self) -> float:
        """Calculate total cost for this call.
        
        Returns:
            Total cost in USD.
        """
        input_cost = (self.prompt_tokens / 1000.0) * self.cost_per_1k_input
        output_cost = (self.completion_tokens / 1000.0) * self.cost_per_1k_output
        return input_cost + output_cost
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation.
        """
        data = asdict(self)
        data['total_cost'] = self.calculate_cost()
        return data


@dataclass
class ModelStats:
    """Aggregated statistics for a model.
    
    Attributes:
        model_id: Model identifier
        total_calls: Total number of calls
        success_count: Number of successful calls
        failure_count: Number of failed calls
        avg_latency_ms: Average latency in milliseconds
        total_tokens: Total tokens used
        total_cost: Total cost in USD
        avg_tokens_per_call: Average tokens per call
        avg_cost_per_call: Average cost per call
    """
    model_id: str
    total_calls: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_tokens_per_call: float = 0.0
    avg_cost_per_call: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation.
        """
        return asdict(self)


class ModelEvaluator:
    """Evaluator for tracking and analyzing LLM performance metrics.
    
    This class provides functionality to:
    - Track metrics for each LLM call
    - Calculate aggregated statistics
    - Persist metrics to disk
    - Load historical metrics
    
    Example:
        >>> evaluator = ModelEvaluator(metrics_dir="data/metrics")
        >>> with evaluator.track_call("openai-gpt-4o-mini", "openai", "gpt-4o-mini") as metrics:
        ...     response = llm.chat(messages)
        ...     metrics.prompt_tokens = response.usage.get("prompt_tokens", 0)
        ...     metrics.completion_tokens = response.usage.get("completion_tokens", 0)
    """
    
    # Default pricing (per 1000 tokens) - can be overridden
    # Note: Pricing data sources:
    # - OpenAI official pricing: https://openai.com/api/pricing/
    # - DeepSeek pricing: https://www.deepseek.com/pricing (approximate)
    # - Qwen pricing: Based on Alibaba Cloud pricing (approximate)
    # - GLM pricing: Based on Zhipu AI pricing (approximate)
    DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
        "openai": {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            # Models via 智增增 proxy (OpenAI-compatible format)
            "deepseek-chat": {"input": 0.00014, "output": 0.00028},  # DeepSeek pricing
            "qwen-max": {"input": 0.0002, "output": 0.0004},  # Qwen pricing (approximate)
            "qwen-plus": {"input": 0.0001, "output": 0.0002},  # Qwen pricing (approximate)
            "glm-4-plus": {"input": 0.0003, "output": 0.0006},  # GLM pricing (approximate)
            # Fallback for unknown OpenAI-compatible models
            "default": {"input": 0.00015, "output": 0.0006},  # Use gpt-4o-mini pricing as fallback
        },
        "ollama": {
            "default": {"input": 0.0, "output": 0.0},  # Local, no cost
        },
        "deepseek": {
            "default": {"input": 0.00014, "output": 0.00028},
        },
    }
    
    def __init__(
        self,
        metrics_dir: str = "data/metrics",
        pricing: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    ) -> None:
        """Initialize ModelEvaluator.
        
        Args:
            metrics_dir: Directory to store metrics JSON files.
            pricing: Optional pricing override. Format: {provider: {model: {input: float, output: float}}}
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Merge custom pricing with defaults
        self.pricing = pricing or {}
        for provider, models in self.DEFAULT_PRICING.items():
            if provider not in self.pricing:
                self.pricing[provider] = {}
            for model, prices in models.items():
                if model not in self.pricing[provider]:
                    self.pricing[provider][model] = prices.copy()
        
        self._metrics_cache: List[ModelMetrics] = []
        self._cache_file = self.metrics_dir / "metrics_cache.jsonl"
    
    def get_pricing(
        self,
        provider: str,
        model_name: str,
    ) -> tuple[float, float]:
        """Get pricing for a model.
        
        Args:
            provider: Provider name.
            model_name: Model name.
            
        Returns:
            Tuple of (input_cost_per_1k, output_cost_per_1k).
        """
        provider_pricing = self.pricing.get(provider.lower(), {})
        
        # Try exact match first
        if model_name in provider_pricing:
            prices = provider_pricing[model_name]
            return prices.get("input", 0.0), prices.get("output", 0.0)
        
        # Try default
        if "default" in provider_pricing:
            prices = provider_pricing["default"]
            return prices.get("input", 0.0), prices.get("output", 0.0)
        
        # No pricing found
        return 0.0, 0.0
    
    def track_call(
        self,
        model_id: str,
        provider: str,
        model_name: str,
        query: str = "",
    ) -> "MetricsContext":
        """Create a context manager to track a single LLM call.
        
        Args:
            model_id: Unique model identifier.
            provider: Provider name.
            model_name: Model name.
            query: Input query (for length calculation).
            
        Returns:
            MetricsContext manager.
        """
        return MetricsContext(
            evaluator=self,
            model_id=model_id,
            provider=provider,
            model_name=model_name,
            query=query,
        )
    
    def record_metrics(self, metrics: ModelMetrics) -> None:
        """Record metrics to disk.
        
        Args:
            metrics: ModelMetrics instance to record.
        """
        # Add to cache
        self._metrics_cache.append(metrics)
        
        # Append to JSONL file
        try:
            with open(self._cache_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write metrics: {e}")
    
    def load_metrics(
        self,
        model_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ModelMetrics]:
        """Load metrics from disk.
        
        Args:
            model_id: Optional filter by model_id.
            limit: Optional limit on number of records.
            
        Returns:
            List of ModelMetrics.
        """
        if not self._cache_file.exists():
            return []
        
        metrics = []
        try:
            with open(self._cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if model_id is None or data.get("model_id") == model_id:
                        # Reconstruct ModelMetrics
                        metric = ModelMetrics(
                            model_id=data["model_id"],
                            provider=data["provider"],
                            model_name=data["model_name"],
                            latency_ms=data["latency_ms"],
                            prompt_tokens=data.get("prompt_tokens", 0),
                            completion_tokens=data.get("completion_tokens", 0),
                            total_tokens=data.get("total_tokens", 0),
                            cost_per_1k_input=data.get("cost_per_1k_input", 0.0),
                            cost_per_1k_output=data.get("cost_per_1k_output", 0.0),
                            timestamp=data.get("timestamp", ""),
                            query_length=data.get("query_length", 0),
                            response_length=data.get("response_length", 0),
                            success=data.get("success", True),
                            error=data.get("error"),
                        )
                        metrics.append(metric)
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
        
        if limit:
            metrics = metrics[-limit:]  # Most recent N
        
        return metrics
    
    def get_stats(
        self,
        model_id: Optional[str] = None,
    ) -> Dict[str, ModelStats]:
        """Get aggregated statistics for models.
        
        Args:
            model_id: Optional filter by model_id.
            
        Returns:
            Dictionary mapping model_id to ModelStats.
        """
        metrics = self.load_metrics(model_id=model_id)
        
        # Group by model_id
        by_model: Dict[str, List[ModelMetrics]] = {}
        for metric in metrics:
            if metric.model_id not in by_model:
                by_model[metric.model_id] = []
            by_model[metric.model_id].append(metric)
        
        # Calculate stats for each model
        stats_dict: Dict[str, ModelStats] = {}
        for mid, model_metrics in by_model.items():
            if not model_metrics:
                continue
            
            latencies = [m.latency_ms for m in model_metrics if m.success]
            latencies.sort()
            
            total_calls = len(model_metrics)
            success_count = sum(1 for m in model_metrics if m.success)
            failure_count = total_calls - success_count
            
            total_tokens = sum(m.total_tokens for m in model_metrics)
            total_cost = sum(m.calculate_cost() for m in model_metrics)
            
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
            min_latency = latencies[0] if latencies else 0.0
            max_latency = latencies[-1] if latencies else 0.0
            
            # Percentiles
            p50 = latencies[len(latencies) // 2] if latencies else 0.0
            p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
            p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0.0
            
            stats = ModelStats(
                model_id=mid,
                total_calls=total_calls,
                success_count=success_count,
                failure_count=failure_count,
                avg_latency_ms=avg_latency,
                total_tokens=total_tokens,
                total_cost=total_cost,
                avg_tokens_per_call=total_tokens / total_calls if total_calls > 0 else 0.0,
                avg_cost_per_call=total_cost / total_calls if total_calls > 0 else 0.0,
                min_latency_ms=min_latency,
                max_latency_ms=max_latency,
                p50_latency_ms=p50,
                p95_latency_ms=p95,
                p99_latency_ms=p99,
            )
            stats_dict[mid] = stats
        
        return stats_dict


class MetricsContext:
    """Context manager for tracking a single LLM call."""
    
    def __init__(
        self,
        evaluator: ModelEvaluator,
        model_id: str,
        provider: str,
        model_name: str,
        query: str,
    ) -> None:
        """Initialize metrics context.
        
        Args:
            evaluator: ModelEvaluator instance.
            model_id: Model identifier.
            provider: Provider name.
            model_name: Model name.
            query: Input query.
        """
        self.evaluator = evaluator
        self.metrics = ModelMetrics(
            model_id=model_id,
            provider=provider,
            model_name=model_name,
            latency_ms=0.0,
            query_length=len(query),
        )
        
        # Set pricing
        input_cost, output_cost = evaluator.get_pricing(provider, model_name)
        self.metrics.cost_per_1k_input = input_cost
        self.metrics.cost_per_1k_output = output_cost
        
        self._start_time: Optional[float] = None
    
    def __enter__(self) -> ModelMetrics:
        """Enter context and start timing.
        
        Returns:
            ModelMetrics instance to update.
        """
        self._start_time = time.monotonic()
        return self.metrics
    
    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any],
    ) -> None:
        """Exit context and record metrics.
        
        Args:
            exc_type: Exception type if any.
            exc_val: Exception value if any.
            exc_tb: Exception traceback if any.
        """
        if self._start_time:
            elapsed = time.monotonic() - self._start_time
            self.metrics.latency_ms = elapsed * 1000.0
        
        if exc_type is not None:
            self.metrics.success = False
            self.metrics.error = str(exc_val) if exc_val else str(exc_type)
        
        # Record metrics
        self.evaluator.record_metrics(self.metrics)

