"""Scoring Engine for normalizing and computing composite scores.

This module provides functionality to normalize metrics and compute
composite scores for model comparison in the LLM Arena.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StrategyMetrics:
    """Metrics for a single strategy (model or combination).
    
    Attributes:
        strategy_name: Name of the strategy
        success_rate: Success rate (0-1)
        avg_latency_s: Average latency in seconds
        p95_latency_s: P95 latency in seconds
        avg_tokens_per_query: Average tokens per query
        avg_cost_per_query: Average cost per query in USD
        total_cost: Total cost for all queries in USD
        avg_quality_score: Average Ragas quality score (0-1)
        routing_total_accuracy: Total routing accuracy (0-1) or None for single models
        routing_simple_accuracy: Simple intent routing accuracy (0-1) or None
        routing_complex_accuracy: Complex intent routing accuracy (0-1) or None
    """
    strategy_name: str
    success_rate: float
    avg_latency_s: float
    p95_latency_s: float
    avg_tokens_per_query: float
    avg_cost_per_query: float
    total_cost: float
    avg_quality_score: float
    routing_total_accuracy: Optional[float] = None
    routing_simple_accuracy: Optional[float] = None
    routing_complex_accuracy: Optional[float] = None


class ScoringEngine:
    """Engine for normalizing metrics and computing composite scores.
    
    This class provides functionality to:
    - Normalize positive metrics (higher is better)
    - Normalize negative metrics (lower is better)
    - Compute composite scores with weighted components
    
    Example:
        >>> engine = ScoringEngine()
        >>> metrics = StrategyMetrics(...)
        >>> score = engine.compute_composite_score(metrics, all_metrics=[...])
    """
    
    def __init__(
        self,
        cost_weight: float = 0.33,
        latency_weight: float = 0.33,
        quality_weight: float = 0.33,
        routing_weight: float = 0.0,  # Only for hybrid strategies
    ) -> None:
        """Initialize ScoringEngine.
        
        Args:
            cost_weight: Weight for cost component (default 0.33)
            latency_weight: Weight for latency component (default 0.33)
            quality_weight: Weight for quality component (default 0.33)
            routing_weight: Weight for routing accuracy (default 0.0, only for hybrid)
        """
        self.cost_weight = cost_weight
        self.latency_weight = latency_weight
        self.quality_weight = quality_weight
        self.routing_weight = routing_weight
        
        # Normalize weights to sum to 1.0
        total = cost_weight + latency_weight + quality_weight + routing_weight
        if total > 0:
            self.cost_weight /= total
            self.latency_weight /= total
            self.quality_weight /= total
            self.routing_weight /= total
    
    def normalize_positive(
        self,
        value: float,
        min_val: float,
        max_val: float,
    ) -> float:
        """Normalize a positive metric (higher is better).
        
        Formula: Score = (x - x_min) / (x_max - x_min)
        
        Args:
            value: The value to normalize
            min_val: Minimum value in the range
            max_val: Maximum value in the range
            
        Returns:
            Normalized score in [0, 1]
        """
        if max_val == min_val:
            return 0.5  # Neutral score if all values are the same
        
        normalized = (value - min_val) / (max_val - min_val)
        # Clamp to [0, 1]
        return max(0.0, min(1.0, normalized))
    
    def normalize_negative(
        self,
        value: float,
        min_val: float,
        max_val: float,
    ) -> float:
        """Normalize a negative metric (lower is better).
        
        Formula: Score = (x_max - x) / (x_max - x_min)
        
        Args:
            value: The value to normalize
            min_val: Minimum value in the range
            max_val: Maximum value in the range
            
        Returns:
            Normalized score in [0, 1]
        """
        if max_val == min_val:
            return 0.5  # Neutral score if all values are the same
        
        normalized = (max_val - value) / (max_val - min_val)
        # Clamp to [0, 1]
        return max(0.0, min(1.0, normalized))
    
    def compute_composite_score(
        self,
        metrics: StrategyMetrics,
        all_metrics: List[StrategyMetrics],
    ) -> float:
        """Compute composite score for a strategy.
        
        Args:
            metrics: StrategyMetrics for the current strategy
            all_metrics: List of all StrategyMetrics for normalization
            
        Returns:
            Composite score in [0, 100], guaranteed to be a valid float (not NaN)
        """
        import math
        
        # Helper function to filter out NaN and invalid values
        def is_valid(value: float) -> bool:
            """Check if value is valid (not NaN, not inf, not None)."""
            if value is None:
                return False
            try:
                return not (math.isnan(value) or math.isinf(value))
            except (TypeError, ValueError):
                return False
        
        # Extract ranges for normalization, filtering out NaN and invalid values
        costs = [
            m.avg_cost_per_query 
            for m in all_metrics 
            if is_valid(m.avg_cost_per_query) and m.avg_cost_per_query > 0
        ]
        latencies = [
            m.p95_latency_s 
            for m in all_metrics 
            if is_valid(m.p95_latency_s) and m.p95_latency_s >= 0
        ]
        qualities = [
            m.avg_quality_score 
            for m in all_metrics 
            if is_valid(m.avg_quality_score) and 0 <= m.avg_quality_score <= 1
        ]
        routing_accs = [
            m.routing_total_accuracy
            for m in all_metrics
            if m.routing_total_accuracy is not None 
            and is_valid(m.routing_total_accuracy) 
            and 0 <= m.routing_total_accuracy <= 1
        ]
        
        # Compute normalized scores with NaN protection
        cost_score = 0.5  # Default neutral
        if costs and is_valid(metrics.avg_cost_per_query) and metrics.avg_cost_per_query > 0:
            try:
                min_cost = min(costs)
                max_cost = max(costs)
                if min_cost < max_cost:  # Avoid division by zero
                    cost_score = self.normalize_negative(
                        metrics.avg_cost_per_query,
                        min_cost,
                        max_cost,
                    )
                    # Ensure result is valid
                    if not is_valid(cost_score):
                        cost_score = 0.5
            except (ValueError, TypeError, ZeroDivisionError):
                cost_score = 0.5
        
        latency_score = 0.5  # Default neutral
        if latencies and is_valid(metrics.p95_latency_s) and metrics.p95_latency_s >= 0:
            try:
                min_latency = min(latencies)
                max_latency = max(latencies)
                if min_latency < max_latency:  # Avoid division by zero
                    latency_score = self.normalize_negative(
                        metrics.p95_latency_s,
                        min_latency,
                        max_latency,
                    )
                    # Ensure result is valid
                    if not is_valid(latency_score):
                        latency_score = 0.5
            except (ValueError, TypeError, ZeroDivisionError):
                latency_score = 0.5
        
        quality_score = 0.5  # Default neutral
        if qualities and is_valid(metrics.avg_quality_score) and 0 <= metrics.avg_quality_score <= 1:
            try:
                min_quality = min(qualities)
                max_quality = max(qualities)
                if min_quality < max_quality:  # Avoid division by zero
                    quality_score = self.normalize_positive(
                        metrics.avg_quality_score,
                        min_quality,
                        max_quality,
                    )
                    # Ensure result is valid
                    if not is_valid(quality_score):
                        quality_score = 0.5
            except (ValueError, TypeError, ZeroDivisionError):
                quality_score = 0.5
        
        # Routing accuracy (only for hybrid strategies)
        routing_score = 0.5  # Default neutral
        if (metrics.routing_total_accuracy is not None 
            and routing_accs 
            and is_valid(metrics.routing_total_accuracy)
            and 0 <= metrics.routing_total_accuracy <= 1):
            try:
                min_routing = min(routing_accs)
                max_routing = max(routing_accs)
                if min_routing < max_routing:  # Avoid division by zero
                    routing_score = self.normalize_positive(
                        metrics.routing_total_accuracy,
                        min_routing,
                        max_routing,
                    )
                    # Ensure result is valid
                    if not is_valid(routing_score):
                        routing_score = 0.5
            except (ValueError, TypeError, ZeroDivisionError):
                routing_score = 0.5
        
        # Compute weighted composite score with NaN protection
        try:
            composite = (
                self.cost_weight * cost_score +
                self.latency_weight * latency_score +
                self.quality_weight * quality_score
            )
            
            # Add routing score if applicable
            if metrics.routing_total_accuracy is not None and self.routing_weight > 0:
                composite += self.routing_weight * routing_score
                # Re-normalize to account for routing weight
                composite /= (1.0 - self.routing_weight + self.routing_weight)
            
            # Ensure composite is valid before converting
            if not is_valid(composite):
                composite = 0.5
            
            # Convert to 0-100 scale and clamp to valid range
            result = composite * 100.0
            if not is_valid(result):
                result = 50.0  # Default to 50 if invalid
            
            return max(0.0, min(100.0, result))  # Clamp to [0, 100]
        except (ValueError, TypeError, ZeroDivisionError):
            # Fallback to neutral score if any calculation fails
            return 50.0

