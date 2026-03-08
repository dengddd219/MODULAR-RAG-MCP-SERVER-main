"""Query Complexity Classifier for predicting simple vs complex queries.

This module provides a classifier that predicts whether a query is simple or complex,
which is used for routing queries to appropriate models (small model for simple,
large model for complex).

Design goals:
- Local-only inference (no external APIs)
- Binary classification: simple vs complex
- Graceful degradation when model is missing
- Simple interface for routing decisions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path("data/models")
DEFAULT_COMPLEXITY_MODEL_NAME = "query_complexity_classifier.pkl"


@dataclass
class ComplexityPrediction:
    """Result of query complexity prediction.
    
    Attributes:
        is_simple: True if query is predicted as simple, False if complex.
        is_complex: True if query is predicted as complex, False if simple.
        confidence: Confidence score for the prediction in [0, 1].
        probability_simple: Probability that query is simple.
        probability_complex: Probability that query is complex.
    """
    is_simple: bool
    is_complex: bool
    confidence: float
    probability_simple: float
    probability_complex: float


class QueryComplexityClassifier:
    """Classifier for predicting query complexity (simple vs complex).
    
    This classifier loads a locally trained binary classification model
    from ``data/models/query_complexity_classifier.pkl``.
    
    Usage:
        >>> classifier = QueryComplexityClassifier()
        >>> result = classifier.predict("什么是RAG？")
        >>> result.is_simple  # True or False
        >>> result.confidence  # 0.0 - 1.0
    """
    
    def __init__(
        self,
        models_dir: Path | None = None,
        model_path: Path | None = None,
        threshold: float = 0.5,
    ) -> None:
        """Initialize QueryComplexityClassifier.
        
        Args:
            models_dir: Directory containing model files. Defaults to data/models.
            model_path: Direct path to model file. If None, uses models_dir/query_complexity_classifier.pkl.
            threshold: Threshold for binary classification. Defaults to 0.5.
                      If probability_simple >= threshold, query is classified as simple.
        """
        self.models_dir = models_dir or DEFAULT_MODELS_DIR
        self.threshold = threshold
        
        self.model_path = model_path or self.models_dir / DEFAULT_COMPLEXITY_MODEL_NAME
        
        self._model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load complexity classifier model from disk."""
        if not self.model_path.exists():
            logger.info(
                "Query complexity model not found at %s (will use default prediction).",
                self.model_path,
            )
            self._model = None
            return
        
        try:
            self._model = joblib.load(self.model_path)
            logger.info("Query complexity model loaded from %s", self.model_path)
        except Exception as exc:
            logger.error("Failed to load query complexity model: %s", exc)
            self._model = None
    
    def predict(self, query: str) -> ComplexityPrediction:
        """Predict whether a query is simple or complex.
        
        Args:
            query: User query string.
            
        Returns:
            ComplexityPrediction with is_simple, is_complex, confidence, and probabilities.
            If model is not available, returns default prediction (complex with 0.5 confidence).
        """
        if not query or not query.strip():
            # Empty query defaults to complex
            return ComplexityPrediction(
                is_simple=False,
                is_complex=True,
                confidence=0.5,
                probability_simple=0.0,
                probability_complex=1.0,
            )
        
        # If model is not available, use default prediction
        if self._model is None:
            logger.debug("Query complexity model not available, using default (complex)")
            return ComplexityPrediction(
                is_simple=False,
                is_complex=True,
                confidence=0.5,
                probability_simple=0.0,
                probability_complex=1.0,
            )
        
        try:
            # Prepare input (same format as intent classifier)
            X = np.array([query]).reshape(-1, 1)
            
            # Get probabilities
            proba = self._model.predict_proba(X)[0]
            
            # Determine which class is "simple" and which is "complex"
            # The model might have different label encodings, so we need to check
            label_encoder = getattr(self._model, "label_encoder", None)
            classes = getattr(self._model, "classes_", None)
            
            # Try to identify which class index corresponds to "simple"
            simple_idx = None
            complex_idx = None
            
            if label_encoder is not None:
                # Try to find "simple" and "complex" labels
                all_labels = label_encoder.inverse_transform(range(len(proba)))
                for idx, label in enumerate(all_labels):
                    label_str = str(label).lower()
                    if "simple" in label_str or "简单" in label_str:
                        simple_idx = idx
                    elif "complex" in label_str or "复杂" in label_str:
                        complex_idx = idx
            elif classes is not None:
                # Try to find "simple" and "complex" in classes
                for idx, cls in enumerate(classes):
                    cls_str = str(cls).lower()
                    if "simple" in cls_str or "简单" in cls_str:
                        simple_idx = idx
                    elif "complex" in cls_str or "复杂" in cls_str:
                        complex_idx = idx
            
            # If we couldn't identify labels, assume binary classification:
            # - Index 0 = simple (or first class)
            # - Index 1 = complex (or second class)
            if simple_idx is None or complex_idx is None:
                if len(proba) == 2:
                    # Binary classification: assume first is simple, second is complex
                    simple_idx = 0
                    complex_idx = 1
                else:
                    # Multi-class: use highest probability as prediction
                    best_idx = int(np.argmax(proba))
                    # Default: assume lower indices are simpler
                    if best_idx < len(proba) / 2:
                        simple_idx = best_idx
                        complex_idx = len(proba) - 1
                    else:
                        simple_idx = 0
                        complex_idx = best_idx
            
            # Get probabilities
            prob_simple = float(proba[simple_idx])
            prob_complex = float(proba[complex_idx]) if complex_idx < len(proba) else (1.0 - prob_simple)
            
            # Normalize probabilities to sum to 1.0
            total = prob_simple + prob_complex
            if total > 0:
                prob_simple /= total
                prob_complex /= total
            else:
                prob_simple = 0.5
                prob_complex = 0.5
            
            # Determine prediction based on threshold
            is_simple = prob_simple >= self.threshold
            is_complex = not is_simple
            
            # Confidence is the probability of the predicted class
            confidence = prob_simple if is_simple else prob_complex
            
            return ComplexityPrediction(
                is_simple=is_simple,
                is_complex=is_complex,
                confidence=confidence,
                probability_simple=prob_simple,
                probability_complex=prob_complex,
            )
        except Exception as exc:
            logger.error("Query complexity prediction failed: %s", exc)
            # Return default prediction on error
            return ComplexityPrediction(
                is_simple=False,
                is_complex=True,
                confidence=0.5,
                probability_simple=0.0,
                probability_complex=1.0,
            )


def create_query_complexity_classifier(
    models_dir: Optional[Path] = None,
    threshold: float = 0.5,
) -> QueryComplexityClassifier:
    """Factory function to create QueryComplexityClassifier with default settings."""
    return QueryComplexityClassifier(
        models_dir=models_dir or DEFAULT_MODELS_DIR,
        threshold=threshold,
    )

