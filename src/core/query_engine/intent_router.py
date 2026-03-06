"""Intent Router implementing two-layer query routing.

Layer 1: Spam / low-value traffic gateway
Layer 2: Fine-grained business intent classifier

This module loads locally trained models from ``data/models``:
- ``spam_gateway.pkl``: binary classifier (0 = keep, 1 = spam)
- ``intent_classifier.pkl``: multi-class classifier (business intents)

Design goals:
- Local-only inference (no external APIs)
- Graceful degradation when models are missing
- Simple interface for QueryProcessor and higher-level components
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np

from src.core.types import IntentType


logger = logging.getLogger(__name__)


DEFAULT_MODELS_DIR = Path("data/models")
DEFAULT_SPAM_MODEL_NAME = "spam_gateway.pkl"
DEFAULT_INTENT_MODEL_NAME = "intent_classifier.pkl"


@dataclass
class IntentRoutingResult:
    """Result of intent routing for a single query.

    Attributes:
        is_spam: Whether the query should be treated as spam/low-value.
        spam_score: Spam probability in [0, 1] if model is available.
        intent: High-level intent type, if classified.
        intent_label: Raw classifier label (string) if available.
        intent_confidence: Confidence score for predicted intent, if available.
    """

    is_spam: bool = False
    spam_score: Optional[float] = None
    intent: Optional[IntentType] = None
    intent_label: Optional[str] = None
    intent_confidence: Optional[float] = None


class IntentRouter:
    """Two-layer intent router using locally trained models.

    Usage:
        >>> router = IntentRouter()
        >>> result = router.route("这件羊毛大衣可以机洗吗？")
        >>> result.is_spam  # False
        >>> result.intent   # IntentType.FABRIC_CARE
    """

    def __init__(
        self,
        models_dir: Path | None = None,
        spam_model_path: Path | None = None,
        intent_model_path: Path | None = None,
        spam_threshold: float = 0.5,
    ) -> None:
        self.models_dir = models_dir or DEFAULT_MODELS_DIR
        self.spam_threshold = spam_threshold

        self.spam_model_path = spam_model_path or self.models_dir / DEFAULT_SPAM_MODEL_NAME
        self.intent_model_path = intent_model_path or self.models_dir / DEFAULT_INTENT_MODEL_NAME

        self._spam_model = None
        self._intent_model = None

        self._load_models()

    def _load_models(self) -> None:
        """Load models from disk with graceful degradation."""
        # Load spam model
        if self.spam_model_path.exists():
            try:
                self._spam_model = joblib.load(self.spam_model_path)
                logger.info("Spam gateway model loaded from %s", self.spam_model_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to load spam model: %s", exc)
                self._spam_model = None
        else:
            logger.info("Spam model not found at %s (router will skip spam detection).", self.spam_model_path)

        # Load intent model
        if self.intent_model_path.exists():
            try:
                self._intent_model = joblib.load(self.intent_model_path)
                logger.info("Intent classifier model loaded from %s", self.intent_model_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to load intent model: %s", exc)
                self._intent_model = None
        else:
            logger.info(
                "Intent model not found at %s (router will skip intent classification).",
                self.intent_model_path,
            )

    def route(self, query: str) -> IntentRoutingResult:
        """Run two-layer routing for a query.

        Args:
            query: Raw or normalized user query.

        Returns:
            IntentRoutingResult with spam flag and optional intent.
        """
        if not query or not query.strip():
            return IntentRoutingResult()

        # Layer 1: spam gateway
        is_spam = False
        spam_score: Optional[float] = None

        if self._spam_model is not None:
            try:
                proba = self._spam_model.predict_proba([query])[0]
                # Assumes label 1 is "spam"
                if proba.shape[0] == 2:
                    spam_score = float(proba[1])
                else:
                    # Fallback: take probability of the last class
                    spam_score = float(proba[-1])
                is_spam = spam_score >= self.spam_threshold
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Spam model inference failed: %s", exc)

        # If classified as spam, we don't run intent classification
        if is_spam or self._intent_model is None:
            return IntentRoutingResult(
                is_spam=is_spam,
                spam_score=spam_score,
                intent=None,
                intent_label=None,
                intent_confidence=None,
            )

        # Layer 2: fine-grained intent classification
        intent_label: Optional[str] = None
        intent_confidence: Optional[float] = None
        intent_type: Optional[IntentType] = None

        try:
            # XGBoost pipeline expects 2D input; reshape accordingly
            X = np.array([query]).reshape(-1, 1)
            proba = self._intent_model.predict_proba(X)[0]

            best_idx = int(np.argmax(proba))
            intent_confidence = float(proba[best_idx])

            # Recover label name using label_encoder if present
            label_encoder = getattr(self._intent_model, "label_encoder", None)
            if label_encoder is not None:
                intent_label = str(label_encoder.inverse_transform([best_idx])[0])
            else:
                # Fallback: try to use classes_ attribute if available
                classes = getattr(self._intent_model, "classes_", None)
                if classes is not None and 0 <= best_idx < len(classes):
                    intent_label = str(classes[best_idx])

            if intent_label is not None:
                intent_type = self._map_label_to_intent(intent_label)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Intent model inference failed: %s", exc)

        return IntentRoutingResult(
            is_spam=is_spam,
            spam_score=spam_score,
            intent=intent_type,
            intent_label=intent_label,
            intent_confidence=intent_confidence,
        )

    @staticmethod
    def _map_label_to_intent(label: str) -> IntentType:
        """Map raw classifier label string to IntentType enum."""
        normalized = label.strip().lower()
        mapping: Dict[str, IntentType] = {
            "chitchat": IntentType.CHITCHAT,
            "chat": IntentType.CHITCHAT,
            "闲聊": IntentType.CHITCHAT,
            "escalation": IntentType.ESCALATION,
            "complaint": IntentType.ESCALATION,
            "投诉": IntentType.ESCALATION,
            "fabric_care": IntentType.FABRIC_CARE,
            "care": IntentType.FABRIC_CARE,
            "洗护": IntentType.FABRIC_CARE,
            "returns": IntentType.RETURNS,
            "refund": IntentType.RETURNS,
            "退货": IntentType.RETURNS,
            "styling": IntentType.STYLING,
            "styling_advice": IntentType.STYLING,
            "穿搭": IntentType.STYLING,
        }
        return mapping.get(normalized, IntentType.UNKNOWN)


def create_intent_router(
    models_dir: Optional[Path] = None,
    spam_threshold: float = 0.5,
) -> IntentRouter:
    """Factory function to create IntentRouter with default settings."""
    return IntentRouter(
        models_dir=models_dir or DEFAULT_MODELS_DIR,
        spam_threshold=spam_threshold,
    )



