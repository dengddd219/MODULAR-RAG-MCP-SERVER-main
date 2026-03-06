"""Document-level intent classification for ingestion pipeline.

This module classifies each ingested document into a high-level intent
category (e.g. returns / fabric_care / styling / escalation / chitchat),
reusing the same locally trained intent classifier as the query-side
IntentRouter.

Design goals:
- Run as part of the ingestion pipeline (config-free MVP)
- Use existing local model at ``data/models/intent_classifier.pkl``
- Write intent into chunk metadata (``doc_intent``) for downstream filtering
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

import joblib
import numpy as np

from src.core.settings import resolve_path
from src.core.types import Chunk, Document, IntentType
from src.observability.logger import get_logger


logger = get_logger(__name__)

DEFAULT_MODELS_DIR = Path("data/models")
DEFAULT_INTENT_MODEL_NAME = "intent_classifier.pkl"


@dataclass
class DocumentIntentPrediction:
    """Prediction result with confidence and per-class distribution.

    This is useful for UI surfaces (e.g., Streamlit dashboard) that want to
    display the model's suggested label together with probabilities for all
    candidate classes while still allowing user override.
    """

    intent: IntentType
    label: str
    confidence: float
    distribution: Dict[str, float]


class DocumentIntentClassifier:
    """Classify documents into high-level intent categories.

    This classifier is lightweight by design and will gracefully skip
    classification when the local model is missing or fails to load.
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        intent_model_path: Optional[Path] = None,
        max_chunks: int = 5,
    ) -> None:
        self.models_dir = models_dir or DEFAULT_MODELS_DIR
        self.intent_model_path = intent_model_path or self.models_dir / DEFAULT_INTENT_MODEL_NAME
        self.max_chunks = max_chunks

        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load intent classifier model from disk."""
        path = resolve_path(self.intent_model_path)
        if not path.exists():
            logger.info(
                "Document intent model not found at %s (skipping document intent tagging).",
                path,
            )
            self._model = None
            return

        try:
            self._model = joblib.load(path)
            logger.info("Document intent model loaded from %s", path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load document intent model: %s", exc)
            self._model = None

    def classify_document(
        self,
        document: Document,
        chunks: List[Chunk],
    ) -> Optional[IntentType]:
        """Classify a document into a high-level intent (backwards compatible).

        This retains the original public API used by the ingestion pipeline.
        For richer UI use-cases (e.g. showing probabilities), prefer
        :meth:`classify_document_with_scores`.
        """
        prediction = self.classify_document_with_scores(document, chunks)
        return prediction.intent if prediction is not None else None

    def classify_document_with_scores(
        self,
        document: Document,
        chunks: List[Chunk],
    ) -> Optional[DocumentIntentPrediction]:
        """Classify a document and return intent together with probabilities.

        Args:
            document: Loaded Document object from loader.
            chunks: List of refined/enriched chunks for this document.

        Returns:
            DocumentIntentPrediction if classification succeeds, otherwise None.
        """
        if self._model is None:
            return None

        # Build a representative text snippet for the document:
        # 1) document title (if any)
        # 2) first N chunk summaries or texts
        parts: List[str] = []

        title = str(document.metadata.get("title", "")).strip()
        if title:
            parts.append(title)

        for chunk in chunks[: self.max_chunks]:
            summary = str(chunk.metadata.get("summary", "")).strip()
            if summary:
                parts.append(summary)
            else:
                # Fallback to first 200 characters of chunk text
                text = (chunk.text or "").strip()
                if text:
                    parts.append(text[:200])

        if not parts:
            # As ultimate fallback, use the first 500 chars of the document text
            text = (document.text or "").strip()
            if not text:
                return None
            parts.append(text[:500])

        joined = "\n".join(parts)

        try:
            X = np.array([joined]).reshape(-1, 1)
            proba = self._model.predict_proba(X)[0]
            best_idx = int(np.argmax(proba))
            confidence = float(proba[best_idx])

            # Recover raw labels for all classes
            label_encoder = getattr(self._model, "label_encoder", None)
            classes = getattr(self._model, "classes_", None)

            if label_encoder is not None:
                all_labels = [str(x) for x in label_encoder.inverse_transform(range(len(proba)))]
                best_label = all_labels[best_idx]
            elif classes is not None and len(classes) == len(proba):
                all_labels = [str(x) for x in classes]
                best_label = all_labels[best_idx]
            else:
                # As a fallback, we only know the best index and not the string labels
                all_labels = [str(i) for i in range(len(proba))]
                best_label = all_labels[best_idx]

            if not best_label:
                return None

            distribution: Dict[str, float] = {
                str(label): float(p) for label, p in zip(all_labels, proba)
            }

            intent = self._map_label_to_intent(best_label)
            logger.info(
                "Document %s classified as intent=%s (label=%s, confidence=%.4f)",
                document.id,
                intent.value,
                best_label,
                confidence,
            )

            return DocumentIntentPrediction(
                intent=intent,
                label=best_label,
                confidence=confidence,
                distribution=distribution,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Document intent classification failed: %s", exc)
            return None

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


