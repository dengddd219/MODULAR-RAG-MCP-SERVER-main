"""Graph Retriever (mock/stub) for graph-augmented retrieval.

This module provides a contract-first GraphRetriever implementation that can be
used before a real graph database (Neo4j/Nebula/etc.) is available.
It exposes the same retrieval contract as other retrievers:
    retrieve(query, top_k, trace) -> List[RetrievalResult]

Design goals:
- Keep interface stable for future DB-backed implementation.
- Return deterministic results for dashboard/evaluation benchmarking.
- Preserve observability fields in metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.core.types import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _GraphNode:
    """In-memory graph node used by the stub retriever."""

    chunk_id: str
    text: str
    source_path: str
    keywords: List[str]
    relation_hints: List[str]


class GraphRetriever:
    """Graph retriever with a deterministic in-memory stub corpus.

    The current implementation is intentionally lightweight and does not require
    any external graph database. It computes a simple lexical overlap score over
    graph-oriented keywords and returns standardized `RetrievalResult` objects.
    """

    def __init__(self, settings: Optional[Any] = None, default_top_k: int = 10) -> None:
        self.settings = settings
        retrieval_cfg = getattr(settings, "retrieval", None) if settings is not None else None
        self.default_top_k = int(
            getattr(retrieval_cfg, "graph_top_k", default_top_k)
        )
        self._nodes = self._build_stub_nodes()
        logger.info(
            "GraphRetriever initialized (stub mode): nodes=%d, default_top_k=%d",
            len(self._nodes),
            self.default_top_k,
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """Retrieve graph-enhanced results for a query.

        Args:
            query: User query text.
            top_k: Optional max result count.
            trace: Optional TraceContext for observability.

        Returns:
            List[RetrievalResult] sorted by mock graph relevance.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        effective_top_k = top_k if top_k is not None else self.default_top_k
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored: List[RetrievalResult] = []
        for node in self._nodes:
            overlap = len(set(query_tokens) & set(node.keywords))
            relation_overlap = len(set(query_tokens) & set(node.relation_hints))
            if overlap == 0 and relation_overlap == 0:
                continue

            # Score in [0, 1] range with relation signal boost.
            score = min(1.0, (overlap + 0.5 * relation_overlap) / max(1, len(node.keywords)))
            scored.append(
                RetrievalResult(
                    chunk_id=node.chunk_id,
                    score=float(score),
                    text=node.text,
                    metadata={
                        "source_path": node.source_path,
                        "retrieval_route": "graph",
                        "graph_stub": True,
                        "graph_overlap_keywords": overlap,
                        "graph_overlap_relations": relation_overlap,
                    },
                )
            )

        scored.sort(key=lambda r: (-r.score, r.chunk_id))
        results = scored[: max(0, int(effective_top_k))]

        if trace is not None:
            try:
                trace.record_stage(
                    "graph_retrieval",
                    {
                        "method": "graph_stub",
                        "top_k": effective_top_k,
                        "result_count": len(results),
                        "chunk_ids": [r.chunk_id for r in results],
                    },
                )
            except Exception:
                # Tracing must not affect retrieval path.
                logger.debug("Graph trace recording failed", exc_info=True)

        return results

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [tok.strip().lower() for tok in text.replace("\n", " ").split() if tok.strip()]

    @staticmethod
    def _build_stub_nodes() -> List[_GraphNode]:
        """Build deterministic graph nodes for local contract testing."""
        return [
            _GraphNode(
                chunk_id="graph_chunk_architecture",
                text="GraphRetriever connects entities, documents, and relation paths for multi-hop recall.",
                source_path="graph://architecture/retrieval",
                keywords=["graph", "retriever", "entity", "relation", "recall", "multi-hop"],
                relation_hints=["connect", "path", "edge", "node"],
            ),
            _GraphNode(
                chunk_id="graph_chunk_rrf",
                text="RRF fusion can combine dense, sparse, and graph ranking signals.",
                source_path="graph://architecture/fusion",
                keywords=["rrf", "fusion", "dense", "sparse", "graph", "ranking"],
                relation_hints=["combine", "merge", "strategy"],
            ),
            _GraphNode(
                chunk_id="graph_chunk_eval",
                text="Evaluation compares baseline retrieval against graph-enhanced retrieval strategies.",
                source_path="graph://evaluation/benchmark",
                keywords=["evaluation", "benchmark", "baseline", "graph", "strategy"],
                relation_hints=["compare", "leaderboard", "quality"],
            ),
        ]


def create_graph_retriever(settings: Optional[Any] = None) -> GraphRetriever:
    """Factory helper for GraphRetriever."""
    return GraphRetriever(settings=settings)
