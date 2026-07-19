"""
retriever.py — Retrieval helper for the RAG pipeline.

Wraps vector-store similarity search and adds post-processing:
  - Source metadata extraction
  - Deduplication
  - Priority ranking (website data first)
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from config import settings
from vector_store import get_vector_store
from utils import logger


@dataclass
class RetrievalResult:
    """A single retrieval result with cleaned context and source info."""

    content: str
    source: str
    source_type: str
    score: float = 0.0

    def __str__(self) -> str:
        return f"[{self.source}] {self.content}"


class Retriever:
    """
    Retrieves relevant context from the FAISS vector store.

    Usage::

        retriever = Retriever()
        results = retriever.retrieve("What is the fee for Data Science?")
        for r in results:
            print(r.source, r.content[:100])
    """

    def __init__(self) -> None:
        self.vs = get_vector_store()

    def retrieve(
        self, query: str, k: int | None = None
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant documents for *query* and post-process them.

        Args:
            query: The user's question.
            k: Number of results to return (defaults to settings.top_k).

        Returns:
            List of RetrievalResult, sorted by source priority
            (website first, then PDFs).
        """
        k = k or settings.top_k
        documents = self.vs.similarity_search(query, k=k)

        if not documents:
            logger.info("No documents retrieved for query: %s", query[:80])
            return []

        results = self._process_results(documents)
        results = self._sort_by_priority(results)
        results = self._deduplicate(results)
        logger.info("Retrieved %d result(s) for query.", len(results))
        return results

    def get_context_string(self, results: list[RetrievalResult]) -> str:
        """Format retrieval results into a single context string for the LLM."""
        if not results:
            return ""

        parts: list[str] = []
        for i, r in enumerate(results, start=1):
            parts.append(f"[{i}] Source: {r.source}\n{r.content}")

        return "\n\n---\n\n".join(parts)

    def get_source_list(self, results: list[RetrievalResult]) -> list[str]:
        """Extract unique source names from results."""
        seen: set[str] = set()
        sources: list[str] = []
        for r in results:
            if r.source not in seen:
                seen.add(r.source)
                sources.append(r.source)
        return sources

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _process_results(
        self, documents: list[Document]
    ) -> list[RetrievalResult]:
        """Convert LangChain Documents to RetrievalResult objects."""
        results: list[RetrievalResult] = []
        for doc in documents:
            source = doc.metadata.get("source", "Unknown")
            source_type = doc.metadata.get("source_type", "unknown")
            results.append(
                RetrievalResult(
                    content=doc.page_content,
                    source=source,
                    source_type=source_type,
                )
            )
        return results

    def _sort_by_priority(
        self, results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """
        Sort results: website sources first, then PDFs.
        Within each group, maintain the original retrieval order.
        """
        def _sort_key(r: RetrievalResult) -> int:
            return 0 if r.source_type == "website" else 1

        return sorted(results, key=_sort_key)

    @staticmethod
    def _deduplicate(
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Remove results with duplicate content (keep first occurrence)."""
        seen: set[str] = set()
        deduped: list[RetrievalResult] = []
        for r in results:
            content_key = r.content.strip()[:200]
            if content_key not in seen:
                seen.add(content_key)
                deduped.append(r)
        return deduped


if __name__ == "__main__":
    ret = Retriever()
    results = ret.retrieve("When does the Python batch start?")
    for r in results:
        print(f"\n--- {r.source} ---")
        print(r.content[:200])