"""
embeddings.py — Free local embedding model wrapper.

Uses HuggingFace sentence-transformers (no API key required).
The model is downloaded once and cached locally by HuggingFace Hub.
"""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from config import settings
from utils import logger


class EmbeddingManager:
    """
    Manages the local HuggingFace embedding model.

    Usage::

        em = EmbeddingManager()
        vectors = em.embed_texts(["Hello world", "RAG is great"])
        em = em.get_langchain_embeddings()  # for FAISS / LangChain
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self._embeddings: HuggingFaceEmbeddings | None = None

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazy-load and cache the LangChain HuggingFaceEmbeddings wrapper."""
        if self._embeddings is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                encode_kwargs={"normalize_embeddings": True},
                show_progress=True,
            )
            logger.info("Embedding model loaded successfully.")
        return self._embeddings

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a list of text strings."""
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Return embedding vector for a single query string."""
        return self.embeddings.embed_query(text)

    def get_langchain_embeddings(self) -> HuggingFaceEmbeddings:
        """Return the underlying LangChain embeddings object (for FAISS etc.)."""
        return self.embeddings


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------
_embedding_manager: EmbeddingManager | None = None


def get_embedding_manager() -> EmbeddingManager:
    """Return the singleton EmbeddingManager instance."""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager


if __name__ == "__main__":
    em = get_embedding_manager()
    vec = em.embed_query("What is the fee for Data Science?")
    print(f"Embedding dimension: {len(vec)}")
    print(f"First 5 values: {vec[:5]}")