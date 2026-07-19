"""
vector_store.py — FAISS-based vector store with persistent local storage.

Manages creation, persistence, loading, and rebuilding of the FAISS
index.  Website data is always indexed first (higher priority),
followed by PDF documents.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import settings
from embeddings import get_embedding_manager
from utils import ensure_dir, logger


class VectorStoreManager:
    """
    Wraps FAISS vector store with:
      - Persistent save / load to disk
      - Separate tracking of metadata (docstore, index→doc-id mapping)
      - Rebuild capability (scrape website + reload PDFs)
    """

    def __init__(self) -> None:
        self._faiss: FAISS | None = None
        self._embeddings = get_embedding_manager().get_langchain_embeddings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_index(
        self,
        website_text: str,
        pdf_documents: list[Document],
    ) -> None:
        """
        Build (or rebuild) the FAISS index from scratch.

        Args:
            website_text: Cleaned text scraped from the website.
            pdf_documents: List of LangChain Document objects from PDFs.
        """
        logger.info("Building FAISS index from scratch …")

        all_docs: list[Document] = []

        # --- 1. Website documents (higher priority — indexed first) ---
        if website_text and len(website_text.strip()) > 20:
            website_doc = Document(
                page_content=website_text,
                metadata={
                    "source": "Website → Course Schedule",
                    "source_type": "website",
                    "priority": 1,
                },
            )
            all_docs.append(website_doc)
            logger.info("Added website document (%d chars).", len(website_text))

        # --- 2. PDF documents ---
        if pdf_documents:
            all_docs.extend(pdf_documents)
            logger.info("Added %d PDF document(s).", len(pdf_documents))

        if not all_docs:
            logger.error("No documents to index!")
            return

        # --- 3. Create FAISS index ---
        try:
            self._faiss = FAISS.from_documents(
                all_docs,
                self._embeddings,
            )
            self._persist()
            logger.info(
                "FAISS index built with %d document(s).", len(all_docs)
            )
        except Exception as exc:
            logger.error("Failed to build FAISS index: %s", exc)
            raise

    def add_pdf_documents(self, documents: list[Document]) -> None:
        """Incrementally add PDF documents to an existing index."""
        if not documents:
            return

        faiss = self._get_or_load()
        if faiss is None:
            logger.warning("No index exists — building from scratch.")
            self.build_index(website_text="", pdf_documents=documents)
            return

        faiss.add_documents(documents)
        self._faiss = faiss
        self._persist()
        logger.info("Incrementally added %d PDF document(s).", len(documents))

    def similarity_search(
        self, query: str, k: int | None = None
    ) -> list[Document]:
        """
        Retrieve the *k* most similar documents for *query*.
        Returns an empty list if no index exists.
        """
        faiss = self._get_or_load()
        if faiss is None:
            logger.warning("No FAISS index available for search.")
            return []

        k = k or settings.top_k
        try:
            results = faiss.similarity_search(query, k=k)
            logger.debug("Retrieved %d result(s) for query.", len(results))
            return results
        except Exception as exc:
            logger.error("Similarity search failed: %s", exc)
            return []

    def clear_index(self) -> None:
        """Delete the persisted FAISS index and all related files."""
        for path in (
            settings.faiss_index_path,
            settings.faiss_docstore_path,
            settings.faiss_index_to_doc_id_path,
        ):
            if path.exists():
                path.unlink()
                logger.info("Deleted: %s", path)
        self._faiss = None

    def index_exists(self) -> bool:
        """Check whether a persisted FAISS index already exists."""
        return settings.faiss_index_path.exists()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _get_or_load(self) -> FAISS | None:
        """Return the in-memory FAISS instance, loading from disk if needed."""
        if self._faiss is not None:
            return self._faiss

        if not self.index_exists():
            logger.info("No persisted FAISS index found.")
            return None

        try:
            self._faiss = FAISS.load_local(
                str(settings.faiss_index_path),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("Loaded FAISS index from disk.")
            return self._faiss
        except Exception as exc:
            logger.error("Failed to load FAISS index: %s", exc)
            return None

    def _persist(self) -> None:
        """Save the current FAISS index to disk."""
        if self._faiss is None:
            return
        ensure_dir(settings.vector_store_dir)
        self._faiss.save_local(str(settings.faiss_index_path))
        logger.info("FAISS index saved to %s", settings.faiss_index_path)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---
_vector_store_manager: VectorStoreManager | None = None


def get_vector_store() -> VectorStoreManager:
    """Return the singleton VectorStoreManager instance."""
    global _vector_store_manager
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager()
    return _vector_store_manager


if __name__ == "__main__":
    vs = get_vector_store()
    print(f"Index exists: {vs.index_exists()}")