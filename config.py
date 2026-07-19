"""
config.py — Central configuration for the NareshIT RAG Assistant.

Loads environment variables from .env and exposes them as a single
Settings dataclass so every module can import `settings` once.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Resolve project root (two levels up from this file)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Immutable application settings sourced from .env."""

    # --- Groq LLM ---
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    # --- Embedding model (free, local) ---
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- Website ---
    website_url: str = os.getenv(
        "WEBSITE_URL", "https://nareshit.in/course-schedule/"
    )

    # --- Chunking ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # --- Retrieval ---
    top_k: int = int(os.getenv("TOP_K", "5"))

    # --- Paths ---
    data_dir: Path = PROJECT_ROOT / "data"
    pdfs_dir: Path = PROJECT_ROOT / "data" / "pdfs"
    scraped_dir: Path = PROJECT_ROOT / "data" / "scraped"
    vector_store_dir: Path = PROJECT_ROOT / "vector_store"

    # --- Logging ---
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # --- Derived / helpers ---
    @property
    def faiss_index_path(self) -> Path:
        return self.vector_store_dir / "faiss_index"

    @property
    def faiss_docstore_path(self) -> Path:
        return self.vector_store_dir / "docstore.pkl"

    @property
    def faiss_index_to_doc_id_path(self) -> Path:
        return self.vector_store_dir / "index_to_doc_id.pkl"

    @property
    def scraped_json_path(self) -> Path:
        return self.scraped_dir / "course_schedule.json"

    @property
    def scraped_text_path(self) -> Path:
        return self.scraped_dir / "course_schedule.txt"


# ---------------------------------------------------------------------------
# Singleton instance used throughout the project
# ---------------------------------------------------------------------------
settings = Settings()

# Ensure required directories exist
for _dir in (settings.pdfs_dir, settings.scraped_dir, settings.vector_store_dir):
    _dir.mkdir(parents=True, exist_ok=True)