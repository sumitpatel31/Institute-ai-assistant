

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

from config import settings
from text_cleaner import TextCleaner
from utils import list_files, logger


class PDFLoader:
    """Extract text from all PDFs in the configured pdfs directory."""

    def __init__(self, pdfs_dir: Path | None = None) -> None:
        self.pdfs_dir = pdfs_dir or settings.pdfs_dir
        self.cleaner = TextCleaner()

    def load_all(self) -> list[Document]:
        """
        Load every PDF in the pdfs directory.
        Returns a list of LangChain ``Document`` objects with metadata.
        """
        pdf_files = list_files(self.pdfs_dir, "*.pdf")

        if not pdf_files:
            logger.warning("No PDF files found in %s", self.pdfs_dir)
            return []

        logger.info("Found %d PDF file(s) in %s", len(pdf_files), self.pdfs_dir)
        documents: list[Document] = []

        for pdf_path in pdf_files:
            docs = self._load_single(pdf_path)
            documents.extend(docs)
            logger.info(
                "  %s → %d document chunk(s)", pdf_path.name, len(docs)
            )

        logger.info(
            "PDF loading complete — %d total document(s).", len(documents)
        )
        return documents

    def load_single(self, path: Path) -> list[Document]:
        """Load a single PDF file by path."""
        return self._load_single(Path(path))

  
    # Internals

    def _load_single(self, pdf_path: Path) -> list[Document]:
        """Extract text from a single PDF and return Document objects."""
        try:
            reader = PdfReader(str(pdf_path))
        except Exception as exc:
            logger.error("Failed to read PDF %s: %s", pdf_path, exc)
            return []

        documents: list[Document] = []
        total_pages = len(reader.pages)

        for page_num, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text()
            if not raw_text:
                continue

            # Clean the extracted text
            cleaned = self.cleaner.clean_text(raw_text)
            if not cleaned or len(cleaned.strip()) < 10:
                continue

            documents.append(
                Document(
                    page_content=cleaned,
                    metadata={
                        "source": pdf_path.name,
                        "source_type": "pdf",
                        "page": page_num,
                        "total_pages": total_pages,
                    },
                )
            )

        return documents


# Convenience

def load_pdfs() -> list[Document]:
    """One-call helper used by the RAG pipeline on startup."""
    loader = PDFLoader()
    return loader.load_all()


if __name__ == "__main__":
    docs = load_pdfs()
    print(f"\nLoaded {len(docs)} document pages.")
    for doc in docs[:3]:
        print(f"\n--- {doc.metadata['source']} (p.{doc.metadata['page']}) ---")
        print(doc.page_content[:300])
