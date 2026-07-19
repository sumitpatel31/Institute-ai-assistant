"""
rag_pipeline.py — Core RAG pipeline orchestrator.

Ties together all components:
  1. Website scraper  →  cleaned text
  2. PDF loader        →  Document list
  3. Vector store      →  FAISS index
  4. Retriever         →  relevant chunks
  5. Groq LLM          →  final answer

Exposes a simple ``ask(question)`` interface for the chatbot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_groq import ChatGroq

from config import settings
from embeddings import get_embedding_manager
from pdf_loader import load_pdfs
from prompt_template import build_user_prompt, get_system_prompt
from retriever import Retriever, RetrievalResult
from utils import logger
from vector_store import get_vector_store
from website_scraper import scrape_website


@dataclass
class PipelineResponse:
    """Structured response from the RAG pipeline."""

    answer: str
    sources: list[str] = field(default_factory=list)
    retrieval_results: list[RetrievalResult] = field(default_factory=list)
    context_used: str = ""


class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Usage::

        pipeline = RAGPipeline()
        pipeline.initialize()          # scrape + load PDFs + build index
        response = pipeline.ask("When does the Python batch start?")
        print(response.answer)
    """

    def __init__(self) -> None:
        self.retriever = Retriever()
        self._llm: ChatGroq | None = None

    # Initialization

    def initialize(self, website_text: str | None = None) -> None:
        """
        Full initialisation pipeline (called at app startup).

        1. Scrape the website (or use provided text).
        2. Load all PDFs.
        3. Build / rebuild the FAISS index.
        4. Warm up the LLM client.
        """
        logger.info("=" * 50)
        logger.info("Initialising RAG pipeline …")
        logger.info("=" * 50)

        # Step 1: Scrape website
        if website_text is None:
            website_text = scrape_website()
        else:
            logger.info("Using provided website text (%d chars).", len(website_text))

        # Step 2: Load PDFs
        pdf_docs = load_pdfs()

        # Step 3: Build / rebuild FAISS index
        vs = get_vector_store()
        vs.build_index(website_text=website_text, pdf_documents=pdf_docs)

        # Step 4: Warm up LLM client
        _ = self._get_llm()

        logger.info("RAG pipeline initialised successfully.")
        logger.info("=" * 50)

    def reload_website_data(self) -> None:
        """
        Re-scrape the website and rebuild the vector store.
        Called when the user clicks "Refresh Website Data".
        Old index is deleted and a fresh one is created.
        """
        logger.info("Reloading website data …")
        vs = get_vector_store()
        vs.clear_index()

        # Re-scrape
        website_text = scrape_website()

        # Re-build with new website data + existing PDFs
        pdf_docs = load_pdfs()
        vs.build_index(website_text=website_text, pdf_documents=pdf_docs)

        logger.info("Website data reloaded and index rebuilt.")


    # Query
  
    def ask(self, question: str) -> PipelineResponse:
        """
        Full RAG flow: retrieve → prompt → generate → return.

        Args:
            question: The student's natural-language question.

        Returns:
            PipelineResponse with answer, sources, and raw retrieval results.
        """
        # 1. Retrieve relevant documents
        results = self.retriever.retrieve(question)
        context = self.retriever.get_context_string(results)
        sources = self.retriever.get_source_list(results)

        if not results:
            logger.info("No retrieval results for: %s", question[:80])
            return PipelineResponse(
                answer=(
                    "I couldn't find this information in the coaching "
                    "institute's documents or website."
                ),
                sources=[],
            )

        # 2. Build prompts
        system = get_system_prompt()
        user = build_user_prompt(question, context)

        # 3. Call Groq LLM
        answer = self._call_llm(system, user)

        return PipelineResponse(
            answer=answer,
            sources=sources,
            retrieval_results=results,
            context_used=context,
        )

    # LLM
  
    def _get_llm(self) -> ChatGroq:
        """Lazy-init the Groq Chat model."""
        if self._llm is None:
            if not settings.groq_api_key or settings.groq_api_key.startswith("gsk_your_"):
                logger.error(
                    "Groq API key not configured. Set GROQ_API_KEY in .env"
                )
                raise ValueError(
                    "Groq API key not configured. "
                    "Please set GROQ_API_KEY in your .env file."
                )
            self._llm = ChatGroq(
                model=settings.groq_model,
                temperature=settings.llm_temperature,
                api_key=settings.groq_api_key,
                max_tokens=1024,
            )
            logger.info("Groq LLM initialised: %s", settings.groq_model)
        return self._llm

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Send messages to the Groq LLM and return the response text."""
        try:
            llm = self._get_llm()
            response = llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            answer = response.content.strip()
            logger.debug("LLM response (%d chars): %s", len(answer), answer[:100])
            return answer
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return (
                "An error occurred while generating the response. "
                "Please try again later."
            )

# Module-level singleton

_rag_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Return the singleton RAGPipeline instance."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.initialize()
    print("\nAsk a question (or 'quit'):")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        resp = pipeline.ask(q)
        print(f"\nAssistant: {resp.answer}")
        if resp.sources:
            print(f"Sources: {', '.join(resp.sources)}")
