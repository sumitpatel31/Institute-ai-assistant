# NareshIT AI Assistant — RAG-Powered Student Chatbot

An intelligent AI Assistant for **NareshIT Software Training Institute** built using **Retrieval Augmented Generation (RAG)**. The assistant answers student questions based **only** on the institute's official course schedule (scraped from the website) and uploaded PDF documents.

## Architecture

```
Student Question
       │
       ▼
   Embedding (all-MiniLM-L6-v2, free local model)
       │
       ▼
   FAISS Similarity Search (local, persistent)
       │
       ▼
   Retrieve Relevant Chunks
       │
       ▼
   Build Prompt (System + Context + Question)
       │
       ▼
   Groq LLM (llama-3.3-70b-versatile)
       │
       ▼
   Final Answer with Source Citations
```

## Features

- **Website Scraping**: Automatically scrapes the NareshIT course schedule page (`nareshit.in/course-schedule/`), parses HTML tables with BeautifulSoup, and extracts structured data.
- **PDF Ingestion**: Reads all PDFs from `data/pdfs/`, extracts text, cleans it, and indexes it.
- **Real-Time Updates**: Every time the app starts, it re-scrapes the website. Users can also click "Refresh Website Data" to force a refresh.
- **RAG Pipeline**: Complete retrieval-augmented generation with FAISS vector store and Groq LLM.
- **No Hallucination**: The assistant only answers from retrieved context. If information is not found, it says so explicitly.
- **Source Citations**: Every answer includes the source (e.g., "Website → Course Schedule" or "nareshiT_context2.pdf").
- **Session Memory**: Maintains conversation history during the session.
- **Admin PDF Upload**: Upload new PDFs through the sidebar — they are automatically indexed.
- **Dark Theme Streamlit UI**: Responsive, clean interface with sidebar controls.

## Project Structure

```
nareshit-rag/
├── app.py                  # Streamlit web interface (entry point)
├── chatbot.py              # Chatbot with session memory
├── rag_pipeline.py         # Core RAG pipeline orchestrator
├── pdf_loader.py           # PDF text extraction
├── website_scraper.py      # BeautifulSoup website scraper
├── text_cleaner.py         # Text cleaning utilities
├── embeddings.py           # Free local embedding model (sentence-transformers)
├── vector_store.py         # FAISS vector store with persistence
├── retriever.py            # Retrieval helper with priority sorting
├── prompt_template.py      # System and user prompt templates
├── config.py               # Central configuration (loads .env)
├── utils.py                # Shared utilities (logging, helpers)
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API keys)
├── data/
│   ├── pdfs/               # Place PDF files here
│   │   └── nareshiT_context2.pdf
│   └── scraped/            # Auto-generated scraped data
│       ├── course_schedule.json
│       └── course_schedule.txt
├── vector_store/           # Persistent FAISS index (auto-generated)
│   ├── faiss_index/
│   ├── docstore.pkl
│   └── index_to_doc_id.pkl
└── README.md
```

## Installation

### Prerequisites

- **Python 3.10+**
- **Groq API Key**: Get one free at [https://console.groq.com/keys](https://console.groq.com/keys)

### Steps

1. **Navigate to the project directory:**
   ```bash
   cd nareshit-rag
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   # venv\Scripts\activate    # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your Groq API key:**
   Edit `.env` and replace `gsk_your_groq_api_key_here` with your actual key:
   ```
   GROQ_API_KEY=gsk_abc123your_real_key_here
   ```

5. **Place PDF files** in `data/pdfs/` (the default `nareshiT_context2.pdf` is already included).

6. **Run the application:**
   ```bash
   streamlit run app.py
   ```

   The app will:
   - Scrape the NareshIT course schedule page
   - Load all PDFs from `data/pdfs/`
   - Build the FAISS vector store
   - Open in your browser at `http://localhost:8501`

## Configuration

All configuration is in the `.env` file:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Your Groq API key (required) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | LLM model |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Free local embedding model |
| `WEBSITE_URL` | `https://nareshit.in/course-schedule/` | URL to scrape |
| `CHUNK_SIZE` | `500` | Text chunk size for PDFs |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `5` | Number of documents to retrieve |
| `LLM_TEMPERATURE` | `0.1` | LLM temperature (lower = more factual) |

## Usage Examples

- "What courses are available?"
- "When does the Python batch start?"
- "Who teaches Machine Learning?"
- "What are today's classes?"
- "What is the office timing?"
- "Show all available courses."
- "Which batch is available on weekends?"
- "Who is Mr. Veera Babu?"

## Technologies

- **Python 3.10+**
- **LangChain** — RAG orchestration
- **Groq** — Fast LLM inference
- **FAISS** — Local vector database
- **SentenceTransformers** — Free embeddings (all-MiniLM-L6-v2)
- **BeautifulSoup4** — HTML table parsing
- **Streamlit** — Web UI
- **PyPDF** — PDF text extraction

## Refresh / Reload

When the user clicks **"Refresh Website Data"** in the sidebar:
1. The old FAISS index is deleted
2. The website is re-scraped for the latest schedule
3. PDFs are re-loaded
4. A new FAISS index is built

This ensures the assistant always uses the latest available data.

