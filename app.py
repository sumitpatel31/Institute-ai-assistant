"""
app.py — Streamlit UI for the NareshIT RAG Assistant.

Features:
  - Dark theme, responsive layout
  - Chat interface with conversation history
  - Sidebar with admin controls
  - PDF upload for admin
  - Refresh Website Data button (re-scrapes & rebuilds index)
  - Clear Chat button
  - Show Sources toggle
  - Institute branding / logo
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

# Ensure project root is on sys.path so all imports work when
# Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chatbot import Chatbot, ChatMessage, get_chatbot
from config import settings
from rag_pipeline import get_pipeline
from utils import logger

# =====================================================================
# Page configuration — MUST be the first Streamlit command
# =====================================================================
st.set_page_config(
    page_title="NareshIT AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# Custom CSS — Dark theme + responsive tweaks
# =====================================================================
CUSTOM_CSS = """
<style>
    /* ---- Global dark overrides ---- */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }

    /* ---- Chat bubbles ---- */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }

    /* ---- User message ---- */
    [data-testid="stChatMessage"][data-testid*="user"] {
        background-color: #1e3a5f;
    }

    /* ---- Assistant message ---- */
    [data-testid="stChatMessage"][data-testid*="assistant"] {
        background-color: #1a1a2e;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background-color: #161b22;
    }

    /* ---- Input box ---- */
    [data-testid="stChatInput"] {
        background-color: #1a1a2e;
        border-color: #30363d;
    }

    /* ---- Scrollbar ---- */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0e1117;
    }
    ::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 4px;
    }

    /* ---- Source badges ---- */
    .source-badge {
        display: inline-block;
        background-color: #238636;
        color: #ffffff;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        margin: 2px 4px 2px 0;
    }

    /* ---- Logo / header ---- */
    .logo-header {
        text-align: center;
        padding: 16px 0 8px 0;
    }
    .logo-header h1 {
        color: #58a6ff;
        font-size: 28px;
        margin-bottom: 4px;
    }
    .logo-header p {
        color: #8b949e;
        font-size: 14px;
    }

    /* ---- Status indicator ---- */
    .status-ready { color: #3fb950; }
    .status-loading { color: #d29922; }
    .status-error { color: #f85149; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =====================================================================
# Session state initialisation
# =====================================================================
if "chatbot" not in st.session_state:
    st.session_state.chatbot: Chatbot | None = None
if "pipeline_initialised" not in st.session_state:
    st.session_state.pipeline_initialised = False
if "show_sources" not in st.session_state:
    st.session_state.show_sources = True


# =====================================================================
# Helper functions
# =====================================================================
def render_message(msg: ChatMessage, show_sources: bool) -> None:
    """Render a chat message in the Streamlit chat UI."""
    with st.chat_message(msg.role):
        st.markdown(msg.content)
        if show_sources and msg.sources:
            st.markdown("**Sources:**")
            for src in msg.sources:
                st.markdown(f'<span class="source-badge">{src}</span>', unsafe_allow_html=True)


def initialise_pipeline() -> bool:
    """
    Initialise the RAG pipeline (scrape website + load PDFs + build index).
    Returns True on success, False on failure.
    """
    try:
        pipeline = get_pipeline()
        pipeline.initialize()
        st.session_state.pipeline_initialised = True
        st.session_state.chatbot = get_chatbot()
        return True
    except Exception as exc:
        logger.error("Pipeline initialisation failed: %s", exc)
        st.error(f"Failed to initialise: {exc}")
        return False


def refresh_website_data() -> None:
    """
    Re-scrape the website, clear old index, and rebuild.
    Called when the admin clicks "Refresh Website Data".
    """
    try:
        with st.spinner("Scraping website for latest course schedule …"):
            pipeline = get_pipeline()
            pipeline.reload_website_data()
        st.success("Website data refreshed successfully!")
        st.cache_data.clear()
    except Exception as exc:
        logger.error("Refresh failed: %s", exc)
        st.error(f"Failed to refresh website data: {exc}")


def handle_pdf_upload(uploaded_files) -> None:
    """
    Save uploaded PDFs to the pdfs directory and rebuild the index.
    """
    if not uploaded_files:
        return

    saved_count = 0
    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            save_path = settings.pdfs_dir / file.name
            with open(save_path, "wb") as f:
                f.write(file.read())
            logger.info("Uploaded PDF saved: %s", save_path)
            saved_count += 1

    if saved_count > 0:
        st.success(f"{saved_count} PDF(s) uploaded. Rebuilding index …")
        try:
            pipeline = get_pipeline()
            pipeline.initialize()
            st.success("Index rebuilt with new PDFs!")
        except Exception as exc:
            st.error(f"Failed to rebuild index: {exc}")


# =====================================================================
# Sidebar
# =====================================================================
def render_sidebar() -> None:
    """Render the sidebar with branding, controls, and info."""
    with st.sidebar:
        # --- Branding ---
        st.markdown(
            '<div class="logo-header">'
            '<h1>🎓 NareshIT</h1>'
            '<p>AI-Powered Student Assistant</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        # --- Show Sources Toggle ---
        st.session_state.show_sources = st.toggle(
            "Show Sources",
            value=st.session_state.show_sources,
            help="Display source citations with each answer.",
        )

        st.divider()

        # --- Admin Section ---
        st.subheader("⚙️ Admin Controls")

        # Refresh website data
        if st.button(
            "🔄 Refresh Website Data",
            use_container_width=True,
            help="Re-scrape the website, delete old knowledge base, and rebuild.",
        ):
            refresh_website_data()

        # PDF upload
        st.markdown("**Upload PDFs**")
        uploaded = st.file_uploader(
            "Add institute PDFs (brochures, FAQs, etc.):",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
        )
        if uploaded:
            handle_pdf_upload(uploaded)

        # Clear chat
        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True,
            help="Clear the conversation history.",
        ):
            if st.session_state.chatbot:
                st.session_state.chatbot.clear_history()
            st.rerun()

        st.divider()

        # --- Info ---
        st.subheader("ℹ️ Info")
        st.caption(
            "This assistant answers questions based **only** on "
            "NareshIT's official course schedule (website) and uploaded PDFs. "
            "It does not use outside information."
        )

        # Status
        if st.session_state.pipeline_initialised:
            st.markdown('<span class="status-ready">● Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-loading">● Initialising …</span>', unsafe_allow_html=True)


# =====================================================================
# Main chat interface
# =====================================================================
def render_chat() -> None:
    """Render the main chat area with all messages and input."""
    st.title("💬 Ask NareshIT Assistant")

    # Render conversation history
    chatbot = st.session_state.chatbot
    if chatbot:
        for msg in chatbot.history:
            render_message(msg, st.session_state.show_sources)

    # Chat input
    if prompt := st.chat_input(
        "Ask about courses, schedules, faculty, fees …",
        disabled=not st.session_state.pipeline_initialised,
    ):
        if chatbot is None:
            st.error("Chatbot not initialised. Please refresh the page.")
            return

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking …"):
                start_time = time.time()
                response = chatbot.chat(prompt)
                elapsed = time.time() - start_time

            st.markdown(response.content)
            if st.session_state.show_sources and response.sources:
                st.markdown("**Sources:**")
                for src in response.sources:
                    st.markdown(
                        f'<span class="source-badge">{src}</span>',
                        unsafe_allow_html=True,
                    )
            st.caption(f"Responded in {elapsed:.2f}s")

        st.rerun()


# =====================================================================
# Entry point
# =====================================================================
def main() -> None:
    """Streamlit app entry point."""
    # Render sidebar first (it has branding + controls)
    render_sidebar()

    # Initialise pipeline on first load
    if not st.session_state.pipeline_initialised:
        with st.status("Initialising NareshIT AI Assistant …", expanded=True) as status:
            st.write("📄 Loading PDFs from data directory …")
            time.sleep(0.3)
            st.write("🌐 Scraping course schedule from nareshit.in …")
            time.sleep(0.3)
            st.write("🧠 Building vector database (FAISS) …")
            time.sleep(0.3)
            st.write("🔧 Connecting to Groq LLM …")
            time.sleep(0.3)

            success = initialise_pipeline()

            if success:
                status.update(
                    label="✅ NareshIT AI Assistant is ready!",
                    state="complete",
                    expanded=False,
                )
            else:
                status.update(
                    label="❌ Initialisation failed. Check .env for API key.",
                    state="error",
                    expanded=True,
                )

    # Render the chat interface
    render_chat()


if __name__ == "__main__":
    main()