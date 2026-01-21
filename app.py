# app.py
# -------------------------------------------------------------
# Streamlit UI for the RAG system (LangChain + Chroma + Gemini)
#
# Features:
#   - Add web URLs dynamically from the sidebar
#   - Always load local PDFs and images from /local_docs and /local_docs/images
#   - Build a dedicated Chroma DB per URL-set (so new URLs are
#     always included when the question is asked)
#   - Ask arbitrary questions
#   - Compare:
#       * RAG answer (with retrieval)
#       * Baseline LLM answer (no retrieval)
#   - Inspect retrieved chunks with metadata (source, similarity)
# -------------------------------------------------------------

import streamlit as st
import hashlib
from pathlib import Path

from rag_core import (
    build_chroma_vectorstore,
    answer_with_rag,
    answer_without_rag,
    WEB_URLS,  # mutable list used by rag_core loaders
)

# -------------------------------------------------------------
# Cached vectorstore builder
# Each DISTINCT set of URLs gets its own Chroma DB folder:
#   chroma_db/db_<hash_of_urls>
# This guarantees that when URLs change, a fresh DB is built
# from scratch with:
#   - local PDFs
#   - local images
#   - those URLs
# -------------------------------------------------------------
@st.cache_resource
def get_vectorstore(web_urls_tuple):
    """
    Build a Chroma DB for the current set of URLs.
    Streamlit caches this per 'web_urls_tuple', so:
      - Same URLs  -> reuse existing vectorstore
      - New URLs   -> build a new DB in a new folder
    """
    # Sync global WEB_URLS used in rag_core
    WEB_URLS.clear()
    WEB_URLS.extend(web_urls_tuple)

    # Build a stable key string from URLs
    key = "|".join(web_urls_tuple) if web_urls_tuple else "no_urls"
    db_hash = hashlib.md5(key.encode("utf-8")).hexdigest()
    db_dir = Path("chroma_db") / f"db_{db_hash}"

    # Build DB from scratch for this URL set
    return build_chroma_vectorstore(
        force_rebuild=True,
        persist_directory=str(db_dir),
    )


def main():
    # ---------------------------------------------------------
    # Page config and header
    # ---------------------------------------------------------
    st.set_page_config(
        page_title="RAG Demo (LangChain + Chroma + OpenAI)",
        layout="wide",
    )

    st.title("📚 RAG Demo – LangChain + Chroma + OpenAI")
    st.write(
        """
        This app demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline.

        **Data sources:**
        - Local PDFs in `./local_docs/pdfs/`
        - Local text files in `./local_docs/texts/`
        - Local videos in `./local_docs/videos/`
        - Local images in `./local_docs/images/`
        - ~~Optional web pages~~ (currently disabled)

        **Comparison:**
        - 🔎 *RAG Answer*: uses retrieved chunks from your documents
        - 🧠 *Baseline Answer*: uses the LLM alone (no retrieval)
        """
    )

    # ---------------------------------------------------------
    # Sidebar: Settings + dynamic Web URL input
    # ---------------------------------------------------------
    st.sidebar.header("Settings")

    # Top K slider: how many chunks to retrieve
    top_k = st.sidebar.slider(
        "Number of retrieved chunks (Top K)",
        min_value=1,
        max_value=10,
        value=3,
    )

    st.sidebar.subheader("Add Web URLs (Disabled)")

    # # Initialize URL list in session_state
    # if "web_urls" not in st.session_state:
    #     st.session_state.web_urls = []
    #
    # # Input for a new URL
    # new_url = st.sidebar.text_input(
    #     "Enter a web URL to include:",
    #     placeholder="https://en.wikipedia.org/wiki/Vector_database",
    # )
    #
    # # Button to add URL to list
    # if st.sidebar.button("Add URL"):
    #     if new_url.strip():
    #         st.session_state.web_urls.append(new_url.strip())
    #         st.sidebar.success(f"Added URL: {new_url.strip()}")
    #     else:
    #         st.sidebar.warning("Please enter a valid URL.")
    #
    # # Show list of URLs currently used
    # if st.session_state.web_urls:
    #     st.sidebar.write("### URLs to load:")
    #     for url in st.session_state.web_urls:
    #         st.sidebar.write(f"• {url}")
    # else:
    #     st.sidebar.info("No web URLs added yet. Only local PDFs/images will be used.")
    
    # Force empty URL list
    if "web_urls" not in st.session_state:
        st.session_state.web_urls = []
    
    # Input for a new URL
    # new_url = st.sidebar.text_input(
    #     "Enter a web URL to include:",
    #     placeholder="https://en.wikipedia.org/wiki/Vector_database",
    # )

    # Button to add URL to list
    # if st.sidebar.button("Add URL"):
    #     if new_url.strip():
    #         st.session_state.web_urls.append(new_url.strip())
    #         st.sidebar.success(f"Added URL: {new_url.strip()}")
    #     else:
    #         st.sidebar.warning("Please enter a valid URL.")

    # Show list of URLs currently used
    # if st.session_state.web_urls:
    #     st.sidebar.write("### URLs to load:")
    #     for url in st.session_state.web_urls:
    #         st.sidebar.write(f"• {url}")
    # else:
    #     st.sidebar.info("No web URLs added yet. Only local PDFs/images will be used.")

    # ---------------------------------------------------------
    # Main question input (no default question)
    # ---------------------------------------------------------
    question = st.text_input(
        "Enter your question:",
        value="",
        placeholder="Type a question based on your PDFs/images or added URLs...",
    )

    # ---------------------------------------------------------
    # Load vectorstore (cached per set of URLs)
    # ---------------------------------------------------------
    urls_tuple = tuple(st.session_state.web_urls)

    with st.spinner(
        "Building/Loading vector store for current URLs "
        "(first time may take ~20–40 seconds)..."
    ):
        vectorstore = get_vectorstore(urls_tuple)

    # ---------------------------------------------------------
    # "Ask" button triggers retrieval + LLM calls
    # ---------------------------------------------------------
    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question before clicking Ask.")
            return

        # Two-column layout for RAG vs Baseline answers
        col1, col2 = st.columns(2)

        # -----------------------------
        # Column 1: RAG Answer
        # -----------------------------
        with col1:
            st.subheader("🔎 RAG Answer (with document retrieval)")
            st.write(
                """
                This answer is generated by **retrieving relevant chunks**
                from your PDFs, images, and any web URLs you added, and
                then passing them as context to the LLM.
                """
            )

            with st.spinner("Generating RAG answer..."):
                rag_answer, retrieved = answer_with_rag(
                    vectorstore, question, top_k=top_k
                )

            st.success("RAG answer generated:")
            st.write(rag_answer)

        # -----------------------------
        # Column 2: Baseline LLM Answer
        # -----------------------------
        with col2:
            st.subheader("🧠 Baseline LLM Answer (no retrieval)")
            st.write(
                """
                This answer is produced by the LLM **without** using your documents.
                It may be more general and can hallucinate facts.
                """
            )

            with st.spinner("Generating baseline answer..."):
                base_answer = answer_without_rag(question)

            st.info("Baseline answer generated:")
            st.write(base_answer)

        # -----------------------------------------------------
        # Retrieved chunks: show what context RAG actually used
        # -----------------------------------------------------
        st.markdown("---")
        st.subheader("📄 Retrieved Document Chunks (for inspection)")

        st.write(
            """
            Below are the document chunks retrieved from the vector database.
            They show **which sources** the RAG answer is grounded in
            (PDFs, text files, videos, images, or web URLs).
            """
        )

        for i, r in enumerate(retrieved, start=1):
            src = r["metadata"].get("source")
            page = r["metadata"].get("page")
            similarity = r["similarity"]

            label = f"DOC {i} – source: {src}"
            if page is not None:
                label += f", page {page}"
            label += f" — similarity: {similarity:.4f}"

            with st.expander(label):
                st.write(r["chunk"])


# -------------------------------------------------------------
# Run the app
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
