"""
Vector store management using Chroma.
Handles building, loading, and searching the vector database.
"""
import shutil
from pathlib import Path
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from config import embeddings, CHROMA_PATH
from document_loaders import (
    load_pdf_docs_with_langchain,
    load_txt_docs_with_langchain,
    load_video_docs_with_whisper,
    load_web_docs_with_langchain,
    load_image_docs,
)
from utils import sanitize_metadata


# ------------------------------------------------------------
# Text Chunking
# ------------------------------------------------------------
def split_docs_with_langchain(docs: List[Document]) -> List[Document]:
    """Split documents into smaller chunks for better retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,        # ~200–300 words
        chunk_overlap=200,      # preserve context
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return text_splitter.split_documents(docs)


# ------------------------------------------------------------
# Build or Load Chroma Vectorstore
# ------------------------------------------------------------
def build_chroma_vectorstore(
    force_rebuild: bool = False,
    persist_directory: Optional[str] = None,
) -> Chroma:
    """
    Build or load a persistent Chroma vector store.

    - persist_directory:
        Path to the DB folder (used to separate DBs per URL-set).
        If None, uses global CHROMA_PATH.
    - If force_rebuild=True:
        Try to delete existing DB at that path and rebuild it.
    - Otherwise:
        Load existing DB, or create new if missing.
    """
    db_path = Path(persist_directory) if persist_directory is not None else CHROMA_PATH

    if force_rebuild and db_path.exists():
        try:
            shutil.rmtree(db_path)
        except PermissionError:
            print(
                f"⚠️ Could not delete existing Chroma DB at {db_path} (file in use). "
                "Reusing the existing DB instead."
            )

    # Build new DB if path does not exist
    if not db_path.exists():
        db_path.mkdir(parents=True, exist_ok=True)

        pdf_docs = load_pdf_docs_with_langchain()
        web_docs = load_web_docs_with_langchain()
        image_docs = load_image_docs()
        txt_docs = load_txt_docs_with_langchain()
        video_docs = load_video_docs_with_whisper()
        all_docs = pdf_docs + web_docs + image_docs + txt_docs + video_docs

        # If there are no documents at all, create an empty DB
        if not all_docs:
            print("⚠️ No documents found. Creating an empty Chroma DB.")
            vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory=str(db_path),
                collection_name="rag_penguins",
            )
            return vectorstore

        # Split and sanitize metadata before inserting into Chroma
        split_docs = split_docs_with_langchain(all_docs)
        sanitized_docs: List[Document] = []
        for d in split_docs:
            d.metadata = sanitize_metadata(d.metadata)
            sanitized_docs.append(d)

        vectorstore = Chroma.from_documents(
            documents=sanitized_docs,
            embedding=embeddings,
            persist_directory=str(db_path),
            collection_name="rag_penguins",
        )
        vectorstore.persist()
        return vectorstore

    # Load existing DB
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=str(db_path),
        collection_name="rag_penguins",
    )
    return vectorstore


# ------------------------------------------------------------
# Retrieval: similarity search
# ------------------------------------------------------------
def search_similar_chunks(vectorstore: Chroma, query: str, top_k: int = 3):
    """Run similarity search and return top-k chunks with metadata."""
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    out = []

    for doc, score in results:
        similarity = 1.0 - score  # convert distance → similarity-ish
        out.append(
            {
                "chunk": doc.page_content,
                "similarity": similarity,
                "metadata": doc.metadata,
            }
        )

    return out
