"""
RAG Core - Main module (backward compatibility wrapper).
All functionality has been split into separate modules:
- config.py: Configuration and model setup
- utils.py: Helper functions
- document_loaders.py: Document loading for all file types
- vectorstore.py: Vector database management
- rag_engine.py: RAG question answering

This module re-exports everything for backward compatibility with app.py.
"""

# Re-export everything for backward compatibility
from config import (
    BASE_DIR,
    DOCS_DIR,
    PDFS_DIR,
    TEXTS_DIR,
    VIDEOS_DIR,
    IMAGES_DIR,
    CHROMA_PATH,
    OPENAI_API_KEY,
    openai_client,
    EMBEDDING_MODEL_NAME,
    embeddings,
    WEB_URLS,
)

from utils import clean_text, sanitize_metadata

from document_loaders import (
    load_pdf_docs_with_langchain,
    load_txt_docs_with_langchain,
    load_video_docs_with_whisper,
    load_web_docs_with_langchain,
    describe_image,
    load_image_docs,
)

from vectorstore import (
    split_docs_with_langchain,
    build_chroma_vectorstore,
    search_similar_chunks,
)

from rag_engine import answer_with_rag, answer_without_rag

# Make all imports available
__all__ = [
    # Config
    "BASE_DIR",
    "DOCS_DIR",
    "PDFS_DIR",
    "TEXTS_DIR",
    "VIDEOS_DIR",
    "IMAGES_DIR",
    "CHROMA_PATH",
    "OPENAI_API_KEY",
    "openai_client",
    "EMBEDDING_MODEL_NAME",
    "embeddings",
    "WEB_URLS",
    # Utils
    "clean_text",
    "sanitize_metadata",
    # Document loaders
    "load_pdf_docs_with_langchain",
    "load_txt_docs_with_langchain",
    "load_video_docs_with_whisper",
    "load_web_docs_with_langchain",
    "describe_image",
    "load_image_docs",
    # Vectorstore
    "split_docs_with_langchain",
    "build_chroma_vectorstore",
    "search_similar_chunks",
    # RAG engine
    "answer_with_rag",
    "answer_without_rag",
]