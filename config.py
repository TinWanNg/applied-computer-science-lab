"""
Configuration and setup for the RAG system.
Handles paths, API keys, and model initialization.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

# ------------------------------------------------------------
# File paths and folder structure
# ------------------------------------------------------------
BASE_DIR = Path(".")
DOCS_DIR = BASE_DIR / "local_docs"          # Root folder for all documents
PDFS_DIR = DOCS_DIR / "pdfs"                # Folder for PDFs
TEXTS_DIR = DOCS_DIR / "texts"              # Folder for text files
VIDEOS_DIR = DOCS_DIR / "videos"            # Folder for videos
IMAGES_DIR = DOCS_DIR / "images"            # Folder for images
CHROMA_PATH = BASE_DIR / "chroma_db"        # Default vector DB base folder

DOCS_DIR.mkdir(exist_ok=True)
PDFS_DIR.mkdir(exist_ok=True)
TEXTS_DIR.mkdir(exist_ok=True)
VIDEOS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
CHROMA_PATH.mkdir(exist_ok=True)

# ------------------------------------------------------------
# OpenAI API Key loading
# ------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Set OPENAI_API_KEY in your environment or .env file.")

# Configure OpenAI client
from openai import OpenAI
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------------------------------------
# Embeddings model (HuggingFace)
# ------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# ------------------------------------------------------------
# Web URLs list (mutated by Streamlit app)
# ------------------------------------------------------------
WEB_URLS = []
