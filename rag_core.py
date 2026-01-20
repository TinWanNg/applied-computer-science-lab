import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import whisper

# For fallback web scraping
import requests
from bs4 import BeautifulSoup

# Environment / API
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

# LangChain loaders and utilities
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredURLLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings + Vector Store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Document object   11
from langchain_core.documents import Document

# ------------------------------------------------------------
# File paths and folder structure
# ------------------------------------------------------------
BASE_DIR = Path(".")
DOCS_DIR = BASE_DIR / "local_docs"      # Folder for PDFs
IMAGES_DIR = DOCS_DIR / "images"        # Folder for images
CHROMA_PATH = BASE_DIR / "chroma_db"    # Default vector DB base folder

DOCS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
CHROMA_PATH.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Gemini API Key loading
# ------------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Set GEMINI_API_KEY in your environment or .env file.")

# Configure Gemini client (text + vision)
genai.configure(api_key=GEMINI_API_KEY)
gemini_text = genai.GenerativeModel("gemini-2.5-flash")
gemini_vision = gemini_text  # same model can handle image input

# ------------------------------------------------------------
# Embeddings model (HuggingFace)
# ------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# ------------------------------------------------------------
# Web URLs list (mutated by Streamlit app)
# ------------------------------------------------------------
WEB_URLS: List[str] = [
    # Will be filled dynamically from app.py
]

# ------------------------------------------------------------
# Helper: basic text cleaning
# ------------------------------------------------------------
def clean_text(text: str) -> str:
    """Remove extra whitespace, page labels, and broken hyphenation."""
    text = re.sub(r"-\s*\n", "", text)     # remove hyphen line-breaks
    text = re.sub(r"\s+", " ", text)       # collapse whitespace
    text = re.sub(r"Page\s+\d+", "", text) # remove "Page X" footer labels
    return text.strip()

# ------------------------------------------------------------
# Helper: sanitize metadata for Chroma
# ------------------------------------------------------------
def sanitize_metadata(md: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Chroma only accepts metadata values that are:
      - str, int, float, bool, None, or SparseVector

    This function converts any list/dict/other objects to strings,
    so we don't get errors like:
      "Expected metadata value to be a str, int, float, bool, SparseVector, or None, got ['eng']"
    """
    if md is None:
        return {}

    simple: Dict[str, Any] = {}
    for k, v in md.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            simple[k] = v
        else:
            # Convert lists, dicts, etc. to string representation
            simple[k] = str(v)
    return simple

# ------------------------------------------------------------
# PDF Loading (LangChain)
# ------------------------------------------------------------
# --- BLOQUE DE FUNCIONES DE CARGA (PEGAR ANTES DE build_chroma_vectorstore) ---

def load_pdf_docs_with_langchain() -> List[Document]:
    """Carga todos los archivos .pdf en local_docs/ usando PyPDFLoader."""
    docs: List[Document] = []
    pdf_paths = list(DOCS_DIR.glob("*.pdf"))
    
    for pdf_path in pdf_paths:
        try:
            loader = PyPDFLoader(str(pdf_path))
            loaded_docs = loader.load()
            for d in loaded_docs:
                d.page_content = clean_text(d.page_content)
                docs.append(d)
            print(f"Cargado PDF: {pdf_path.name}")
        except Exception as e:
            print(f"Error cargando PDF {pdf_path}: {e}")
    return docs

def load_txt_docs_with_langchain() -> List[Document]:
    """Carga todos los archivos .txt en local_docs/ como objetos Document."""
    docs: List[Document] = []
    txt_paths = list(DOCS_DIR.glob("*.txt"))

    for txt_path in txt_paths:
        try:
            # Importante: encoding='utf-8' para evitar errores con tildes
            loader = TextLoader(str(txt_path), encoding="utf-8")
            loaded_docs = loader.load()
            for d in loaded_docs:
                d.page_content = clean_text(d.page_content)
                docs.append(d)
            print(f"Cargado TXT: {txt_path.name}")
        except Exception as e:
            print(f"Error cargando TXT {txt_path}: {e}")
    return docs

def load_video_docs_with_whisper() -> List[Document]:
    """
    Busca archivos .mp4, los transcribe con Whisper y carga el texto resultante.
    Si ya existe la transcripción (.txt), la usa para ahorrar tiempo.
    """
    docs: List[Document] = []
    # Buscamos archivos mp4
    video_paths = list(DOCS_DIR.glob("*.mp4"))
    
    if not video_paths:
        return []

    print(f"Detectados {len(video_paths)} videos. Preparando Whisper...")
    
    # Cargamos el modelo 'base' (equilibrio entre velocidad y precisión)
    # Si tienes GPU lo usará, si no, usará CPU (más lento pero funciona)
    model = whisper.load_model("base")

    for video_path in video_paths:
        try:
            # Definimos el nombre del archivo de texto de salida
            txt_output_path = video_path.with_suffix(".txt")
            
            transcript_text = ""

            # 1. Comprobamos si YA existe la transcripción para no repetir el proceso
            if txt_output_path.exists():
                print(f"Usando transcripción existente para: {video_path.name}")
                with open(txt_output_path, "r", encoding="utf-8") as f:
                    transcript_text = f.read()
            
            # 2. Si NO existe, transcribimos
            else:
                print(f"Transcribiendo video (esto puede tardar): {video_path.name}...")
                result = model.transcribe(str(video_path))
                transcript_text = result["text"]
                
                # Guardamos el resultado en un .txt para la próxima vez
                with open(txt_output_path, "w", encoding="utf-8") as f:
                    f.write(transcript_text)
                print(f"Transcripción guardada en: {txt_output_path.name}")

            # 3. Convertimos a objeto Document de LangChain
            if transcript_text:
                # Creamos el documento manualmente
                doc = Document(page_content=clean_text(transcript_text))
                doc.metadata["source"] = video_path.name # Guardamos origen
                docs.append(doc)

        except Exception as e:
            print(f"Error procesando video {video_path.name}: {e}")
            print("NOTA: Si el error menciona 'ffmpeg', necesitas instalarlo en tu sistema.")

    return docs

# ---------------------------------------------------------------------------
# ------------------------------------------------------------
# Web Page Loading (UnstructuredURLLoader + fallback)
# ------------------------------------------------------------
def load_web_docs_with_langchain() -> List[Document]:
    """
    Load web documents.

    Strategy:
      1. Try UnstructuredURLLoader (LangChain) – good for many sites.
      2. If it fails or returns no documents, fall back to:
         requests + BeautifulSoup + manual HTML parsing.
    """
    if not WEB_URLS:
        return []

    docs: List[Document] = []

    # --- Try UnstructuredURLLoader first ---
    try:
        loader = UnstructuredURLLoader(
            urls=WEB_URLS,
            ssl_verify=False,   # avoid some SSL issues
            mode="elements",    # returns clean text segments
        )
        docs = loader.load()

        for d in docs:
            d.page_content = clean_text(d.page_content)

        if docs:
            print(f"Loaded {len(docs)} web docs via UnstructuredURLLoader.")
            return docs
        else:
            print("UnstructuredURLLoader returned no docs. Falling back to requests + BeautifulSoup.")
    except Exception as e:
        print(f"Failed loading URLs with UnstructuredURLLoader: {e}. Falling back to requests + BeautifulSoup.")

    # --- Fallback: manual requests + BeautifulSoup parsing ---
    fallback_docs: List[Document] = []
    for url in WEB_URLS:
        try:
            resp = requests.get(url, timeout=15, verify=True)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")
            # Simple strategy: concatenate all <p> texts
            paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
            raw_text = " ".join(paragraphs)
            cleaned = clean_text(raw_text)

            if cleaned:
                fallback_docs.append(
                    Document(
                        page_content=cleaned,
                        metadata={"source": url},
                    )
                )
                print(f"Loaded web doc via fallback for {url}, chars={len(cleaned)}")
            else:
                print(f"Fallback got empty text for {url}")
        except Exception as e:
            print(f"Fallback loading failed for {url}: {e}")

    return fallback_docs

# ------------------------------------------------------------
# Image Description (Gemini Vision)
# ------------------------------------------------------------
def describe_image(path: Path) -> str:
    """Use Gemini Vision to generate a factual description of an image."""
    img = Image.open(path)
    prompt = (
        "Describe this image in factual detail. "
        "Mention objects, environment, and any visible text."
    )
    resp = gemini_vision.generate_content([prompt, img])
    return resp.text.strip()

def load_image_docs() -> List[Document]:
    """
    Load all images from local_docs/images/, describe them via Gemini Vision,
    and wrap descriptions in LangChain Document objects.
    """
    docs: List[Document] = []
    image_paths: List[Path] = []

    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        image_paths.extend(IMAGES_DIR.glob(ext))

    for img_path in image_paths:
        try:
            desc = describe_image(img_path)
            docs.append(
                Document(
                    page_content=clean_text(desc),
                    metadata={"source": f"image:{img_path.name}"},
                )
            )
        except Exception as e:
            print(f"Failed describing image {img_path}: {e}")

    return docs

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

# ------------------------------------------------------------
# RAG Answer: LLM + retrieved context
# ------------------------------------------------------------
def answer_with_rag(vectorstore: Chroma, question: str, top_k: int = 3):
    """
    Retrieve relevant chunks from the vectorstore and ask Gemini
    to answer only using that context.
    """
    retrieved = search_similar_chunks(vectorstore, question, top_k=top_k)

    context = ""
    for i, r in enumerate(retrieved, start=1):
        src = r["metadata"].get("source")
        page = r["metadata"].get("page")
        page_info = f", page {page}" if page is not None else ""
        context += f"[DOC {i} | source: {src}{page_info}]\n{r['chunk']}\n\n"

    prompt = f"""
You are a helpful assistant. Answer the user's question ONLY using the context below.
If the answer is not in the context, say: "I don't know based on the given documents."

Context:
{context}

Question: {question}

At the end, list which DOC numbers you used in the format: [Citations: DOC 1, DOC 3]
"""
    resp = gemini_text.generate_content(prompt)
    return resp.text.strip(), retrieved

# ------------------------------------------------------------
# Baseline Answer: LLM without retrieval
# ------------------------------------------------------------
def answer_without_rag(question: str):
    """Ask Gemini directly without any retrieval (for comparison)."""
    prompt = f"Answer this question using your own knowledge:\n\n{question}"
    resp = gemini_text.generate_content(prompt)
    return resp.text.strip()