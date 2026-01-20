"""
Document loaders for different file types and sources.
Handles PDFs, text files, web pages, images, and videos.
"""
from typing import List
from pathlib import Path
import whisper
import requests
from bs4 import BeautifulSoup
from PIL import Image

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredURLLoader,
    TextLoader,
)
from langchain_core.documents import Document

from config import PDFS_DIR, TEXTS_DIR, VIDEOS_DIR, IMAGES_DIR, WEB_URLS, gemini_vision
from utils import clean_text


# ------------------------------------------------------------
# PDF Loading
# ------------------------------------------------------------
def load_pdf_docs_with_langchain() -> List[Document]:
    """Load all .pdf files from local_docs/pdfs/ using PyPDFLoader."""
    docs: List[Document] = []
    pdf_paths = list(PDFS_DIR.glob("*.pdf"))
    
    for pdf_path in pdf_paths:
        try:
            loader = PyPDFLoader(str(pdf_path))
            loaded_docs = loader.load()
            for d in loaded_docs:
                d.page_content = clean_text(d.page_content)
                docs.append(d)
            print(f"Loaded PDF: {pdf_path.name}")
        except Exception as e:
            print(f"Error loading PDF {pdf_path}: {e}")
    return docs


# ------------------------------------------------------------
# Text File Loading
# ------------------------------------------------------------
def load_txt_docs_with_langchain() -> List[Document]:
    """Load all .txt files from local_docs/texts/ as Document objects."""
    docs: List[Document] = []
    txt_paths = list(TEXTS_DIR.glob("*.txt"))

    for txt_path in txt_paths:
        try:
            # Important: encoding='utf-8' to avoid errors with special characters
            loader = TextLoader(str(txt_path), encoding="utf-8")
            loaded_docs = loader.load()
            for d in loaded_docs:
                d.page_content = clean_text(d.page_content)
                docs.append(d)
            print(f"Loaded TXT: {txt_path.name}")
        except Exception as e:
            print(f"Error loading TXT {txt_path}: {e}")
    return docs


# ------------------------------------------------------------
# Video Loading with Whisper
# ------------------------------------------------------------
def load_video_docs_with_whisper() -> List[Document]:
    """
    Find .gif files, transcribe them with Whisper, and load the resulting text.
    If the transcription (.txt) already exists, it uses it to save time.
    """
    docs: List[Document] = []
    # Search for gif files
    video_paths = list(VIDEOS_DIR.glob("*.gif"))
    
    if not video_paths:
        return []

    print(f"Detected {len(video_paths)} videos. Loading Whisper...")
    
    # Load the 'base' model (balance between speed and accuracy)
    # If you have GPU it will use it, otherwise CPU (slower but works)
    model = whisper.load_model("base")

    for video_path in video_paths:
        try:
            # Define the output text file name
            txt_output_path = video_path.with_suffix(".txt")
            
            transcript_text = ""

            # 1. Check if transcription already exists to avoid repeating the process
            if txt_output_path.exists():
                print(f"Using existing transcription for: {video_path.name}")
                with open(txt_output_path, "r", encoding="utf-8") as f:
                    transcript_text = f.read()
            
            # 2. If it doesn't exist, transcribe
            else:
                print(f"Transcribing video (this may take a while): {video_path.name}...")
                result = model.transcribe(str(video_path))
                transcript_text = result["text"]
                
                # Save the result to a .txt file for next time
                with open(txt_output_path, "w", encoding="utf-8") as f:
                    f.write(transcript_text)
                print(f"Transcription saved to: {txt_output_path.name}")

            # 3. Convert to LangChain Document object
            if transcript_text:
                # Create the document manually
                doc = Document(page_content=clean_text(transcript_text))
                doc.metadata["source"] = video_path.name # Save origin
                docs.append(doc)

        except Exception as e:
            print(f"Error processing video {video_path.name}: {e}")
            print("NOTE: If the error mentions 'ffmpeg', you need to install it on your system.")

    return docs


# ------------------------------------------------------------
# Web Page Loading
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
# Image Description and Loading
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
