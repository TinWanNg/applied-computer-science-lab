"""
Document loaders for different file types and sources.
Handles PDFs, text files, web pages, images, and videos.
"""
from typing import List
from pathlib import Path
import time
import whisper
import requests
from bs4 import BeautifulSoup
from PIL import Image
import cv2

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredURLLoader,
    TextLoader,
)
from langchain_core.documents import Document

from config import PDFS_DIR, TEXTS_DIR, VIDEOS_DIR, IMAGES_DIR, WEB_URLS, openai_client
from utils import clean_text
import base64
import io


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
# Video Loading with Whisper and Vision
# ------------------------------------------------------------
def extract_key_frames(video_path: Path, num_frames: int = 3) -> List[Image.Image]:
    """Extract key frames from a video for visual analysis."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return []
    
    # Extract frames evenly spaced throughout the video
    frame_indices = [int(total_frames * i / (num_frames + 1)) for i in range(1, num_frames + 1)]
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB and then to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
    
    cap.release()
    return frames


def describe_video_frames(video_path: Path, frames: List[Image.Image]) -> str:
    """Use OpenAI Vision to describe video frames."""
    if not frames:
        return ""
    
    prompt = (
        "Describe what's happening in this video based on these key frames. "
        "Include objects, actions, environment, and any visible text. "
        "Provide a cohesive description of the video content."
    )
    
    return describe_visual_content(frames, prompt, video_path.name)


def load_video_docs_with_whisper() -> List[Document]:
    """
    Find .mp4 files and process them:
    1. Extract and describe key frames using Gemini Vision (for visual content)
    2. Transcribe audio using Whisper (if audio exists)
    If the transcription (.txt) already exists, it uses it to save time.
    """
    docs: List[Document] = []
    # Search for mp4 files
    video_paths = list(VIDEOS_DIR.glob("*.mp4"))
    
    if not video_paths:
        return []

    print(f"Detected {len(video_paths)} videos. Processing...")
    
    for video_idx, video_path in enumerate(video_paths, 1):
        print(f"\nProcessing video {video_idx}/{len(video_paths)}: {video_path.name}")
        combined_content = []
        
        # 1. Visual description from frames
        try:
            print(f"  Extracting frames from {video_path.name}...")
            frames = extract_key_frames(video_path, num_frames=3)
            if frames:
                print(f"  Describing visual content...")
                visual_desc = describe_video_frames(video_path, frames)
                if visual_desc and not visual_desc.startswith("["):
                    combined_content.append(f"Visual Description: {visual_desc}")
        except Exception as e:
            print(f"  Warning: Could not extract/describe frames: {e}")
        
        # 2. Audio transcription with Whisper
        try:
            txt_output_path = video_path.with_suffix(".txt")
            transcript_text = ""
            
            # Check if transcription already exists
            if txt_output_path.exists():
                print(f"  Using existing audio transcription...")
                with open(txt_output_path, "r", encoding="utf-8") as f:
                    transcript_text = f.read()
            else:
                # Try to transcribe (requires ffmpeg)
                print(f"  Transcribing audio (if any)...")
                try:
                    model = whisper.load_model("base")
                    result = model.transcribe(str(video_path))
                    transcript_text = result["text"]
                    
                    # Save transcription
                    with open(txt_output_path, "w", encoding="utf-8") as f:
                        f.write(transcript_text)
                    print(f"  Audio transcription saved")
                except Exception as whisper_error:
                    if "ffmpeg" in str(whisper_error):
                        print(f"  Note: ffmpeg not found - skipping audio transcription")
                    else:
                        print(f"  Warning: Audio transcription failed: {whisper_error}")
            
            if transcript_text and transcript_text.strip():
                combined_content.append(f"Audio Transcription: {transcript_text}")
        except Exception as e:
            print(f"  Warning: Could not process audio: {e}")
        
        # 3. Create document if we got any content
        if combined_content:
            full_content = "\n\n".join(combined_content)
            doc = Document(page_content=clean_text(full_content))
            doc.metadata["source"] = video_path.name
            docs.append(doc)
            print(f"  ✓ Video processed successfully")
        else:
            print(f"  ✗ No content extracted from video")

    return docs


# ------------------------------------------------------------
# Web Page Loading (DISABLED)
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
            # Ensure source URL is properly set
            if "source" not in d.metadata or not d.metadata["source"]:
                # Try to match to the correct URL or use first one
                d.metadata["source"] = WEB_URLS[0] if WEB_URLS else "web_url"
            # Add metadata to indicate this is a web source
            d.metadata["type"] = "web"

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
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            resp = requests.get(url, timeout=20, headers=headers, verify=True)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Remove scripts, styles, and other non-content elements
            for element in soup(["script", "style", "noscript", "iframe"]):
                element.decompose()
            
            # Get all visible text
            raw_text = soup.get_text(separator=" ", strip=True)
            cleaned = clean_text(raw_text)

            if cleaned and len(cleaned) > 100:
                fallback_docs.append(
                    Document(
                        page_content=cleaned,
                        metadata={"source": url, "type": "web"},
                    )
                )
                print(f"Loaded web doc via fallback for {url}, chars={len(cleaned)}")
                print(f"  First 500 chars: {cleaned[:500]}...")  # Debug output
            else:
                print(f"Fallback got empty text for {url}")
        except Exception as e:
            print(f"Fallback loading failed for {url}: {e}")

    return fallback_docs

# Stub function to avoid import errors
def load_web_docs_with_langchain() -> List[Document]:
    """Web loading is currently disabled."""
    if WEB_URLS:
        print("⚠️ Web URL loading is disabled. Web URLs will not be loaded.")
    return []


# ------------------------------------------------------------
# Shared Vision Description with OpenAI
# ------------------------------------------------------------
def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string for OpenAI API."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def describe_visual_content(images: List[Image.Image], prompt: str, source_name: str) -> str:
    """
    Use OpenAI Vision to describe one or more images.
    Handles rate limiting and retries automatically.
    
    Args:
        images: List of PIL Image objects (can be single image or multiple frames)
        prompt: The prompt to send to OpenAI
        source_name: Name of the source (for error messages)
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Convert images to base64 for OpenAI API
            image_content = []
            for img in images:
                base64_image = image_to_base64(img)
                image_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })
            
            # Build messages with text prompt and images
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": prompt}] + image_content
            }]
            
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",  # or "gpt-4-vision-preview" for better quality
                messages=messages,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                if attempt < max_retries - 1:
                    wait_time = 10
                    print(f"Rate limit hit. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to describe {source_name} after {max_retries} attempts")
                    return f"[{source_name} - description unavailable due to rate limit]"
            else:
                print(f"Error describing {source_name}: {e}")
                return f"[{source_name} - description failed]"
    
    return ""


# ------------------------------------------------------------
# Image Description and Loading
# ------------------------------------------------------------
def describe_image(path: Path) -> str:
    """Use OpenAI Vision to generate a factual description of an image."""
    img = Image.open(path)
    prompt = (
        "Describe this image in factual detail. "
        "Mention objects, environment, and any visible text."
    )
    return describe_visual_content([img], prompt, path.name)


def load_image_docs() -> List[Document]:
    """
    Load all images from local_docs/images/, describe them via Gemini Vision,
    and wrap descriptions in LangChain Document objects.
    """
    docs: List[Document] = []
    image_paths: List[Path] = []

    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        image_paths.extend(IMAGES_DIR.glob(ext))

    print(f"Found {len(image_paths)} images to process...")
    
    for i, img_path in enumerate(image_paths, 1):
        print(f"Processing image {i}/{len(image_paths)}: {img_path.name}")
        desc = describe_image(img_path)
        docs.append(
            Document(
                page_content=clean_text(desc),
                metadata={"source": f"image:{img_path.name}"},
            )
        )
        # Add delay between requests to avoid rate limits (free tier: 5 req/min)
        if i < len(image_paths):
            time.sleep(13)  # Wait ~13 seconds between images (4-5 per minute

    return docs
