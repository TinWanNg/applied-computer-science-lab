from pypdf import PdfReader
import re

# --- Step 1: Dataset chosen ---
pdf_path = "penguins.pdf"
reader = PdfReader(pdf_path)

# --- Step 2: Split text into chunks ---

## extract text from PDF
raw_text = ""
for page in reader.pages:
    raw_text += page.extract_text() + " "

## clean text
def clean_text(text):
    text = re.sub(r"-\s*\n", "", text)  # remove hyphen-line breaks
    text = re.sub(r"\s+", " ", text)    # collapse whitespace
    text = re.sub(r"Page\s+\d+", "", text)  # remove page labels
    return text.strip()

cleaned_text = clean_text(raw_text)

## chunk text
def chunk_text(text, chunk_size=300, chunk_overlap=50):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - chunk_overlap

    return chunks

chunks = chunk_text(cleaned_text, chunk_size=300, chunk_overlap=50)

print("Number of chunks:", len(chunks))
print("Example chunk:\n", chunks[0][:400])

# --- Step 3: Generate embeddings ---
# --- Step 4: Store embeddings in vector db ---
# --- Step 5: Similarity search ---
# --- Step 6: Retrieve top k chunks for query ---