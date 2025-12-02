from pypdf import PdfReader
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- Step 1: Dataset chosen ---
pdf_path = "vector-embeddings/penguins.pdf"
reader = PdfReader(pdf_path)

# --- Step 2: Split text into chunks ---
# Chunking is essential for:
# - Managing context window limits of embedding models
# - Providing focused, relevant results
# - Balancing between granularity and context preservation

## extract text from PDF
raw_text = ""
for page in reader.pages:
    raw_text += page.extract_text() + " "

## clean text_ remove unwanted artifacts
def clean_text(text):
    text = re.sub(r"-\s*\n", "", text)  # remove hyphen-line breaks
    text = re.sub(r"\s+", " ", text)    # collapse whitespace
    text = re.sub(r"Page\s+\d+", "", text)  # remove page labels
    return text.strip()

cleaned_text = clean_text(raw_text)

## chunk text
# Split text into overlapping chunks to preserve context across boundaries
# chunk_size: number of words per chunk (affects granularity)
# chunk_overlap: words shared between consecutive chunks (preserves context)
def chunk_text(text, chunk_size=300, chunk_overlap=50):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - chunk_overlap  # Move forward, but keep overlap

    return chunks

chunks = chunk_text(cleaned_text, chunk_size=300, chunk_overlap=50)

print("Number of chunks:", len(chunks))
print(f"Example chunks:\n {chunks[0]}\n ...\n {chunks[20]}")

# --- Step 3: Generate embeddings ---
# embeddings = dense vector representations that capture semantic meaning of the chunks
# these vectors enable similarity comparisons in high-dimensional space
# Using 'all-MiniLM-L6-v2': a lightweight model that balances speed and quality
# - Fast inference (good for CPU)
# - 384-dimensional embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks, show_progress_bar=True)
print(f"\nEmbeddings shape: {embeddings.shape}")  # Should be (num_chunks, dimensional embeddings)

# --- Step 4: Store embeddings in vector db ---
# In production, we would use a specialized vector database which enables efficient similarity search
# Here we use a simple numpy array 
embeddings_matrix = np.array(embeddings)
print(f"Stored {len(embeddings_matrix)} embeddings in vector database")

# --- Step 5: Similarity search ---
# Implement semantic search using vector similarity
def search_similar_chunks(query, top_k=3):
    """
    Search for the most similar chunks to the query using cosine similarity
    
    Args:
        query: Natural language question or search term
        top_k: Number of most similar chunks to return
    
    Returns:
        List of dicts containing chunk text, similarity score, and index
    """
    # Encode query using the same model used for chunks
    query_embedding = model.encode([query])
    
    # How similar is the query to each chunk?
    # Cosine similarity = -1 to 1, where 1 = identical
    similarities = cosine_similarity(query_embedding, embeddings_matrix)[0]
    
    # Get indices of top k most similar chunks
    top_indices = np.argsort(similarities)[::-1][:top_k]  # [::-1] reverses to descending order
    
    results = []
    for idx in top_indices:
        results.append({
            'chunk': chunks[idx],
            'similarity': similarities[idx],
            'index': idx
        })
    
    return results

# --- Step 6: Retrieve top k chunks for query ---
# Test the semantic search with a sample query
# This demonstrates RAG (Retrieval-Augmented Generation) - the first step before LLM generation

# Allow user to input custom query and top_k value
print("\n" + "="*60)
query = input("Enter your query (or press Enter for default 'What do penguins eat?'): ").strip()
if not query:
    query = "What do penguins eat?"

top_k_input = input("Enter number of results to retrieve (or press Enter for default 3): ").strip()
if top_k_input and top_k_input.isdigit():
    top_k = int(top_k_input)
else:
    top_k = 3

print(f"\n{'='*60}")
print(f"Query: {query}")
print(f"Retrieving top {top_k} results")
print(f"{'='*60}\n")

# Retrieve the most semantically similar chunks
top_results = search_similar_chunks(query, top_k=top_k)

# Display results with similarity scores
for i, result in enumerate(top_results, 1):
    print(f"Result {i} (Similarity: {result['similarity']:.4f}):")
    print(f"{result['chunk'][:300]}...")  # Show first 300 chars in the chunk only
    print()