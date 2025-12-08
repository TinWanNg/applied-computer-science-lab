from pypdf import PdfReader
import re
from sentence_transformers import SentenceTransformer
import chromadb

# --- Step 1: Dataset chosen ---
pdf_path = "penguins.pdf"
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

# --- Step 4: Store embeddings in vector db (ChromaDB) ---

# Initialize persistent Chroma client and collection
chroma_client = chromadb.PersistentClient(path="chroma_db")  # folder on disk
collection = chroma_client.get_or_create_collection(
    name="penguins_chunks",
    metadata={"hnsw:space": "cosine"}  # cosine distance space
)

# Prepare IDs and optional metadata for each chunk
ids = [f"chunk_{i}" for i in range(len(chunks))]
metadatas = [{"chunk_index": i} for i in range(len(chunks))]

# Upsert into Chroma (so re-running script just overwrites)
collection.upsert(
    ids=ids,
    documents=chunks,
    embeddings=embeddings.tolist(),  # Chroma expects lists, not numpy arrays
    metadatas=metadatas,
)

print(f"Stored {len(chunks)} embeddings in Chroma vector database")

# --- Step 5: Similarity search ---
# Implement semantic search using vector similarity via vector DB
def search_similar_chunks(query, top_k=3):
    """
    Search for the most similar chunks to the query using ChromaDB

    Args:
        query: Natural language question or search term
        top_k: Number of most similar chunks to return

    Returns:
        List of dicts containing chunk text, similarity score (approx), and metadata
    """
    # Encode query using the same model used for chunks
    query_embedding = model.encode([query])[0].tolist()

    # Let Chroma do the similarity search internally
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances", "metadatas"],  # removed "ids"
    )

    docs = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    ids = results["ids"][0]

    results = []
    for i in range(len(docs)):
        # Chroma returns distances in cosine space → smaller = closer
        # We convert to a similarity-ish score for display
        similarity = 1.0 - distances[i]
        results.append({
            "id": ids[i],
            "chunk": docs[i],
            "similarity": similarity,
            "index": metadatas[i].get("chunk_index"),
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