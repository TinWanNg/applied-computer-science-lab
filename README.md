# applied-computer-science-lab

## Getting Started
1. `pip install -r requirements.txt`
2. Fill in `OPENAI_API_KEY` in `.env`, referring to `.envTemplate`
3. Run using `streamlit run app.py`

---

## Core Theories

### 1. Retrieval-Augmented Generation (RAG)
RAG combines two powerful concepts:
- **Retrieval**: Finding relevant information from a knowledge base
- **Generation**: Using an LLM to create coherent answers

**Why RAG?** LLMs alone can hallucinate or lack specific domain knowledge. RAG grounds the LLM's responses in actual documents, making answers more accurate and verifiable.

### 2. Vector Embeddings
The system converts text into high-dimensional vectors using **all-MiniLM-L6-v2** (a HuggingFace model):
- Each chunk of text → 384-dimensional vector
- Similar meanings → similar vectors (close in vector space)
- Enables semantic search rather than just keyword matching

**Example**: "dog" and "canine" have similar vectors even though they're different words.

### 3. Semantic Search
Instead of exact keyword matching, the system finds documents by **meaning**:
- Question "What did they eat?" → embedding vector
- Compare with all document chunk vectors
- Return top-k most similar chunks (default: top 3)
- Uses **cosine similarity** (or distance) to measure relevance

### 4. Document Chunking
Large documents are split into smaller pieces:
- **Chunk size**: 1000 characters (~200-300 words)
- **Chunk overlap**: 200 characters (preserves context across boundaries)
- **Why?** Smaller chunks = more precise retrieval; overlap prevents losing context at boundaries

### 5. Multimodal Processing
The system handles multiple data types:
- **PDFs**: Extracted text per page
- **Images**: OpenAI Vision describes visual content → searchable text
- **Videos**: Whisper transcribes audio + Vision describes key frames
- **Text files**: Direct loading
- **Web pages**: (disabled) Would extract clean text from HTML

---

## Complete Workflow

### Phase 1: Document Ingestion & Vectorization

#### Step 1: Document Loading
**PDFs**:
```
local_docs/pdfs/*.pdf → PyPDFLoader → Document objects with page metadata
```

**Text Files**:
```
local_docs/texts/*.txt → TextLoader → Document objects
```

**Videos**:
```
local_docs/videos/*.mp4 → 
  1. Extract 3 key frames → OpenAI Vision describes visual content
  2. Whisper transcribes audio → text
  3. Combine: "Visual Description: ... Audio Transcription: ..."
```

**Images**:
```
local_docs/images/*.{jpg,png,webp} → OpenAI Vision API → textual descriptions
```

#### Step 2: Text Chunking
```
Raw documents → RecursiveCharacterTextSplitter →
  Chunks of ~1000 chars with 200-char overlap
  Separator priority: "\n\n" > "\n" > "." > " "
```

**Why recursive?** Tries to split at natural boundaries (paragraphs, sentences) before falling back to character splits.

#### Step 3: Metadata Sanitization
```
Clean metadata → Chroma-compatible format
  Only str, int, float, bool, None allowed
  Convert complex types (lists, dicts) → strings
```

#### Step 4: Embedding & Storage
```
Each chunk → all-MiniLM-L6-v2 → 384-dim vector →
  Stored in Chroma DB at chroma_db/db_<hash>/
  
Chroma stores:
  - Original text chunks
  - Vector embeddings
  - Metadata (source, page numbers, etc.)
```

**Smart Caching**: Each unique set of URLs gets its own DB folder (hashed). Same URLs = reuse DB; new URLs = fresh DB.

---

### Phase 2: Query Processing & Retrieval

#### Step 5: User Question
```
User types: "What is a vector database?"
```

#### Step 6: Question Embedding
```
Question text → all-MiniLM-L6-v2 → query vector (384 dimensions)
```
**Same model** as document embedding ensures compatibility!

#### Step 7: Similarity Search
```
Chroma.similarity_search_with_score(query, k=3) →
  1. Compare query vector with ALL document vectors
  2. Compute cosine similarity (or L2 distance)
  3. Return top-k most similar chunks + scores
```

---

### Phase 3: Answer Generation

#### Step 8: Context Assembly
```
Retrieved chunks → Format as context:

"[DOC 1 | source: db.pdf, page 5]
Vector databases store embeddings and enable semantic search...

[DOC 2 | source: ml.pdf, page 12]
Embeddings represent text as numerical vectors...

[DOC 3 | source: chroma.pdf, page 1]
Chroma is a vector database optimized for AI applications..."
```

#### Step 9: RAG Prompt Construction
```
System prompt:
  "Answer ONLY using the context below.
   If answer not in context, say 'I don't know'.
   At the end, cite which DOC numbers you used."

Context: [formatted retrieved chunks]

Question: What is a vector database?
```

#### Step 10: LLM Generation
```
GPT-4o-mini processes prompt →
  Generates grounded answer with citations
```

#### Step 11: Baseline Comparison
For comparison, a second answer is generated **without retrieval** - relies purely on LLM's training data.

---

## Data Flow Summary

```
User Documents
    ↓
[Loaders] → PDF pages, Images, Videos→text
    ↓
[Text Chunking] → 1000-char chunks with overlap
    ↓
[Embedding] → all-MiniLM-L6-v2 → 384-dim vectors
    ↓
[Chroma DB] → Persistent vector storage
    

User Question
    ↓
[Embedding] → Same model → Query vector
    ↓
[Similarity Search] → Top-k relevant chunks
    ↓
[Context Assembly] → Format chunks with metadata
    ↓
[LLM Prompt] → "Answer using this context..."
    ↓
[GPT-4o-mini] → Generated answer with citations
    ↓
[Display] → RAG answer + baseline + retrieved chunks
```