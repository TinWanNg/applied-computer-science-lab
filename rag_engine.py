"""
RAG engine for question answering.
Handles both retrieval-augmented and baseline LLM responses.
"""
from langchain_community.vectorstores import Chroma

from config import openai_client
from vectorstore import search_similar_chunks


# ------------------------------------------------------------
# RAG Answer: LLM + retrieved context
# ------------------------------------------------------------
def answer_with_rag(vectorstore: Chroma, question: str, top_k: int = 3):
    """
    Retrieve relevant chunks from the vectorstore and ask OpenAI
    to answer only using that context.
    """
    retrieved = search_similar_chunks(vectorstore, question, top_k=top_k)

    # Filter for relevance - only include chunks with similarity > 0.3 (cosine similarity)
    relevant_chunks = [r for r in retrieved if r["similarity"] > 0.3]
    
    context = ""
    if relevant_chunks:
        for i, r in enumerate(relevant_chunks, start=1):
            src = r["metadata"].get("source")
            page = r["metadata"].get("page")
            sim = r["similarity"]
            page_info = f", page {page}" if page is not None else ""
            context += f"[DOC {i} | source: {src}{page_info} | relevance: {sim:.2f}]\n{r['chunk']}\n\n"
    else:
        context = "[No highly relevant documents found in the knowledge base]\n"

    prompt = f"""
    You are a helpful assistant. Answer the user's question using the provided context and your knowledge.

    Instructions:
    1. PRIORITIZE information from the context below when it's relevant to the question
    2. If the context contains relevant information, use it as the primary source and cite the documents
    3. If the context is partially relevant, blend it with your general knowledge
    4. If the context is not relevant at all, clearly state that and answer using your general knowledge
    5. Be conversational and helpful - don't refuse to answer just because information isn't in the documents

    Context from retrieved documents:
    {context}

    Question: {question}

    At the end, list which DOC numbers you used in the format: [Citations: DOC 1, DOC 3]
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    
    return response.choices[0].message.content.strip(), retrieved


# ------------------------------------------------------------
# Baseline Answer: LLM without retrieval
# ------------------------------------------------------------
def answer_without_rag(question: str):
    """Ask OpenAI directly without any retrieval (for comparison)."""
    prompt = f"Answer this question using your own knowledge:\n\n{question}"
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    
    return response.choices[0].message.content.strip()
