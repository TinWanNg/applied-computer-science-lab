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
