"""
agent_demo.py
A simple "AI Agent" wrapper around existing RAG flow.
- Uses Chroma similarity_search_with_score for a confidence check
- If confidence low -> does a simple DuckDuckGo HTML search scrape (no API key)
- Re-queries the local Ollama phi3 model with enriched context
"""

import os
import math
import requests
from bs4 import BeautifulSoup
from typing import List, Tuple

# make sure these imports match the versions in environment
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
# We'll call the local ollama server through the ollama python package for simplicity
import ollama

# ======= ======= CONFIG ======= =======
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 3                 # how many chunks to retrieve from the vector DB
CONFIDENCE_THRESHOLD = 0.7  # adjust this after experiments (0..1)
DUCKDUCKGO_MAX_RESULTS = 5
LLM_MODEL = "phi3"        # ollama model name

# ======= ======= Helper: simple DuckDuckGo HTML search (no API) =======
def duckduckgo_search(query: str, max_results: int = DUCKDUCKGO_MAX_RESULTS) -> List[str]:
    """
    Perform a simple HTML search on DuckDuckGo and return short snippet texts.
    Note: this is a simple scraping approach (no API) and returns textual snippets.
    """
    url = "https://html.duckduckgo.com/html/"
    resp = requests.post(url, data={"q": query})
    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for r in soup.select(".result__snippet, .result__a")[:max_results]:
        text = r.get_text(separator=" ", strip=True)
        if text:
            results.append(text)
    return results

# ======= ======= Build RAG (reuse same docs chunking approach) =======
def build_vectordb(docs: List[str]) -> Tuple[Chroma, List[str]]:
    splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for d in docs:
        chunks += splitter.split_text(d)
    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(chunks, emb)
    return vectordb, chunks

# ======= ======= Simple similarity/confidence check =======
def retrieve_with_scores(vectordb: Chroma, query: str, k: int = TOP_K):
    """
    Use Chroma similarity_search_with_score to get docs + scores.
    Returns list of tuples (doc, score) where score is similarity (higher better).
    """
    # many Chroma wrappers return (Document, score). Adjust if installed API differs.
    docs_and_scores = vectordb.similarity_search_with_score(query, k=k)
    return docs_and_scores

def normalize_scores_to_confidence(scores: List[float]) -> float:
    """
    Turn many vector DB distances / scores into a 0..1 confidence.
    The exact mapping depends on vectorstore API; here we assume `score` is similarity (higher better).
    We'll take the average of the retrieved scores and clip to 0..1.
    """
    if not scores:
        return 0.0
    avg = sum(scores) / len(scores)
    # Many vector DBs return cosine-like similarity between 0..1; clamp as safety.
    return max(0.0, min(1.0, avg))

# ======= ======= LLM call helper (using ollama python package) =======
def ask_ollama(prompt: str, model: str = LLM_MODEL, stream: bool = False) -> str:
    """
    Query the local ollama model using the ollama python client.
    This returns the generated text.
    """
    # If you prefer token-streaming, use stream=True and iterate results
    # Here we use a simple synchronous chat call
    messages = [{"role": "user", "content": prompt}]
    # ollama.chat returns an iterator if stream True, else returns a dict result
    if stream:
        out = []
        for chunk in ollama.chat(model=model, messages=messages, stream=True):
            # chunk shape: {'id':..., 'message': {'role': 'assistant', 'content': '...'}}
            content = chunk.get("message", {}).get("content", "")
            print(content, end="", flush=True)
            out.append(content)
        return "".join(out)
    else:
        res = ollama.chat(model=model, messages=messages)
        # res may be a dict like {'id':..., 'message': {'role': 'assistant', 'content': '...'}}
        return res.get("message", {}).get("content", "")

# ======= ======= Agent orchestration =======
def agent_answer(vectordb: Chroma, original_docs: List[str], query: str) -> str:
    # 1) RAG retrieval + scores
    docs_and_scores = retrieve_with_scores(vectordb, query, k=TOP_K)
    retrieved_texts = [d[0].page_content if hasattr(d[0], "page_content") else str(d[0]) for d in docs_and_scores]
    scores = [s for (_, s) in docs_and_scores]
    confidence = normalize_scores_to_confidence(scores)

    print(f"[Agent] RAG confidence (avg similarity): {confidence:.3f}")
    print("[Agent] Retrieved snippets:")
    for t, sc in zip(retrieved_texts, scores):
        print(f" - (score {sc:.3f}) {t[:200]}")

    # 2) If confidence is high, answer using only the local context
    if confidence >= CONFIDENCE_THRESHOLD:
        context = "\n\n".join(retrieved_texts)
        prompt = f"You are an expert advisor. Use ONLY the context below to answer the question. If the answer is not present, say 'I don't know'.\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer concisely and cite the relevant facts."
        answer = ask_ollama(prompt)
        return answer

    # 3) Confidence low -> call web search tool
    print("[Agent] Low confidence. Performing web search to gather more context...")
    web_snippets = duckduckgo_search(query)
    print(f"[Agent] Collected {len(web_snippets)} web snippets")
    # combine retrieved local docs + web snippets as extended context
    combined_context = "\n\n".join(retrieved_texts + web_snippets)
    prompt2 = (
        "You are an expert advisor. Use the CONTEXT below (local docs + web search) to answer the question. "
        "If contradictory or uncertain information exists, state the uncertainty and provide the most likely answer with reasons.\n\n"
        f"CONTEXT:\n{combined_context}\n\nQUESTION: {query}\n\nAnswer concisely and list which sources (local/web) support each claim."
    )
    answer2 = ask_ollama(prompt2)
    # Simple post-check: if the answer contains "I don't know" or is too short, fallback
    if len(answer2.strip()) < 20 or "I don't know" in answer2.lower():
        return "Ohh, it seems I don't have enough reliable information right now. I can try a deeper web search or you can provide a document."
    return answer2

# ======= ======= Example usage (main) =======
if __name__ == "__main__":
    # Example: reuse the same small docs from rag_demo.py
    docs = [
        "Legacy system stores customer names and sensitive personal ID in plaintext.",
        "We plan to migrate from monolith to microservices using APIs and a new data model.",
        "All PII fields should be encrypted using AES-256 encryption before storage.",
        "Authentication will move from basic auth to OAuth 2.0 with JWT-based tokens.",
        "Developers must sanitize logs to ensure no PII information is stored in plaintext."
    ]

    print("Building local vector DB (this can take a few seconds on first run)...")
    vectordb, chunks = build_vectordb(docs)
    print("Vector DB ready. Example chunk count:", len(chunks))

    # Run a sample query
    # q = "How should we handle PII during migration?"
    q = "What are the recommended patterns for migrating Oracle PL/SQL stored procedures into modern microservices?"
    print("\nQUESTION:", q)
    ans = agent_answer(vectordb, docs, q)
    print("\n\n=== AGENT ANSWER ===\n")
    print(ans)
