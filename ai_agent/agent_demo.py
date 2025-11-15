import requests
from typing import List
import ollama

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter

from sentence_transformers import SentenceTransformer, util
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 3
CONFIDENCE_THRESHOLD = 0.7   # MADE LOWER so search triggers correctly
LLM_MODEL = "phi3"

SEARXNG_URL = "http://localhost:8080/"  # or any other SearxNG instances available in INDIA

# ------------------------------
# SearxNG Search (opensource metasearch)
# ------------------------------
def searx_search(query: str, max_results=5) -> list[str]:
    print("[Agent] 🔍 Searching internet via SearxNG...")
    url = f"{SEARXNG_URL}search"

    params = {
        "q": query,
        "format": "json",
    }

    resp = requests.get(url, params=params, timeout=10)
    data = resp.json()

    snippets = []

    for result in data.get("results", [])[:max_results]:
        text = ""
        if "title" in result:
            text += result["title"] + " "
        if "content" in result:
            text += result["content"]

        if text.strip():
            snippets.append(text.strip())

    print(f"[Agent] 🌐 SearxNG returned {len(snippets)} snippets")
    return snippets



# ------------------------------
# Vector DB Builder
# ------------------------------
def build_vectordb(docs: List[str]):
    splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = [c for d in docs for c in splitter.split_text(d)]
    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(chunks, emb)
    return vectordb, chunks


# ------------------------------
# Corrected Confidence Function
# ------------------------------
def compute_confidence(scores: List[float]) -> float:
    """
    Chroma returns DISTANCES (lower = closer).
    Convert distance → confidence.
    """
    if not scores:
        return 0.0

    # convert distance to similarity  (similarity = 1 / (1 + distance))
    sims = [1 / (1 + s) for s in scores]
    avg_sim = sum(sims) / len(sims)
    return avg_sim  # already 0..1-ish


# ------------------------------
# STREAMING Ollama LLM
# ------------------------------
def ask_ollama_stream(prompt: str) -> str:
    print("[Agent] 🤖 Generating answer...\n")
    full = []
    for chunk in ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], stream=True):
        token = chunk.get("message", {}).get("content", "")
        print(token, end="", flush=True)
        full.append(token)
    print("\n")
    return "".join(full)

# ------------------------------
# Semantic Relevance Check
# ------------------------------
def semantic_relevance(query: str, snippets: List[str], threshold=0.15) -> bool:
    if not snippets:
        return False

    combined = " ".join(snippets)
    q_emb = semantic_model.encode(query, convert_to_tensor=True)
    s_emb = semantic_model.encode(combined, convert_to_tensor=True)

    sim = float(util.cos_sim(q_emb, s_emb))

    print(f"[Agent] 🌐 Web semantic similarity: {sim:.3f}")

    return sim >= threshold

# ------------------------------
# Query Rewriter
# ------------------------------
def rewrite_query_for_search(query: str) -> str:
    # extract important keywords
    words = [w.lower() for w in query.split() if len(w) > 3]
    return " ".join(words)

# ------------------------------
# AGENT LOGIC
# ------------------------------
def agent_answer(vectordb, docs, query: str) -> str:
    print("\n[Agent] 🧠 Thinking...")
    results = vectordb.similarity_search_with_score(query, k=TOP_K)

    retrieved_chunks = [r[0].page_content for r in results]
    distances = [r[1] for r in results]

    confidence = compute_confidence(distances)

    print(f"[Agent] 🔎 Retrieval confidence: {confidence:.3f}")

    # Must show retrieved chunks
    for ch, dist in zip(retrieved_chunks, distances):
        print(f" - chunk dist={dist:.3f}: {ch[:100]}...")

    # ------------------------------
    # Case 1: Low confidence → use search
    # ------------------------------
    if confidence < CONFIDENCE_THRESHOLD:
        print("\n[Agent] 🚨 Low confidence! Triggering internet search...")
        search_query = rewrite_query_for_search(query)
        web_snippets = searx_search(search_query)
        if not semantic_relevance(query, web_snippets):
            print("[Agent] ❌ Web search semantically irrelevant! Fallback trigger.")
            return "Ohh, it seems I don't have enough reliable information to answer this query right now."

        combined_context = "\n\n".join(retrieved_chunks + web_snippets)

        prompt = f"""
Use the context below (local docs + web search) to answer.
If uncertain, state uncertainty.

CONTEXT:
{combined_context}

QUESTION: {query}

Final Answer:
"""
        return ask_ollama_stream(prompt)

    # ------------------------------
    # Case 2: High confidence → RAG only
    # ------------------------------
    ctx = "\n\n".join(retrieved_chunks)
    prompt = f"""
Use ONLY this context to answer:

{ctx}

QUESTION: {query}

If answer not found, say 'I don't know'.
"""
    return ask_ollama_stream(prompt)


# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    docs = [
        "Legacy system stores customer names and sensitive personal ID in plaintext.",
        "We plan to migrate from monolith to microservices using APIs and a new data model.",
        "All PII fields should be encrypted using AES-256 encryption before storage.",
        "Authentication will move from basic auth to OAuth 2.0 with JWT-based tokens.",
        "Developers must sanitize logs to ensure no PII information is stored in plaintext."
    ]

    print("[Agent] Building vector DB...")
    vectordb, chunks = build_vectordb(docs)

    query = "How should we design an API gateway for backward compatibility with SOAP clients?"
    # query = "What are the recommended patterns for migrating Oracle PL/SQL stored procedures into modern microservices?"
    print("\nQUESTION:", query)

    result = agent_answer(vectordb, docs, query)
