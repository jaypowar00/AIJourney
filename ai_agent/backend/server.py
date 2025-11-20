from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict
import json
import uvicorn
import urllib3
from pathlib import Path
import ollama

import time

from sentence_transformers import SentenceTransformer, util
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter

from .storage import save_doc, load_documents

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 3
CONFIDENCE_THRESHOLD = 0.7
LLM_MODEL = "deepseek-r1"

SERPAPI_KEY = "<API KEY HERE>"   # must set this


# -----------------------------------------------------------
# LOAD LOCAL MODELS
# -----------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = str(BASE_DIR / "models" / "sentence-transformer")

semantic_model = SentenceTransformer(MODEL_PATH)

embedding_model = HuggingFaceEmbeddings(
    model_name=MODEL_PATH,
    model_kwargs={"local_files_only": True}
)


# -----------------------------------------------------------
# FASTAPI SETUP
# -----------------------------------------------------------

app = FastAPI(
    title="AIJourney Backend",
    description="Local Semantic Search + AI Agent with Web Search",
    version="2.0.0"
)


# -----------------------------------------------------------
# SCHEMAS
# -----------------------------------------------------------

class DocIn(BaseModel):
    content: str

class DocOut(BaseModel):
    id: str
    content: str

class QueryIn(BaseModel):
    query: str
    top_k: int = 3

class SearchResult(BaseModel):
    id: str
    content: str
    score: float

class QueryResponse(BaseModel):
    results: List[SearchResult]

class AgentQueryIn(BaseModel):
    query: str
    use_web_search: bool = True

class AgentQueryResponse(BaseModel):
    answer: str
    confidence: float
    used_web_search: bool
    retrieved_chunks: List[str]

# -----------------------------------------------------------
# SERPAPI SEARCH
# -----------------------------------------------------------

def serpapi_search(query: str, max_results=5) -> List[str]:
    print("[Agent] 🔍 Searching internet via SerpAPI...")

    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": max_results
    }

    try:
        from serpapi import GoogleSearch
        search = GoogleSearch(params)
        data = search.get_dict()
        data = {'search_metadata': {'id': '691e237cde1f338aacf8e946', 'status': 'Success', 'json_endpoint': 'https://serpapi.com/searches/383369ba5e9e0cff/691e237cde1f338aacf8e946.json', 'pixel_position_endpoint': 'https://serpapi.com/searches/383369ba5e9e0cff/691e237cde1f338aacf8e946.json_with_pixel_position', 'created_at': '2025-11-19 20:07:24 UTC', 'processed_at': '2025-11-19 20:07:24 UTC', 'google_url': 'https://www.google.com/search?q=should+design+gateway+backward+compatibility+with+soap+clients%3F&oq=should+design+gateway+backward+compatibility+with+soap+clients%3F&num=5&sourceid=chrome&ie=UTF-8', 'raw_html_file': 'https://serpapi.com/searches/383369ba5e9e0cff/691e237cde1f338aacf8e946.html', 'total_time_taken': 1.3}, 'search_parameters': {'engine': 'google', 'q': 'should design gateway backward compatibility with soap clients?', 'google_domain': 'google.com', 'num': '5', 'device': 'desktop'}, 'search_information': {'query_displayed': 'should design gateway backward compatibility with soap clients?', 'organic_results_state': 'Results for exact spelling'}, 'organic_results': [{'position': 1, 'title': 'Modernizing SOAP applications using Amazon API ...', 'link': 'https://aws.amazon.com/blogs/compute/modernizing-soap-applications-using-amazon-api-gateway-and-aws-lambda/', 'redirect_link': 'https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://aws.amazon.com/blogs/compute/modernizing-soap-applications-using-amazon-api-gateway-and-aws-lambda/&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQFnoECB0QAQ', 'displayed_link': 'https://aws.amazon.com › blogs › compute › modernizi...', 'favicon': 'https://serpapi.com/searches/691e237cde1f338aacf8e946/images/1deca313f688e651fe99d0d2c3f49181f22a2ad8547987796fa8b801e58d01c9.png', 'date': 'Jul 14, 2025', 'snippet': 'This post demonstrates how you can modernize legacy SOAP applications using Amazon API Gateway and AWS Lambda to create bidirectional proxy ...', 'snippet_highlighted_words': ['can', 'SOAP', 'Gateway', 'create'], 'source': 'Amazon Web Services (AWS)'}, {'position': 2, 'title': 'soap - Backwards compatibility and Web Services', 'link': 'https://stackoverflow.com/questions/1924216/backwards-compatibility-and-web-services', 'redirect_link': 'https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://stackoverflow.com/questions/1924216/backwards-compatibility-and-web-services&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQFnoECCUQAQ', 'displayed_link': '4 answers · 15 years ago', 'favicon': 'https://serpapi.com/searches/691e237cde1f338aacf8e946/images/1deca313f688e651fe99d0d2c3f49181d0fb227bb25ece6ee3186305a7b64a06.png', 'snippet': 'Unfortunately, you have to deal with backwards compatibility for any API you release to the public once you\'re out of "beta" mode (and ...', 'sitelinks': {'list': [{'title': 'Can API Gateway point a to a Web Service (SOAP) or ...', 'link': 'https://stackoverflow.com/questions/68591677/can-api-gateway-point-a-to-a-web-service-soap-or-the-alternative-would-be-can', 'answer_count': 2, 'date': 'Jul 30, 2021'}, {'title': 'Regarding SOAP clients backward compatibility for ...', 'link': 'https://stackoverflow.com/questions/28582689/regarding-soap-clients-backward-compatibility-for-service-name-change', 'answer_count': 1, 'date': 'Feb 18, 2015'}]}, 'missing': ['gateway'], 'must_include': {'word': 'gateway', 'link': 'https://www.google.com/search?num=5&sca_esv=55588fd05011d482&q=should+design+%22gateway%22+backward+compatibility+with+soap+clients?&sa=X&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ5t4CegQIKRAB'}, 'source': 'Stack Overflow'}, {'position': 3, 'title': 'Mastering the API Gateway Pattern in a Microservices ...', 'link': 'https://www.digitalapi.ai/blogs/api-gateway-pattern-in-a-microservices-architecture', 'redirect_link': 'https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.digitalapi.ai/blogs/api-gateway-pattern-in-a-microservices-architecture&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQFnoECBwQAQ', 'displayed_link': 'https://www.digitalapi.ai › blogs › api-gateway-pattern-i...', 'favicon': 'https://serpapi.com/searches/691e237cde1f338aacf8e946/images/1deca313f688e651fe99d0d2c3f491811aeb9b1d9fe3235cdaefdeab58a7470b.png', 'snippet': 'Instead of forcing every client to update instantly, the gateway provides versioned endpoints and backward-compatible facades. This lets teams modernise ...', 'snippet_highlighted_words': ['client', 'gateway', 'backward', 'compatible'], 'source': 'DigitalAPI'}, {'position': 4, 'title': 'SOAP vs REST: 9 Key Differences & When to Use Each in ...', 'link': 'https://www.superblocks.com/blog/soap-vs-rest', 'redirect_link': 'https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.superblocks.com/blog/soap-vs-rest&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQFnoECCMQAQ', 'displayed_link': 'https://www.superblocks.com › blog › soap-vs-rest', 'favicon': 'https://serpapi.com/searches/691e237cde1f338aacf8e946/images/1deca313f688e651fe99d0d2c3f491812657f014bba31a4ef0e7da94a4046e90.png', 'date': 'Jul 18, 2025', 'snippet': 'Migration can be gradual, but you have to plan around backward compatibility and client support. What tools help simplify REST API development?', 'snippet_highlighted_words': ['can', 'have to', 'backward compatibility', 'client'], 'sitelinks': {'inline': [{'title': 'What Is Soap?', 'link': 'https://www.superblocks.com/blog/soap-vs-rest#:~:text=What%20is%20SOAP%3F,-SOAP%20%28Simple%20Object%20Access'}, {'title': 'When To Use Soap Vs Rest', 'link': 'https://www.superblocks.com/blog/soap-vs-rest#:~:text=When%20to%20use%20SOAP%20vs%20REST,-Both%20SOAP%20and%20REST%20can'}, {'title': 'Rest Api Governance And...', 'link': 'https://www.superblocks.com/blog/soap-vs-rest#:~:text=REST%20API%20governance%20and%20security%20best%20practices,-As%20your%20API%20footprint%20grows'}]}, 'source': 'Superblocks'}], 'related_searches': [{'block_position': 1, 'query': 'Should design gateway backward compatibility with soap clients github', 'link': 'https://www.google.com/search?num=5&sca_esv=55588fd05011d482&q=Should+design+gateway+backward+compatibility+with+soap+clients+github&sa=X&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ1QJ6BAgwEAE', 'serpapi_link': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=Should+design+gateway+backward+compatibility+with+soap+clients+github'}, {'block_position': 1, 'query': 'Should design gateway backward compatibility with soap clients javascript', 'link': 'https://www.google.com/search?num=5&sca_esv=55588fd05011d482&q=Should+design+gateway+backward+compatibility+with+soap+clients+javascript&sa=X&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ1QJ6BAgvEAE', 'serpapi_link': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=Should+design+gateway+backward+compatibility+with+soap+clients+javascript'}, {'block_position': 1, 'query': 'AWS API Gateway convert SOAP to REST', 'link': 'https://www.google.com/search?num=5&sca_esv=55588fd05011d482&q=AWS+API+Gateway+convert+SOAP+to+REST&sa=X&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ1QJ6BAguEAE', 'serpapi_link': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=AWS+API+Gateway+convert+SOAP+to+REST'}, {'block_position': 1, 'query': 'Does API Gateway support SOAP', 'link': 'https://www.google.com/search?num=5&sca_esv=55588fd05011d482&q=Does+API+Gateway+support+SOAP&sa=X&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ1QJ6BAgtEAE', 'serpapi_link': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=Does+API+Gateway+support+SOAP'}], 'discussions_and_forums': [{'title': 'Regarding SOAP clients backward compatibility for service name change', 'link': 'https://stackoverflow.com/questions/28582689/regarding-soap-clients-backward-compatibility-for-service-name-change', 'date': '10 years ago', 'extensions': ['1 answer'], 'source': 'Stack Overflow', 'answers': [{'snippet': 'If you are sending SOAP xml packets directly, the service name change will not make any impact in any technology. Further, in .Net technology ...', 'link': 'https://stackoverflow.com/questions/28582689/regarding-soap-clients-backward-compatibility-for-service-name-change'}]}, {'title': 'Solved: SOAP backwards compatible interface', 'link': 'https://www.experts-exchange.com/questions/26750064/SOAP-backwards-compatible-interface.html', 'date': '14 years ago', 'source': 'Experts Exchange'}], 'pagination': {'current': 1, 'next': 'https://www.google.com/search?q=should+design+gateway+backward+compatibility+with+soap+clients?&num=5&sca_esv=55588fd05011d482&ei=fCMeadKDM62qwbkPy_CfoQY&start=5&sa=N&sstk=Af77f_cHtPUF4vdI23sHMfInlAgmNwywVe-3LsjoUtn4sqb_ng8GZ-jBbKoXWukoGBrKzxaJrFS_mHF1Pl0ZYBejYYEMeluPq_cMAQ&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ8NMDegQIDBAW', 'other_pages': {'2': 'https://www.google.com/search?q=should+design+gateway+backward+compatibility+with+soap+clients?&num=5&sca_esv=55588fd05011d482&ei=fCMeadKDM62qwbkPy_CfoQY&start=5&sa=N&sstk=Af77f_cHtPUF4vdI23sHMfInlAgmNwywVe-3LsjoUtn4sqb_ng8GZ-jBbKoXWukoGBrKzxaJrFS_mHF1Pl0ZYBejYYEMeluPq_cMAQ&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ8tMDegQIDBAE', '3': 'https://www.google.com/search?q=should+design+gateway+backward+compatibility+with+soap+clients?&num=5&sca_esv=55588fd05011d482&ei=fCMeadKDM62qwbkPy_CfoQY&start=10&sa=N&sstk=Af77f_cHtPUF4vdI23sHMfInlAgmNwywVe-3LsjoUtn4sqb_ng8GZ-jBbKoXWukoGBrKzxaJrFS_mHF1Pl0ZYBejYYEMeluPq_cMAQ&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ8tMDegQIDBAG', '4': 'https://www.google.com/search?q=should+design+gateway+backward+compatibility+with+soap+clients?&num=5&sca_esv=55588fd05011d482&ei=fCMeadKDM62qwbkPy_CfoQY&start=15&sa=N&sstk=Af77f_cHtPUF4vdI23sHMfInlAgmNwywVe-3LsjoUtn4sqb_ng8GZ-jBbKoXWukoGBrKzxaJrFS_mHF1Pl0ZYBejYYEMeluPq_cMAQ&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ8tMDegQIDBAI', '5': 'https://www.google.com/search?q=should+design+gateway+backward+compatibility+with+soap+clients?&num=5&sca_esv=55588fd05011d482&ei=fCMeadKDM62qwbkPy_CfoQY&start=20&sa=N&sstk=Af77f_cHtPUF4vdI23sHMfInlAgmNwywVe-3LsjoUtn4sqb_ng8GZ-jBbKoXWukoGBrKzxaJrFS_mHF1Pl0ZYBejYYEMeluPq_cMAQ&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ8tMDegQIDBAK', '6': 'https://www.google.com/search?q=should+design+gateway+backward+compatibility+with+soap+clients?&num=5&sca_esv=55588fd05011d482&ei=fCMeadKDM62qwbkPy_CfoQY&start=25&sa=N&sstk=Af77f_cHtPUF4vdI23sHMfInlAgmNwywVe-3LsjoUtn4sqb_ng8GZ-jBbKoXWukoGBrKzxaJrFS_mHF1Pl0ZYBejYYEMeluPq_cMAQ&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ8tMDegQIDBAM', '7': 'https://www.google.com/search?q=should+design+gateway+backward+compatibility+with+soap+clients?&num=5&sca_esv=55588fd05011d482&ei=fCMeadKDM62qwbkPy_CfoQY&start=30&sa=N&sstk=Af77f_cHtPUF4vdI23sHMfInlAgmNwywVe-3LsjoUtn4sqb_ng8GZ-jBbKoXWukoGBrKzxaJrFS_mHF1Pl0ZYBejYYEMeluPq_cMAQ&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ8tMDegQIDBAO', '8': 'https://www.google.com/search?q=should+design+gateway+backward+compatibility+with+soap+clients?&num=5&sca_esv=55588fd05011d482&ei=fCMeadKDM62qwbkPy_CfoQY&start=35&sa=N&sstk=Af77f_cHtPUF4vdI23sHMfInlAgmNwywVe-3LsjoUtn4sqb_ng8GZ-jBbKoXWukoGBrKzxaJrFS_mHF1Pl0ZYBejYYEMeluPq_cMAQ&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ8tMDegQIDBAQ', '9': 'https://www.google.com/search?q=should+design+gateway+backward+compatibility+with+soap+clients?&num=5&sca_esv=55588fd05011d482&ei=fCMeadKDM62qwbkPy_CfoQY&start=40&sa=N&sstk=Af77f_cHtPUF4vdI23sHMfInlAgmNwywVe-3LsjoUtn4sqb_ng8GZ-jBbKoXWukoGBrKzxaJrFS_mHF1Pl0ZYBejYYEMeluPq_cMAQ&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ8tMDegQIDBAS', '10': 'https://www.google.com/search?q=should+design+gateway+backward+compatibility+with+soap+clients?&num=5&sca_esv=55588fd05011d482&ei=fCMeadKDM62qwbkPy_CfoQY&start=45&sa=N&sstk=Af77f_cHtPUF4vdI23sHMfInlAgmNwywVe-3LsjoUtn4sqb_ng8GZ-jBbKoXWukoGBrKzxaJrFS_mHF1Pl0ZYBejYYEMeluPq_cMAQ&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ8tMDegQIDBAU'}}, 'serpapi_pagination': {'current': 1, 'next_link': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=should+design+gateway+backward+compatibility+with+soap+clients%3F&start=5', 'next': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=should+design+gateway+backward+compatibility+with+soap+clients%3F&start=5', 'other_pages': {'2': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=should+design+gateway+backward+compatibility+with+soap+clients%3F&start=5', '3': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=should+design+gateway+backward+compatibility+with+soap+clients%3F&start=10', '4': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=should+design+gateway+backward+compatibility+with+soap+clients%3F&start=15', '5': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=should+design+gateway+backward+compatibility+with+soap+clients%3F&start=20', '6': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=should+design+gateway+backward+compatibility+with+soap+clients%3F&start=25', '7': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=should+design+gateway+backward+compatibility+with+soap+clients%3F&start=30', '8': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=should+design+gateway+backward+compatibility+with+soap+clients%3F&start=35', '9': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=should+design+gateway+backward+compatibility+with+soap+clients%3F&start=40', '10': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=should+design+gateway+backward+compatibility+with+soap+clients%3F&start=45'}}}

    except Exception as e:
        print("[Agent] ❌ SerpAPI error:", e)
        return []

    snippets = []

    for item in data.get("organic_results", [])[:max_results]:
        title = item.get("title", "")
        snippet = item.get("snippet", "") or item.get("description", "")
        text = (title + " " + snippet).strip()

        if text:
            snippets.append(text)

    print(f"[Agent] 🌐 SerpAPI returned {len(snippets)} snippets")
    return snippets


# -----------------------------------------------------------
# VECTOR DB BUILDER (with metadata)
# -----------------------------------------------------------

def build_vectordb_from_docs(docs: List[Dict]):
    contents = [d["content"] for d in docs]

    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []
    metadata = []

    for idx, content in enumerate(contents):
        split = splitter.split_text(content)
        chunks.extend(split)
        metadata.extend([{"source": docs[idx]["id"]}] * len(split))

    vectordb = Chroma.from_texts(
        chunks,
        embedding_model,
        metadatas=metadata
    )

    return vectordb, chunks


# -----------------------------------------------------------
# CONFIDENCE SCORING
# -----------------------------------------------------------

def compute_confidence(distances: List[float]) -> float:
    if not distances:
        return 0.0
    sims = [1 / (1 + d) for d in distances]
    return sum(sims) / len(sims)


# -----------------------------------------------------------
# QUERY REWRITE (ONLY IF NEEDED)
# -----------------------------------------------------------

STOPWORDS = {"what", "how", "can", "should", "does", "why", "where", "when"}

def rewrite_query_if_needed(query: str) -> str:
    words = query.lower().split()
    if not any(w in STOPWORDS for w in words):
        return query  # no rewrite needed
    # rewrite = keep important words
    important = [w for w in words if len(w) > 3 and w not in STOPWORDS]
    return " ".join(important)


# -----------------------------------------------------------
# SEMANTIC RELEVANCE
# -----------------------------------------------------------

def semantic_relevance(query: str, snippets: List[str], threshold=0.15) -> bool:
    if not snippets:
        return False

    combined = " ".join(snippets)
    q_emb = semantic_model.encode(query, convert_to_tensor=True)
    s_emb = semantic_model.encode(combined, convert_to_tensor=True)

    sim = float(util.cos_sim(q_emb, s_emb))

    print(f"[Agent] 🌐 Web semantic similarity: {sim:.3f}")
    return sim >= threshold


# -----------------------------------------------------------
# OLLAMA LLM (STREAMING)
# -----------------------------------------------------------

def ask_ollama_stream(prompt: str) -> str:
    print("[Agent] 🤖 Generating answer...\n")
    full = []

    for chunk in ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    ):
        token = chunk.get("message", {}).get("content", "")
        print(token, end="", flush=True)
        full.append(token)

    print("\n")
    return "".join(full)


# -----------------------------------------------------------
# FASTAPI ROUTES
# -----------------------------------------------------------

@app.get("/documents", response_model=List[DocOut], tags=["Documents"])
def get_documents():
    return load_documents()

@app.post("/documents", response_model=DocOut, tags=["Documents"])
def post_document(doc: DocIn):
    if not doc.content.strip():
        raise HTTPException(status_code=400, detail="Empty content")
    return save_doc(doc.content.strip())


@app.post("/query", response_model=QueryResponse, tags=["Semantic Search"])
def query_docs(q: QueryIn):
    docs = load_documents()
    if not docs:
        return {"results": []}

    q_emb = semantic_model.encode(q.query, convert_to_tensor=True)

    results = []
    for d in docs:
        d_emb = semantic_model.encode(d["content"], convert_to_tensor=True)
        score = float(util.cos_sim(q_emb, d_emb).item())

        results.append(SearchResult(
            id=d["id"],
            content=d["content"],
            score=score
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return QueryResponse(results=results[:q.top_k])


@app.post("/agent/answer")
def agent_answer(q: AgentQueryIn):
    status_updates = []

    status_updates.append("🤖 Thinking...")
    docs = load_documents()

    if not docs:
        return {
            "status": status_updates + ["❌ No documents found"],
            "answer": "",
            "confidence": 0.0,
            "used_web_search": False,
            "retrieved_chunks": []
        }

    status_updates.append("📚 Building vector DB...")
    vectordb, chunks = build_vectordb_from_docs(docs)

    status_updates.append("🔍 Performing semantic search...")
    results = vectordb.similarity_search_with_score(q.query, k=TOP_K)

    retrieved_chunks = [r[0].page_content for r in results]
    distances = [r[1] for r in results]
    confidence = compute_confidence(distances)

    used_web_search = False
    context_chunks = retrieved_chunks[:]

    # fallback: web search
    if confidence < CONFIDENCE_THRESHOLD and q.use_web_search:
        status_updates.append("🌐 Low confidence → Searching internet...")
        rewritten = rewrite_query_if_needed(q.query)

        snippets = serpapi_search(rewritten)
        status_updates.append(f"🌐 SerpAPI returned {len(snippets)} snippets")

        if semantic_relevance(q.query, snippets):
            status_updates.append("📌 Web results relevant → using them")
            context_chunks.extend(snippets)
            used_web_search = True
        else:
            status_updates.append("❌ Web results not relevant")

    # Build final prompt
    context = "\n\n".join(context_chunks)

    prompt = f"""
Use the context below to answer the question.
If uncertain, say you are unsure.

CONTEXT:
{context}

QUESTION: {q.query}

Answer:
"""

    status_updates.append("🤖 Generating response (please wait)...")
    answer = ask_ollama_stream(prompt)  # This takes time

    status_updates.append("✔ Done")

    return {
        "status": status_updates,
        "answer": answer,
        "confidence": confidence,
        "used_web_search": used_web_search,
        "retrieved_chunks": retrieved_chunks
    }



if __name__ == "__main__":
    uvicorn.run("ai_agent.backend.server:app", host="127.0.0.1", port=8000, reload=True)
