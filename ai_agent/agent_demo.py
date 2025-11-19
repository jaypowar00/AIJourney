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
LLM_MODEL = "deepseek-r1"

# --- searxng only works in linux
SEARXNG_URL = "http://localhost:8080/"  # or any other SearxNG instances available in INDIA

from serpapi import GoogleSearch
SERPAPI_KEY = "<API KEY HERE>" # get your free key from https://serpapi.com/. monthly 250 search credits for free plan.

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
# SerpAPI Search
# ------------------------------
def serpapi_search(query: str, max_results=5) -> list[str]:
    print("[Agent] 🔍 Searching internet via SerpAPI...")

    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": max_results
    }

    search = GoogleSearch(params)
    data = search.get_dict()
    # Dummy data for testing without need of getting data from GoogleSearch API
    data = {'search_metadata': {'id': '691e237cde1f338aacf8e946', 'status': 'Success', 'json_endpoint': 'https://serpapi.com/searches/383369ba5e9e0cff/691e237cde1f338aacf8e946.json', 'pixel_position_endpoint': 'https://serpapi.com/searches/383369ba5e9e0cff/691e237cde1f338aacf8e946.json_with_pixel_position', 'created_at': '2025-11-19 20:07:24 UTC', 'processed_at': '2025-11-19 20:07:24 UTC', 'google_url': 'https://www.google.com/search?q=should+design+gateway+backward+compatibility+with+soap+clients%3F&oq=should+design+gateway+backward+compatibility+with+soap+clients%3F&num=5&sourceid=chrome&ie=UTF-8', 'raw_html_file': 'https://serpapi.com/searches/383369ba5e9e0cff/691e237cde1f338aacf8e946.html', 'total_time_taken': 1.3}, 'search_parameters': {'engine': 'google', 'q': 'should design gateway backward compatibility with soap clients?', 'google_domain': 'google.com', 'num': '5', 'device': 'desktop'}, 'search_information': {'query_displayed': 'should design gateway backward compatibility with soap clients?', 'organic_results_state': 'Results for exact spelling'}, 'organic_results': [{'position': 1, 'title': 'Modernizing SOAP applications using Amazon API ...', 'link': 'https://aws.amazon.com/blogs/compute/modernizing-soap-applications-using-amazon-api-gateway-and-aws-lambda/', 'redirect_link': 'https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://aws.amazon.com/blogs/compute/modernizing-soap-applications-using-amazon-api-gateway-and-aws-lambda/&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQFnoECB0QAQ', 'displayed_link': 'https://aws.amazon.com › blogs › compute › modernizi...', 'favicon': 'https://serpapi.com/searches/691e237cde1f338aacf8e946/images/1deca313f688e651fe99d0d2c3f49181f22a2ad8547987796fa8b801e58d01c9.png', 'date': 'Jul 14, 2025', 'snippet': 'This post demonstrates how you can modernize legacy SOAP applications using Amazon API Gateway and AWS Lambda to create bidirectional proxy ...', 'snippet_highlighted_words': ['can', 'SOAP', 'Gateway', 'create'], 'source': 'Amazon Web Services (AWS)'}, {'position': 2, 'title': 'soap - Backwards compatibility and Web Services', 'link': 'https://stackoverflow.com/questions/1924216/backwards-compatibility-and-web-services', 'redirect_link': 'https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://stackoverflow.com/questions/1924216/backwards-compatibility-and-web-services&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQFnoECCUQAQ', 'displayed_link': '4 answers · 15 years ago', 'favicon': 'https://serpapi.com/searches/691e237cde1f338aacf8e946/images/1deca313f688e651fe99d0d2c3f49181d0fb227bb25ece6ee3186305a7b64a06.png', 'snippet': 'Unfortunately, you have to deal with backwards compatibility for any API you release to the public once you\'re out of "beta" mode (and ...', 'sitelinks': {'list': [{'title': 'Can API Gateway point a to a Web Service (SOAP) or ...', 'link': 'https://stackoverflow.com/questions/68591677/can-api-gateway-point-a-to-a-web-service-soap-or-the-alternative-would-be-can', 'answer_count': 2, 'date': 'Jul 30, 2021'}, {'title': 'Regarding SOAP clients backward compatibility for ...', 'link': 'https://stackoverflow.com/questions/28582689/regarding-soap-clients-backward-compatibility-for-service-name-change', 'answer_count': 1, 'date': 'Feb 18, 2015'}]}, 'missing': ['gateway'], 'must_include': {'word': 'gateway', 'link': 'https://www.google.com/search?num=5&sca_esv=55588fd05011d482&q=should+design+%22gateway%22+backward+compatibility+with+soap+clients?&sa=X&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ5t4CegQIKRAB'}, 'source': 'Stack Overflow'}, {'position': 3, 'title': 'Mastering the API Gateway Pattern in a Microservices ...', 'link': 'https://www.digitalapi.ai/blogs/api-gateway-pattern-in-a-microservices-architecture', 'redirect_link': 'https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.digitalapi.ai/blogs/api-gateway-pattern-in-a-microservices-architecture&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQFnoECBwQAQ', 'displayed_link': 'https://www.digitalapi.ai › blogs › api-gateway-pattern-i...', 'favicon': 'https://serpapi.com/searches/691e237cde1f338aacf8e946/images/1deca313f688e651fe99d0d2c3f491811aeb9b1d9fe3235cdaefdeab58a7470b.png', 'snippet': 'Instead of forcing every client to update instantly, the gateway provides versioned endpoints and backward-compatible facades. This lets teams modernise ...', 'snippet_highlighted_words': ['client', 'gateway', 'backward', 'compatible'], 'source': 'DigitalAPI'}, {'position': 4, 'title': 'SOAP vs REST: 9 Key Differences & When to Use Each in ...', 'link': 'https://www.superblocks.com/blog/soap-vs-rest', 'redirect_link': 'https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.superblocks.com/blog/soap-vs-rest&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQFnoECCMQAQ', 'displayed_link': 'https://www.superblocks.com › blog › soap-vs-rest', 'favicon': 'https://serpapi.com/searches/691e237cde1f338aacf8e946/images/1deca313f688e651fe99d0d2c3f491812657f014bba31a4ef0e7da94a4046e90.png', 'date': 'Jul 18, 2025', 'snippet': 'Migration can be gradual, but you have to plan around backward compatibility and client support. What tools help simplify REST API development?', 'snippet_highlighted_words': ['can', 'have to', 'backward compatibility', 'client'], 'sitelinks': {'inline': [{'title': 'What Is Soap?', 'link': 'https://www.superblocks.com/blog/soap-vs-rest#:~:text=What%20is%20SOAP%3F,-SOAP%20%28Simple%20Object%20Access'}, {'title': 'When To Use Soap Vs Rest', 'link': 'https://www.superblocks.com/blog/soap-vs-rest#:~:text=When%20to%20use%20SOAP%20vs%20REST,-Both%20SOAP%20and%20REST%20can'}, {'title': 'Rest Api Governance And...', 'link': 'https://www.superblocks.com/blog/soap-vs-rest#:~:text=REST%20API%20governance%20and%20security%20best%20practices,-As%20your%20API%20footprint%20grows'}]}, 'source': 'Superblocks'}], 'related_searches': [{'block_position': 1, 'query': 'Should design gateway backward compatibility with soap clients github', 'link': 'https://www.google.com/search?num=5&sca_esv=55588fd05011d482&q=Should+design+gateway+backward+compatibility+with+soap+clients+github&sa=X&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ1QJ6BAgwEAE', 'serpapi_link': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=Should+design+gateway+backward+compatibility+with+soap+clients+github'}, {'block_position': 1, 'query': 'Should design gateway backward compatibility with soap clients javascript', 'link': 'https://www.google.com/search?num=5&sca_esv=55588fd05011d482&q=Should+design+gateway+backward+compatibility+with+soap+clients+javascript&sa=X&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ1QJ6BAgvEAE', 'serpapi_link': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=Should+design+gateway+backward+compatibility+with+soap+clients+javascript'}, {'block_position': 1, 'query': 'AWS API Gateway convert SOAP to REST', 'link': 'https://www.google.com/search?num=5&sca_esv=55588fd05011d482&q=AWS+API+Gateway+convert+SOAP+to+REST&sa=X&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ1QJ6BAguEAE', 'serpapi_link': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=AWS+API+Gateway+convert+SOAP+to+REST'}, {'block_position': 1, 'query': 'Does API Gateway support SOAP', 'link': 'https://www.google.com/search?num=5&sca_esv=55588fd05011d482&q=Does+API+Gateway+support+SOAP&sa=X&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ1QJ6BAgtEAE', 'serpapi_link': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=Does+API+Gateway+support+SOAP'}], 'discussions_and_forums': [{'title': 'Regarding SOAP clients backward compatibility for service name change', 'link': 'https://stackoverflow.com/questions/28582689/regarding-soap-clients-backward-compatibility-for-service-name-change', 'date': '10 years ago', 'extensions': ['1 answer'], 'source': 'Stack Overflow', 'answers': [{'snippet': 'If you are sending SOAP xml packets directly, the service name change will not make any impact in any technology. Further, in .Net technology ...', 'link': 'https://stackoverflow.com/questions/28582689/regarding-soap-clients-backward-compatibility-for-service-name-change'}]}, {'title': 'Solved: SOAP backwards compatible interface', 'link': 'https://www.experts-exchange.com/questions/26750064/SOAP-backwards-compatible-interface.html', 'date': '14 years ago', 'source': 'Experts Exchange'}], 'pagination': {'current': 1, 'next': 'https://www.google.com/search?q=should+design+gateway+backward+compatibility+with+soap+clients?&num=5&sca_esv=55588fd05011d482&ei=fCMeadKDM62qwbkPy_CfoQY&start=5&sa=N&sstk=Af77f_cHtPUF4vdI23sHMfInlAgmNwywVe-3LsjoUtn4sqb_ng8GZ-jBbKoXWukoGBrKzxaJrFS_mHF1Pl0ZYBejYYEMeluPq_cMAQ&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ8NMDegQIDBAW', 'other_pages': {'2': 'https://www.google.com/search?q=should+design+gateway+backward+compatibility+with+soap+clients?&num=5&sca_esv=55588fd05011d482&ei=fCMeadKDM62qwbkPy_CfoQY&start=5&sa=N&sstk=Af77f_cHtPUF4vdI23sHMfInlAgmNwywVe-3LsjoUtn4sqb_ng8GZ-jBbKoXWukoGBrKzxaJrFS_mHF1Pl0ZYBejYYEMeluPq_cMAQ&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ8tMDegQIDBAE', '3': 'https://www.google.com/search?q=should+design+gateway+backward+compatibility+with+soap+clients?&num=5&sca_esv=55588fd05011d482&ei=fCMeadKDM62qwbkPy_CfoQY&start=10&sa=N&sstk=Af77f_cHtPUF4vdI23sHMfInlAgmNwywVe-3LsjoUtn4sqb_ng8GZ-jBbKoXWukoGBrKzxaJrFS_mHF1Pl0ZYBejYYEMeluPq_cMAQ&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ8tMDegQIDBAG', '4': 'https://www.google.com/search?q=should+design+gateway+backward+compatibility+with+soap+clients?&num=5&sca_esv=55588fd05011d482&ei=fCMeadKDM62qwbkPy_CfoQY&start=15&sa=N&sstk=Af77f_cHtPUF4vdI23sHMfInlAgmNwywVe-3LsjoUtn4sqb_ng8GZ-jBbKoXWukoGBrKzxaJrFS_mHF1Pl0ZYBejYYEMeluPq_cMAQ&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ8tMDegQIDBAI', '5': 'https://www.google.com/search?q=should+design+gateway+backward+compatibility+with+soap+clients?&num=5&sca_esv=55588fd05011d482&ei=fCMeadKDM62qwbkPy_CfoQY&start=20&sa=N&sstk=Af77f_cHtPUF4vdI23sHMfInlAgmNwywVe-3LsjoUtn4sqb_ng8GZ-jBbKoXWukoGBrKzxaJrFS_mHF1Pl0ZYBejYYEMeluPq_cMAQ&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ8tMDegQIDBAK', '6': 'https://www.google.com/search?q=should+design+gateway+backward+compatibility+with+soap+clients?&num=5&sca_esv=55588fd05011d482&ei=fCMeadKDM62qwbkPy_CfoQY&start=25&sa=N&sstk=Af77f_cHtPUF4vdI23sHMfInlAgmNwywVe-3LsjoUtn4sqb_ng8GZ-jBbKoXWukoGBrKzxaJrFS_mHF1Pl0ZYBejYYEMeluPq_cMAQ&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ8tMDegQIDBAM', '7': 'https://www.google.com/search?q=should+design+gateway+backward+compatibility+with+soap+clients?&num=5&sca_esv=55588fd05011d482&ei=fCMeadKDM62qwbkPy_CfoQY&start=30&sa=N&sstk=Af77f_cHtPUF4vdI23sHMfInlAgmNwywVe-3LsjoUtn4sqb_ng8GZ-jBbKoXWukoGBrKzxaJrFS_mHF1Pl0ZYBejYYEMeluPq_cMAQ&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ8tMDegQIDBAO', '8': 'https://www.google.com/search?q=should+design+gateway+backward+compatibility+with+soap+clients?&num=5&sca_esv=55588fd05011d482&ei=fCMeadKDM62qwbkPy_CfoQY&start=35&sa=N&sstk=Af77f_cHtPUF4vdI23sHMfInlAgmNwywVe-3LsjoUtn4sqb_ng8GZ-jBbKoXWukoGBrKzxaJrFS_mHF1Pl0ZYBejYYEMeluPq_cMAQ&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ8tMDegQIDBAQ', '9': 'https://www.google.com/search?q=should+design+gateway+backward+compatibility+with+soap+clients?&num=5&sca_esv=55588fd05011d482&ei=fCMeadKDM62qwbkPy_CfoQY&start=40&sa=N&sstk=Af77f_cHtPUF4vdI23sHMfInlAgmNwywVe-3LsjoUtn4sqb_ng8GZ-jBbKoXWukoGBrKzxaJrFS_mHF1Pl0ZYBejYYEMeluPq_cMAQ&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ8tMDegQIDBAS', '10': 'https://www.google.com/search?q=should+design+gateway+backward+compatibility+with+soap+clients?&num=5&sca_esv=55588fd05011d482&ei=fCMeadKDM62qwbkPy_CfoQY&start=45&sa=N&sstk=Af77f_cHtPUF4vdI23sHMfInlAgmNwywVe-3LsjoUtn4sqb_ng8GZ-jBbKoXWukoGBrKzxaJrFS_mHF1Pl0ZYBejYYEMeluPq_cMAQ&ved=2ahUKEwjSsf6Hgv-QAxUtVTABHUv4J2QQ8tMDegQIDBAU'}}, 'serpapi_pagination': {'current': 1, 'next_link': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=should+design+gateway+backward+compatibility+with+soap+clients%3F&start=5', 'next': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=should+design+gateway+backward+compatibility+with+soap+clients%3F&start=5', 'other_pages': {'2': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=should+design+gateway+backward+compatibility+with+soap+clients%3F&start=5', '3': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=should+design+gateway+backward+compatibility+with+soap+clients%3F&start=10', '4': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=should+design+gateway+backward+compatibility+with+soap+clients%3F&start=15', '5': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=should+design+gateway+backward+compatibility+with+soap+clients%3F&start=20', '6': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=should+design+gateway+backward+compatibility+with+soap+clients%3F&start=25', '7': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=should+design+gateway+backward+compatibility+with+soap+clients%3F&start=30', '8': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=should+design+gateway+backward+compatibility+with+soap+clients%3F&start=35', '9': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=should+design+gateway+backward+compatibility+with+soap+clients%3F&start=40', '10': 'https://serpapi.com/search.json?device=desktop&engine=google&google_domain=google.com&num=5&q=should+design+gateway+backward+compatibility+with+soap+clients%3F&start=45'}}}

    snippets = []

    # Google results come under 'organic_results'
    for item in data.get("organic_results", [])[:max_results]:
        title = item.get("title", "")
        snippet = item.get("snippet", "") or item.get("description", "")

        text = (title + " " + snippet).strip()

        if text:
            snippets.append(text)

    print(f"[Agent] 🌐 SerpAPI returned {len(snippets)} snippets")
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
        # for linux, we can use opensource searx_search by running local searxng
        # docker instance for unlimited free searches unlike serpapi search 250 Searches/Month in free plan.
        web_snippets = serpapi_search(search_query)
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
