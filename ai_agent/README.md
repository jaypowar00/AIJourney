# 🤖 AI Agent – RAG + Web Search + Semantic Safety

This project implements a **smart AI Agent** that:
- Uses RAG (local documents via Chroma)
- Performs **web search** using private SearxNG
- Applies **semantic filtering** to avoid hallucinations
- Streams responses from a **local Ollama LLM (Phi-3)**

This is a production-ready mini-agent architecture suitable for:
- Hackathons  
- Internal AI events  
- Rapid prototyping  
- Local/private use cases  

---

# 📦 Installation

**macOS/Linux:**
```bash
cd ai_agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-agent.txt
```

**Windows (PowerShell):**
```powershell
cd ai_agent
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements-agent.txt
```

---

# 🐙 Setup LLM (Ollama)

### Install:
- **Windows/macOS:** Download from [ollama.ai](https://ollama.ai)
- **Linux:** `curl -fsSL https://ollama.ai/install.sh | sh`

### Pull model:
```bash
ollama pull phi3
```

### Run server:
```bash
ollama serve
```

---

# 🔍 Setup Web Search (SearxNG)

We run SearxNG locally for:
- Private searching  
- Fast responses  
- No captchas  
- No API limits  

### Start SearxNG:

**With Docker Compose (all platforms):**
```bash
cd ai_agent/searxng
docker compose up -d
```

**Without Docker (manual setup):**
Refer to [SearxNG docs](https://docs.searxng.org/)

### Test JSON API:

```
http://localhost:8080/search?q=hello&format=json
```

If you see JSON output → search tool is ready.

---

# ▶️ Run the Agent

```bash
python agent_demo.py
```

---

# 🧠 Agent Architecture

```
Query  
  ↓  
RAG Retrieval (Chroma)  
  ↓  
Confidence Score  
  ├── High → Use local context only  
  └── Low → Trigger Web Search  
                   ↓  
             SearxNG Results  
                   ↓  
   Semantic Filtering (cosine similarity)  
                   ↓  
    ┌──────────────┴───────────────┐
    │ Relevant → Combine contexts  │
    │ Irrelevant → "I don't know"  │
    └──────────────────────────────┘
                   ↓  
         Ollama LLM (streaming)
```

---

# 🛡️ Hallucination Prevention

The agent:
- Computes retrieval confidence  
- Only uses web search when needed  
- Checks web results using semantic similarity  
- Rejects irrelevant results  
- Uses grounded context to answer  
- Says **"I don't know"** if unsure  

---

# 🔬 Semantic Similarity Testing

You can test semantic matching separately:

```bash
python test_semantics.py
```

---

# 🧩 Customization

| Component | Editable in code |
|----------|------------------|
| CHUNK_SIZE | text chunking |
| CONFIDENCE_THRESHOLD | RAG trust level |
| SEARXNG_URL | Search engine endpoint |
| LLM_MODEL | Local LLM selection |
| semantic_relevance() | strict/loose filtering |

---

# 🏁 Summary

The AI agent in this directory is a robust hybrid system:
- Local RAG + Local LLM  
- Private web search lifecycle  
- Streaming LLM output  
- Strong hallucination control  
- Clean modular Python code  

Perfect for presenting in AI events or expanding into a full product.
