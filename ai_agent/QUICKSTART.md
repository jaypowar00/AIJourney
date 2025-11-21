# 🚀 AIJourney — Quick Start Guide

## One-Minute Setup

### Prerequisites
- ✅ Python 3.9+
- ✅ Ollama running with a model (`ollama pull deepseek-r1`)

### Step 1: Install Dependencies (PowerShell)

```powershell
cd c:\Users\jaypo\Documents\Projects\AI Learning\AIJourney
pip install fastapi uvicorn streamlit requests sentence-transformers ollama langchain langchain-community langchain-huggingface chromadb
```

### Step 2: Start Backend (Terminal 1)

```powershell
uvicorn ai_agent.backend.server:app --reload
```

**Expected output:**
```
Uvicorn running on http://127.0.0.1:8000
```

### Step 3: Start Frontend (Terminal 2)

```powershell
cd c:\Users\jaypo\Documents\Projects\AI Learning\AIJourney
streamlit run ai_agent/frontend/streamlit_app.py
```

**Expected output:**
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

### Step 4: Use the App

1. **Sidebar**: Add documents
2. **Main Area**: Ask questions
3. **Results**: View answers with sources & confidence

---

## What Happens When You Ask a Question?

```
📝 Your Query
    ↓
🔎 Search local documents (semantic similarity)
    ↓
📊 Check confidence score
    ↓
┌─────────────────┐
│ Confidence High │ → Use local context only
└─────────────────┘
         │
         └─→ Low? → 🌐 Search web for more info
                ↓
           🤖 Combine all context
                ↓
           💬 Ollama generates answer
                ↓
           📤 Return answer + metadata
```

---

## Key Features

| Feature | What It Does |
|---------|-------------|
| 💾 **Document Storage** | Save text docs locally in CSV |
| 🔍 **Semantic Search** | Find relevant docs using embeddings |
| 🤖 **AI Agent** | Ask questions; agent retrieves context + calls LLM |
| 🌐 **Web Search Fallback** | If confidence is low, searches the web |
| 📊 **Confidence Score** | Shows how sure the agent is (0-1) |
| 🎨 **Professional UI** | Clean, modern Streamlit interface |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Backend won't start | Check if model files exist in `ai_agent/models/sentence-transformer/` |
| "Connection refused" | Make sure backend is running on port 8000 |
| Ollama timeout | Run `ollama serve` in separate terminal |
| "No results found" | Add documents first using the sidebar |

---

## API Documentation

### View Interactive Docs
- **Swagger**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

### Example cURL Commands

**Add a document:**
```powershell
curl -X POST http://127.0.0.1:8000/documents `
  -H "Content-Type: application/json" `
  -d '{"content": "Python is great for AI"}'
```

**Ask the agent:**
```powershell
curl -X POST http://127.0.0.1:8000/agent/answer `
  -H "Content-Type: application/json" `
  -d '{"query": "What is Python?", "use_web_search": true}'
```

---

## File Overview

```
ai_agent/
├── backend/
│   ├── server.py ............. FastAPI + Agent Logic
│   ├── storage.py ............ CSV Storage
│   └── data/docs.csv ......... Document Database
├── frontend/
│   └── streamlit_app.py ...... Professional UI
├── agent_demo.py ............ Original demo
├── models/sentence-transformer/ Local embeddings
└── README_SETUP.md .......... Full documentation
```

---

## Configuration

Edit `ai_agent/backend/server.py` to customize:

```python
CHUNK_SIZE = 800           # Chunk size for splitting
CHUNK_OVERLAP = 100        # Chunk overlap
TOP_K = 3                  # Results to retrieve
CONFIDENCE_THRESHOLD = 0.7 # Threshold for web search
LLM_MODEL = "deepseek-r1"  # Ollama model name
```

---

## Next Steps

- 📚 Add your own documents
- ❓ Ask questions about them
- 🔧 Customize the LLM model or chunking
- 🗄️ Later: Switch CSV → MongoDB/PostgreSQL
- 🌐 Later: Add real SerpAPI key for web search

---

**You're all set! 🎉**

Visit http://localhost:8501 and start using AIJourney!
