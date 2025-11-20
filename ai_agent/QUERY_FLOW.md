# 🔄 Complete Query Flow

```
USER ENTERS QUESTION IN STREAMLIT
          ↓
    (POST /agent/answer)
          ↓
┌─────────────────────────────────────────┐
│ BACKEND PROCESSES:                      │
│                                         │
│ 1. Load all docs from CSV               │
│ 2. Build vector DB (Chroma + ST model)  │
│ 3. Semantic search (retrieve TOP_K=3)   │
│ 4. Compute confidence score             │
│                                         │
│ IF confidence >= 0.7:                   │
│   → Use local context only              │
│                                         │
│ ELSE (confidence < 0.7):                │
│   → Trigger web search                  │
│   → Check semantic relevance            │
│   → Combine local + web results         │
│                                         │
│ 5. Generate answer using Ollama LLM     │
│ 6. Return: {answer, confidence,         │
│           used_web_search, chunks}      │
└─────────────────────────────────────────┘
          ↓
    STREAMLIT DISPLAYS:
    - Answer in "Answer" tab
    - Sources in "Sources" tab
    - Metrics in "Details" tab
```

---

## 🚀 How to Run

### Terminal 1: Backend
```powershell
uvicorn ai_agent.backend.server:app --reload
```

### Terminal 2: Frontend
```powershell
streamlit run ai_agent/frontend/streamlit_app.py
```

### Access
- **UI**: http://localhost:8501
- **API Docs**: http://127.0.0.1:8000/docs

---

## 📊 Configuration Constants

Edit in `ai_agent/backend/server.py`:

```python
CHUNK_SIZE = 800           # How big each doc chunk is
CHUNK_OVERLAP = 100        # Overlap between chunks
TOP_K = 3                  # Number of local docs to retrieve
CONFIDENCE_THRESHOLD = 0.7 # Trigger for web search (0-1)
LLM_MODEL = "deepseek-r1"  # Ollama model (change to gpt2, llama2, etc.)
```

---

## 🔧 Customization Examples

### Change LLM Model
```python
LLM_MODEL = "llama2"  # or "mistral", "neural-chat", etc.
# Then: ollama pull mistral
```

### Increase Web Search Trigger Sensitivity
```python
CONFIDENCE_THRESHOLD = 0.5  # Lower = more web searches
```

### Adjust Document Chunking
```python
CHUNK_SIZE = 1200          # Larger chunks = more context
CHUNK_OVERLAP = 50         # Less overlap = faster processing
```

### Switch to Different Embeddings Model
Edit `backend/server.py`:
```python
MODEL_PATH = "sentence-transformers/all-mpnet-base-v2"  # Different model
```

---

## 📁 File Structure

```
ai_agent/
├── backend/
│   ├── __init__.py ................... Package marker
│   ├── server.py ..................... FastAPI app + Agent logic (380 lines)
│   ├── storage.py .................... CSV storage functions (50 lines)
│   ├── data/
│   │   └── docs.csv .................. Document database (auto-created)
│   └── models/sentence-transformer/
│       └── [model files already exist]
│
├── frontend/
│   ├── __init__.py ................... Package marker
│   └── streamlit_app.py .............. Professional UI (300+ lines)
│
├── agent_demo.py ..................... Original agent demo
├── requirements-agent.txt ............ Dependencies
└── README.md ......................... Original readme
```

---

## 🎨 UI Highlights

### Professional Features
- ✅ Gradient theme (blue professional palette)
- ✅ Session state management (persistent conversation)
- ✅ Responsive layout (wide mode)
- ✅ Tabbed results display
- ✅ Real-time metrics (confidence, sources used)
- ✅ Document preview in sidebar
- ✅ Conversation history with timestamps
- ✅ Clear error handling with user-friendly messages
- ✅ Loading spinners for better UX

### Color Scheme
- Primary: `#1e3c72` (dark blue)
- Secondary: `#2c5aa0` (medium blue)
- Accent: `#4a90e2` (light blue)
- Background: Light gradient

---

## 🔍 Agent Intelligence

The agent is **smart** about what it does:

### Scenario 1: High Confidence
```
Q: "What is Python?" (with Python docs stored)
→ Local confidence: 0.85 (HIGH)
→ Uses only local context
→ Direct answer
```

### Scenario 2: Low Confidence
```
Q: "Latest AI news?" (no specific docs)
→ Local confidence: 0.45 (LOW)
→ Triggers web search
→ Combines web + local
→ Enhanced answer
```

### Scenario 3: Web Not Relevant
```
Q: "Custom internal process?" (with web search)
→ Local confidence: 0.35 (LOW)
→ Web search triggered
→ Relevance check fails (web results off-topic)
→ Returns message: "Not enough reliable information"
```

---

## 🚀 Next Steps (Future Enhancements)

### Phase 2: Database
- Replace CSV with MongoDB/PostgreSQL
- Add embeddings caching (speed improvement)

### Phase 3: Advanced Features
- Document upload (PDF, DOCX, TXT)
- Multi-language support
- Fine-tuning on custom data
- User authentication

### Phase 4: Scaling
- Distributed embeddings (FAISS)
- Real-time web search (SerpAPI, Tavily)
- Model serving (vLLM, TGI)

---

## 📚 Documentation

Three levels of documentation provided:

1. **QUICKSTART.md** — Get running in 5 minutes
2. **README_SETUP.md** — Full setup + API reference
3. **This file** — Architecture + customization guide

---

## ✨ Key Takeaways

✅ **Full Stack**: Backend + Frontend + Storage
✅ **Intelligent**: Confidence-based web search fallback
✅ **Professional**: Modern UI with conversation history
✅ **Extensible**: Easy to swap CSV for DB, LLM for LLM
✅ **Local-First**: All embeddings locally (no API calls)
✅ **Production-Ready**: Error handling, logging, docs

---

## 🎉 You're All Set!

Your AIJourney RAG system is ready to use. Start by:

1. Run both terminals (backend + frontend)
2. Add some documents
3. Ask questions
4. Watch the agent retrieve context and generate answers

For detailed setup, see **QUICKSTART.md**

---

**Built with ❤️ by JayPowar | November 2025**
