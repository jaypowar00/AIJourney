# 🧠 Basic RAG Demo

A simple Retrieval Augmented Generation (RAG) system using:
- LangChain
- HuggingFace MiniLM embeddings
- Chroma vector database
- Local LLM (Phi-3 via Ollama)

This script demonstrates:
1. Splitting documents into chunks  
2. Embedding text vectors  
3. Performing retrieval  
4. Answering using a local language model  

---

# 📦 Installation

```bash
cd rag_demo
python3 -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

# 🐙 Setup: Local LLM with Ollama

### Install Ollama
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Pull Phi-3 model
```bash
ollama pull phi3
```

### Start LLM server
```bash
ollama serve
```

---

# ▶️ Run the Demo

```bash
python rag_demo.py
```

---

# ⚙️ Configuration Options

Inside `rag_demo.py`:

| Component | Purpose |
|----------|---------|
| `CharacterTextSplitter` | Chunking strategy |
| `HuggingFaceEmbeddings` | Embedding model |
| `Chroma.from_texts` | Vector storage |
| `ChatOllama` / `ollama.chat` | LLM inference |
| `temperature` | Randomness control |

---

# 📘 Notes

This simple RAG pipeline serves as a foundation for more advanced agents, such as the one in the `ai_agent` folder.

