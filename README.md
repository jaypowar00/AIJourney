# 🚀 AIJourney – RAG & AI Agent with Web Search (Local + Private)

This repository contains two complete GenAI systems:

### **1. Basic RAG System (Retrieval Augmented Generation)**  
→ Uses LangChain, HuggingFace Embeddings, and Chroma  
→ Answers from **strict local docs only**

### **2. Full AI Agent System (RAG + Web Search + Semantic Safety)**  
→ Adds intelligent agent behavior  
→ Uses Local LLM (Phi-3 via Ollama)  
→ Uses **private SearxNG** as web search tool  
→ Performs semantic filtering to avoid hallucinations  
→ Streams LLM output live

Perfect for **hackathons, AI events, and team learning**.

---

# 📁 Project Structure

```
AIJourney/
│
├── simple_rag/
│   ├── rag_demo.py
│   ├── requirements.txt
│   ├── quick-setup.sh
│   └── README.md
│
├── ai_agent/
│   ├── agent_demo.py
│   ├── requirements-agent.txt
│   ├── searxng/
│   │   └── docker-compose.yaml
│   ├── quick-setup.sh
│   └── README.md
│
└── README.md
```

---

# 🧠 1. Basic RAG Demo

Simple Retrieval-Augmented Generation pipeline:
- Embeds documents using MiniLM
- Stores vectors in ChromaDB
- Retrieves relevant chunks
- Generates answers using a local LLM (Phi-3)

Run:

```bash
cd rag_demo
pip install -r requirements.txt
python rag_demo.py
```

---

# 🤖 2. Full AI Agent (RAG + Search Tool)

The Agent:
- Uses RAG first (offline)
- If confidence < threshold → **triggers web search**
- Uses private SearxNG instance
- Applies semantic similarity filtering
- Combines local + web context
- Streams LLM output live
- Avoids hallucinations

Run:

```bash
cd ai_agent
pip install -r requirements-agent.txt
python agent_demo.py
```

---

# 🔍 Setting up SearxNG (Local Search Engine)

```
cd ai_agent/searxng
docker compose up -d
```

Open JSON API:

http://localhost:8080/search?q=hello&format=json

---

# 🐙 Setting up Ollama (Local LLM)

### Linux:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull phi3
ollama serve
```

### Windows:
- Install from https://ollama.ai  
- Pull the model:
```powershell
ollama pull phi3
```

---

# 🎯 Summary

This repository showcases:
- Basic RAG  
- Advanced AI Agent  
- Hybrid Retrieval + Search  
- Local inference  
- Private web search  
- Semantic hallucination prevention  

Ready for AI competitions and learning.
