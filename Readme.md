
# RAG Demo - Retrieval Augmented Generation Chat

A Python application demonstrating Retrieval Augmented Generation (RAG) using LangChain, HuggingFace embeddings, and local LLM inference via Ollama.

## Overview

This project implements a QA system that:
- Chunks and embeds documentation about system modernization
- Stores embeddings in a local Chroma vector database
- Retrieves relevant documents based on user queries
- Generates answers using a local LLM (Phi3) or remote API

## Prerequisites

- Python 3.10+
- pip or conda

## Installation

1. Clone the repository and navigate to the project directory:
```bash
cd rag_demo
```

2. Create and activate a virtual environment:

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies from requirements.txt:
```bash
pip install -r requirements.txt
```

## Setup: Local LLM with Ollama

### Linux Setup

1. Install Ollama:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

2. Pull the Phi3 model:
```bash
ollama pull phi3
```

3. Start Ollama service:
```bash
ollama serve
```

### Windows Setup

1. Download Ollama from [ollama.ai](https://ollama.ai)
2. Install and run the application
3. Open PowerShell and pull the model:
```powershell
ollama pull phi3
```

4. Ollama runs automatically in the background

## Usage

Run the script:
```bash
python rag_demo.py
```

## Using Remote LLM APIs

Replace the `ChatOllama` section with your preferred provider:

### OpenAI
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key="your-api-key",
    model="gpt-4",
    temperature=0.7
)
```

### Hugging Face
```python
from langchain_huggingface import HuggingFaceLLM

llm = HuggingFaceLLM(
    api_key="your-api-key",
    model_id="meta-llama/Llama-2-7b"
)
```

## Configuration

- Adjust `chunk_size` and `chunk_overlap` in `CharacterTextSplitter` for different document chunking
- Modify `temperature` parameter to control LLM randomness
- Change embedding model in `HuggingFaceEmbeddings(model_name="...")`

