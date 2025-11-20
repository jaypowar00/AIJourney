import streamlit as st
import requests
from datetime import datetime

# ================================================================
# Page Setup
# ================================================================
st.set_page_config(
    page_title="AIJourney — AI-Powered RAG",
    page_icon="🤖",
    layout="wide",
)

BASE_API = "http://127.0.0.1:8000"


# ================================================================
# Session State Initialization
# ================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "docs_cache" not in st.session_state:
    st.session_state.docs_cache = []


# ================================================================
# API Helpers
# ================================================================
def load_documents():
    try:
        r = requests.get(f"{BASE_API}/documents", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"❌ Failed to load documents: {e}")
        return []


def save_new_document(content: str):
    try:
        r = requests.post(
            f"{BASE_API}/documents",
            json={"content": content},
            timeout=10
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"❌ Failed to save document: {e}")
        return None


def query_agent(query: str, use_web: bool):
    try:
        r = requests.post(
            f"{BASE_API}/agent/answer",
            json={"query": query, "use_web_search": use_web},
            timeout=240,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"❌ Request failed: {e}")
        return None


# ================================================================
# Header
# ================================================================
st.markdown("# 🤖 AIJourney")
st.markdown("<p style='text-align:center; color:#bbb;'>AI with Local Docs + Web Search</p>", unsafe_allow_html=True)
st.divider()


# ================================================================
# Sidebar – Document management
# ================================================================
with st.sidebar:
    st.markdown("### 📚 Documents")

    with st.expander("➕ Add Document"):
        new_doc = st.text_area("Content", height=120)
        if st.button("💾 Save Document"):
            if new_doc.strip():
                save_new_document(new_doc)
                st.success("Saved!")
                st.rerun()
            else:
                st.warning("Enter content first.")

    st.divider()

    docs = load_documents()
    st.session_state.docs_cache = docs

    st.markdown(f"Stored Documents ({len(docs)})")

    if docs:
        for d in docs:
            with st.expander(d["id"][:10] + "..."):
                st.text_area("Content", d["content"], height=150, disabled=True)
    else:
        st.info("No documents yet.")


# ================================================================
# Main – Ask AI
# ================================================================
st.markdown("### 🔍 Ask the AI Agent")

col1, col2 = st.columns([5, 1])
with col1:
    query_text = st.text_input("Your Question:", key="q")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    web_enabled = st.checkbox("Web Search", value=True)


# ================================================================
# Button
# ================================================================
if st.button("🚀 Ask Agent"):
    if not query_text.strip():
        st.warning("⚠ Please enter a question")
        st.stop()

    # Show only spinner, no status messages
    with st.spinner("⏳ Contacting AI agent…"):
        result = query_agent(query_text, web_enabled)

    if not result:
        st.error("❌ No response received")
        st.stop()

    # Final Answer only
    st.subheader("🧠 Final Answer")
    st.markdown(result.get("answer", "No answer returned"))

    # Save minimal history
    st.session_state.messages.append({
        "query": query_text,
        "answer": result.get("answer"),
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })


# ================================================================
# Footer
# ================================================================
st.divider()
st.markdown(
    "<p style='text-align:center; color:#888;'>AIJourney v1.0 — FastAPI + Ollama</p>",
    unsafe_allow_html=True,
)
