"""
NotebookLM-like RAG Application - FULLY FIXED
"""

import os
import sys

# Disable default embeddings for Chroma
os.environ["CHROMA_ENABLE_DEFAULT_EMBEDDING"] = "false"

import sqlite3
sys.modules['pysqlite3'] = sqlite3

import streamlit as st
from rag_engine import RAGEngine
import time

st.set_page_config(
    page_title="NotebookLM-like RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1E88E5; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #666; margin-bottom: 2rem; }
    .source-box { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    .citation { background-color: #e3f2fd; padding: 0.2rem 0.5rem; border-radius: 0.3rem; font-weight: bold; color: #1565c0; }
    .reasoning-box { background-color: #fff3e0; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; border-left: 4px solid #ff9800; }
    </style>
""", unsafe_allow_html=True)

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdfs_loaded" not in st.session_state:
    st.session_state.pdfs_loaded = False
if "pdf_stats" not in st.session_state:
    st.session_state.pdf_stats = {}

with st.sidebar:
    st.markdown("### 🔑 Configuration")
    api_key = st.text_input("Google API Key", type="password")
    
    if api_key and st.session_state.rag_engine is None:
        try:
            st.session_state.rag_engine = RAGEngine(google_api_key=api_key)
            st.success("✅ RAG Engine initialized!")
        except Exception as e:
            st.error(f"❌ {str(e)}")
    
    st.markdown("---")
    st.markdown("### 📄 Upload PDFs")
    
    uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files and st.session_state.rag_engine:
        if len(uploaded_files) < 10:
            st.warning(f"⚠️ Uploaded {len(uploaded_files)} PDFs. 10+ recommended.")
        
        if st.button("🚀 Process PDFs", type="primary"):
            with st.spinner("Processing..."):
                try:
                    progress = st.progress(0)
                    status = st.empty()
                    
                    status.text("Loading PDFs...")
                    progress.progress(25)
                    
                    result = st.session_state.rag_engine.ingest_pdfs(uploaded_files)
                    
                    status.text("Creating embeddings...")
                    progress.progress(75)
                    time.sleep(0.5)
                    progress.progress(100)
                    
                    st.session_state.pdfs_loaded = True
                    st.session_state.pdf_stats = result
                    
                    status.empty()
                    progress.empty()
                    
                    st.success(f"✅ {result['total_pages']} pages → {result['total_chunks']} chunks!")
                    
                    with st.expander("📊 Details"):
                        for name, count in result['pdf_stats'].items():
                            st.write(f"- **{name}**: {count} pages")
                except Exception as e:
                    st.error(f"❌ {str(e)}")
    
    if st.session_state.pdfs_loaded:
        st.markdown("---")
        if st.button("🗑️ Clear All"):
            st.session_state.rag_engine.clear_database()
            st.session_state.pdfs_loaded = False
            st.session_state.chat_history = []
            st.session_state.pdf_stats = {}
            st.rerun()

st.markdown('<p class="main-header">📚 NotebookLM-like RAG System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Citation-grounded answers from your PDFs</p>', unsafe_allow_html=True)

if not api_key:
    st.info("👈 Enter Google API Key to start")
elif not st.session_state.pdfs_loaded:
    st.info("👈 Upload and process PDFs")
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 PDFs", len(st.session_state.pdf_stats.get('pdf_stats', {})))
    with col2:
        st.metric("📃 Pages", st.session_state.pdf_stats.get('total_pages', 0))
    with col3:
        st.metric("🧩 Chunks", st.session_state.pdf_stats.get('total_chunks', 0))
    
    st.markdown("---")
    st.markdown("### 💬 Ask Questions")
    
    # --- UPDATED CHAT HISTORY LOGIC START ---
    for i, chat in enumerate(st.session_state.chat_history):
        st.markdown(f"**🙋 Q{i+1}:** {chat['question']}")
        
        answer = chat['answer']
        
        # Check for reasoning block
        if "**Step-by-Step Reasoning:**" in answer:
            # Try splitting by "Detailed Answer" first (new format)
            if "**Detailed Answer:**" in answer:
                parts = answer.split("**Detailed Answer:**")
            # Fallback to "Answer" (old format)
            elif "**Answer:**" in answer:
                parts = answer.split("**Answer:**")
            else:
                parts = [answer]  # Fallback if neither found
            
            # If split was successful
            if len(parts) > 1:
                reasoning = parts[0].replace("**Step-by-Step Reasoning:**", "").strip()
                remaining = parts[1]
                
                if "**Sources:**" in remaining:
                    ans_parts = remaining.split("**Sources:**")
                    ans = ans_parts[0].strip()
                    srcs = ans_parts[1].strip()
                else:
                    ans = remaining.strip()
                    srcs = ""
                
                st.markdown(f'<div class="reasoning-box"><b>🧠 Reasoning:</b><br>{reasoning}</div>', unsafe_allow_html=True)
                st.markdown(f"**✅ Detailed Answer:**\n{ans}")
                if srcs:
                    st.markdown(f"**📚 Sources:**\n{srcs}")
            else:
                # Fallback if format is weird but reasoning tag exists
                st.markdown(f"**✅ Answer:**\n{answer}")
        else:
            # Fallback for standard answers without reasoning
            st.markdown(f"**✅ Answer:**\n{answer}")
        
        if chat['sources']:
            with st.expander(f"📎 Context ({len(chat['sources'])} sources)"):
                for idx, s in enumerate(chat['sources'], 1):
                    st.markdown(f'<div class="source-box"><b>{idx}. <span class="citation">{s["file"]}:P{s["page"]}</span></b><br><small>{s["excerpt"]}</small></div>', unsafe_allow_html=True)
        st.markdown("---")
    # --- UPDATED CHAT HISTORY LOGIC END ---
    
    question = st.text_input("Your question:", placeholder="What are the main topics?")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask = st.button("🔍 Ask", type="primary")
    with col2:
        if st.button("🧹 Clear"):
            st.session_state.chat_history = []
            st.rerun()
    
    if ask and question:
        with st.spinner("🤔 Analyzing..."):
            result = st.session_state.rag_engine.ask_question(question)
            if result['status'] == 'success':
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': result['answer'],
                    'sources': result['sources']
                })
                st.rerun()
            else:
                st.error(f"❌ {result['answer']}")
    
    if not st.session_state.chat_history:
        st.markdown("### 💡 Suggested:")
        sug = [
            "What are the main topics covered?",
            "Summarize key findings",
            "What methodologies are discussed?",
            "Compare conclusions across documents"
        ]
        
        cols = st.columns(2)
        for i, s in enumerate(sug):
            with cols[i % 2]:
                if st.button(s, key=f"sug{i}"):
                    with st.spinner("Analyzing..."):
                        r = st.session_state.rag_engine.ask_question(s)
                        if r['status'] == 'success':
                            st.session_state.chat_history.append({'question': s, 'answer': r['answer'], 'sources': r['sources']})
                            st.rerun()