"""
NotebookLM-like RAG Application
Streamlit interface for multi-PDF question answering with citations
"""

import streamlit as st
from rag_engine import RAGEngine
import time
from typing import List

# Page configuration
st.set_page_config(
    page_title="NotebookLM-like RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .citation {
        background-color: #e3f2fd;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
        color: #1565c0;
    }
    .reasoning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #ff9800;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdfs_loaded" not in st.session_state:
    st.session_state.pdfs_loaded = False

if "pdf_stats" not in st.session_state:
    st.session_state.pdf_stats = {}

# Sidebar for API Key and PDF Upload
with st.sidebar:
    st.markdown("### 🔑 Configuration")
    
    # API Key input
    api_key = st.text_input(
        "Google API Key",
        type="password",
        help="Enter your Google API key for Gemini models. Get one at https://makersuite.google.com/app/apikey"
    )
    
    if api_key:
        if st.session_state.rag_engine is None:
            try:
                st.session_state.rag_engine = RAGEngine(google_api_key=api_key)
                st.success("✅ RAG Engine initialized!")
            except Exception as e:
                st.error(f"❌ Error initializing engine: {str(e)}")
    
    st.markdown("---")
    st.markdown("### 📄 Upload PDFs")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files (minimum 10 recommended)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload at least 10 PDFs for best results"
    )
    
    if uploaded_files and st.session_state.rag_engine:
        if len(uploaded_files) < 10:
            st.warning(f"⚠️ You've uploaded {len(uploaded_files)} PDFs. For NotebookLM-like experience, upload at least 10 PDFs.")
        
        if st.button("🚀 Process PDFs", type="primary"):
            with st.spinner("Processing PDFs... This may take a few minutes..."):
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Loading and parsing PDFs...")
                    progress_bar.progress(25)
                    
                    result = st.session_state.rag_engine.ingest_pdfs(uploaded_files)
                    
                    status_text.text("Creating embeddings...")
                    progress_bar.progress(75)
                    
                    time.sleep(0.5)  # Brief pause for UX
                    progress_bar.progress(100)
                    
                    st.session_state.pdfs_loaded = True
                    st.session_state.pdf_stats = result
                    
                    status_text.empty()
                    progress_bar.empty()
                    
                    st.success(f"✅ Processed {result['total_pages']} pages into {result['total_chunks']} chunks!")
                    
                    # Show stats
                    with st.expander("📊 Processing Details"):
                        for pdf_name, page_count in result['pdf_stats'].items():
                            st.write(f"- **{pdf_name}**: {page_count} pages")
                    
                except Exception as e:
                    st.error(f"❌ Error processing PDFs: {str(e)}")
    
    # Clear database option
    if st.session_state.pdfs_loaded:
        st.markdown("---")
        if st.button("🗑️ Clear All Documents"):
            if st.session_state.rag_engine:
                st.session_state.rag_engine.clear_database()
                st.session_state.pdfs_loaded = False
                st.session_state.chat_history = []
                st.session_state.pdf_stats = {}
                st.success("✅ Database cleared!")
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <small>
    <b>NotebookLM-like RAG System</b><br>
    Powered by:<br>
    • Gemini 1.5 Flash<br>
    • LangChain<br>
    • ChromaDB<br>
    </small>
    """, unsafe_allow_html=True)

# Main content area
st.markdown('<p class="main-header">📚 NotebookLM-like RAG System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about your uploaded PDFs with citation-grounded answers</p>', unsafe_allow_html=True)

# Check if system is ready
if not api_key:
    st.info("👈 Please enter your Google API Key in the sidebar to get started")
    st.markdown("""
    ### How to use:
    1. Enter your Google API Key (get one [here](https://makersuite.google.com/app/apikey))
    2. Upload at least 10 PDF documents
    3. Click "Process PDFs" to index them
    4. Ask questions and get citation-backed answers!
    
    ### Features:
    - 🎯 **Strict Grounding**: Answers only from your documents
    - 📖 **Citations**: Every claim is cited with filename and page number
    - 🧠 **Reasoning**: Step-by-step thought process shown
    - 📚 **Multi-Document**: Synthesize information across all PDFs
    - 🚫 **No Hallucinations**: Refuses to answer if info isn't in documents
    """)
    
elif not st.session_state.pdfs_loaded:
    st.info("👈 Please upload and process your PDF documents to start asking questions")
    
    # Show example questions
    st.markdown("""
    ### Example Questions:
    Once you upload your documents, you can ask questions like:
    - "What are the main findings in the research papers?"
    - "Compare the methodologies discussed across different documents"
    - "Summarize the key recommendations from all reports"
    - "What does [Document Name] say about [Topic]?"
    """)

else:
    # Show current stats
    stats = st.session_state.rag_engine.get_stats()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 PDFs Loaded", len(st.session_state.pdf_stats.get('pdf_stats', {})))
    with col2:
        st.metric("📃 Total Pages", st.session_state.pdf_stats.get('total_pages', 0))
    with col3:
        st.metric("🧩 Document Chunks", st.session_state.pdf_stats.get('total_chunks', 0))
    
    st.markdown("---")
    
    # Chat interface
    st.markdown("### 💬 Ask Questions")
    
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**🙋 Question {i+1}:** {chat['question']}")
            
            # Answer section
            answer_text = chat['answer']
            
            # Try to parse structured response
            if "**Step-by-Step Reasoning:**" in answer_text and "**Answer:**" in answer_text:
                parts = answer_text.split("**Answer:**")
                reasoning = parts[0].replace("**Step-by-Step Reasoning:**", "").strip()
                
                if "**Sources:**" in parts[1]:
                    answer_parts = parts[1].split("**Sources:**")
                    answer = answer_parts[0].strip()
                    sources_text = answer_parts[1].strip() if len(answer_parts) > 1 else ""
                else:
                    answer = parts[1].strip()
                    sources_text = ""
                
                # Display reasoning
                st.markdown(f'<div class="reasoning-box"><b>🧠 Step-by-Step Reasoning:</b><br>{reasoning}</div>', unsafe_allow_html=True)
                
                # Display answer
                st.markdown(f"**✅ Answer:**")
                st.markdown(answer)
                
                # Display sources from structured response
                if sources_text:
                    st.markdown("**📚 Sources:**")
                    st.markdown(sources_text)
            else:
                # Fallback: display full answer
                st.markdown(f"**✅ Answer:**")
                st.markdown(answer_text)
            
            # Display retrieved sources
            if chat['sources']:
                with st.expander(f"📎 View Retrieved Context ({len(chat['sources'])} sources)"):
                    for idx, source in enumerate(chat['sources'], 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <b>{idx}. <span class="citation">{source['file']} - Page {source['page']}</span></b><br>
                            <small>{source['excerpt']}</small>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("---")
    
    # Question input
    question = st.text_input(
        "Your question:",
        placeholder="e.g., What are the main conclusions from the research papers?",
        key="question_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("🔍 Ask", type="primary")
    with col2:
        if st.button("🧹 Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    if ask_button and question:
        with st.spinner("🤔 Analyzing documents and generating answer..."):
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
    
    # Suggested questions
    if not st.session_state.chat_history:
        st.markdown("### 💡 Suggested Questions:")
        suggestions = [
            "What are the main topics covered in these documents?",
            "Summarize the key findings from all PDFs",
            "What methodologies are discussed?",
            "Compare the conclusions across different documents"
        ]
        
        cols = st.columns(2)
        for idx, suggestion in enumerate(suggestions):
            with cols[idx % 2]:
                if st.button(suggestion, key=f"suggestion_{idx}"):
                    st.session_state.question_input = suggestion
                    st.rerun()