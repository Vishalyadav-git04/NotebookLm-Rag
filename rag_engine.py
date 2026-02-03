"""
RAG Engine for NotebookLM-like PDF Question Answering
FINAL STABLE VERSION - Fixes multi-document retrieval issues
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from unittest.mock import MagicMock

# --- FIX 1: BYPASS BROKEN WINDOWS LIBRARY ---
sys.modules["onnxruntime"] = MagicMock()
sys.modules["onnxruntime.quantization"] = MagicMock()
sys.modules["onnxruntime.capi"] = MagicMock()
sys.modules["onnxruntime.capi._pybind_state"] = MagicMock()

# --- CRITICAL CONFIG: Disable ChromaDB's default embedding ---
os.environ["CHROMA_ENABLE_DEFAULT_EMBEDDING"] = "false"

# Import other dependencies
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self, google_api_key: str, persist_directory: str = "./chroma_db"):
        self.google_api_key = google_api_key
        self.persist_directory = persist_directory
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        
        os.environ["GOOGLE_API_KEY"] = google_api_key
        
        logger.info("Initializing RAG Engine...")
        
        # --- 1. EMBEDDINGS (High Quality) ---
        try:
            logger.info("Creating embeddings with models/text-embedding-004...")
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=google_api_key,
                task_type="retrieval_document"
            )
            logger.info("✅ Embeddings initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize embeddings: {str(e)}")
            raise ValueError(f"Embedding initialization failed: {str(e)}")
        
        # --- 2. LLM (Stable Model) ---
        try:
            logger.info("Creating Gemini LLM...")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",  # ✅ STABLE MODEL
                google_api_key=google_api_key,
                temperature=0.3,
                convert_system_message_to_human=True
            )
            logger.info("✅ LLM initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {str(e)}")
            raise ValueError(f"LLM initialization failed: {str(e)}")
        
        # --- 3. CHUNKING ---
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        logger.info("✅ RAG Engine initialized successfully")
    
    def ingest_pdfs(self, pdf_files: List[Any]) -> Dict[str, Any]:
        """Ingest PDFs with garbage filtering"""
        try:
            logger.info(f"Starting PDF ingestion for {len(pdf_files)} files...")
            all_documents = []
            pdf_stats = {}
            
            temp_dir = Path("./temp_pdfs")
            temp_dir.mkdir(exist_ok=True)
            
            for idx, pdf_file in enumerate(pdf_files, 1):
                try:
                    logger.info(f"[{idx}/{len(pdf_files)}] Processing: {pdf_file.name}")
                    
                    temp_path = temp_dir / pdf_file.name
                    with open(temp_path, "wb") as f:
                        f.write(pdf_file.getvalue())
                    
                    loader = PyPDFLoader(str(temp_path))
                    documents = loader.load()
                    
                    if not documents:
                        pdf_stats[pdf_file.name] = "0 pages (Empty/Scanned)"
                        continue
                    
                    # Clean metadata and content
                    for doc_idx, doc in enumerate(documents):
                        doc.metadata["source_file"] = pdf_file.name
                        doc.metadata["page"] = doc_idx + 1 
                        doc.page_content = " ".join(doc.page_content.split())
                    
                    all_documents.extend(documents)
                    pdf_stats[pdf_file.name] = len(documents)
                    logger.info(f"  ✅ Loaded {len(documents)} pages")
                    
                except Exception as e:
                    logger.error(f"  ❌ Error loading {pdf_file.name}: {str(e)}")
                    pdf_stats[pdf_file.name] = f"Error: {str(e)}"
            
            if not all_documents:
                return {"status": "error", "message": "No readable text found."}
            
            # Split into chunks
            raw_chunks = self.text_splitter.split_documents(all_documents)
            
            # Filter junk chunks
            chunks = []
            for chunk in raw_chunks:
                if len(chunk.page_content) > 50:
                    chunks.append(chunk)
            
            logger.info(f"✅ Created {len(chunks)} valid chunks")
            
            if not chunks:
                return {"status": "error", "message": "No valid text chunks after filtering."}
            
            # Create vector store
            try:
                if self.vector_store is None:
                    self.vector_store = Chroma.from_documents(
                        documents=chunks,
                        embedding=self.embeddings,
                        persist_directory=self.persist_directory,
                        collection_name="rag_documents"
                    )
                else:
                    self.vector_store.add_documents(chunks)
                
            except Exception as e:
                raise ValueError(f"ChromaDB error: {str(e)}")
            
            # --- CRITICAL FIX: USE MMR SEARCH ---
            # MMR ensures diversity so one big doc doesn't dominate results
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr",  # ✅ Changed from similarity to MMR
                search_kwargs={
                    "k": 15,            # ✅ Retrieve 15 chunks (was 6)
                    "fetch_k": 50,      # ✅ Look at top 50 candidates first
                    "lambda_mult": 0.7  # ✅ Balance diversity vs relevance
                }
            )
            
            self._create_qa_chain()
            
            return {
                "status": "success",
                "total_pages": len(all_documents),
                "total_chunks": len(chunks),
                "pdf_stats": pdf_stats
            }
            
        except Exception as e:
            logger.error(f"❌ FATAL ERROR: {str(e)}")
            raise ValueError(f"Processing failed: {str(e)}")
    
    def _create_qa_chain(self):
        """Create QA chain with DETAILED system prompt"""
        system_prompt = """You are an expert research assistant. Your task is to provide a comprehensive, detailed answer based ONLY on the provided Context.

CRITICAL RULES:
1. **BE DETAILED:** Do not write short summaries. Explain the concepts fully.
2. **USE STRUCTURE:** Use bullet points, headers, and paragraphs.
3. **CITE SOURCES:** Always format citations as [Filename:Page X] immediately after the information.
4. **NO HALLUCINATION:** If the answer is not in the context, say "I cannot answer this based on the documents."
5. **LOOK CAREFULLY:** The context may contain information from multiple different files. Check the 'Filename' in the sources carefully.

FORMAT:
**Step-by-Step Reasoning:**
[Briefly explain how you found the answer]

**Detailed Answer:**
[Your long, comprehensive answer here]

**Sources:**
- Filename:Page X - [Brief note]

Context:
{context}

Question: {input}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        combine_docs_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt
        )
        
        self.qa_chain = create_retrieval_chain(
            retriever=self.retriever,
            combine_docs_chain=combine_docs_chain
        )
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        if self.qa_chain is None:
            return {"status": "error", "answer": "Please process PDFs first."}
        
        try:
            response = self.qa_chain.invoke({"input": question})
            retrieved_docs = response.get("context", [])
            answer_text = response.get("answer", "")
            
            sources = []
            seen = set()
            for doc in retrieved_docs:
                key = f"{doc.metadata.get('source_file')}:{doc.metadata.get('page')}"
                if key not in seen:
                    sources.append({
                        "file": doc.metadata.get("source_file"),
                        "page": doc.metadata.get("page"),
                        "excerpt": doc.page_content[:150] + "..."
                    })
                    seen.add(key)
            
            return {
                "status": "success",
                "answer": answer_text,
                "sources": sources
            }
            
        except Exception as e:
            return {"status": "error", "answer": f"Error: {str(e)}"}

    def clear_database(self):
        try:
            if self.vector_store:
                self.vector_store = None
                self.retriever = None
            
            import shutil
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
            return {"status": "success"}
        except Exception as e:
            return {"status": "error", "message": str(e)}