"""
RAG Engine for NotebookLM-like PDF Question Answering
FULLY FIXED VERSION - All model names corrected
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
        """Initialize RAG Engine with correct model names"""
        self.google_api_key = google_api_key
        self.persist_directory = persist_directory
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        
        os.environ["GOOGLE_API_KEY"] = google_api_key
        
        logger.info("Initializing RAG Engine...")
        
        # --- FIX: USE CORRECT EMBEDDING MODEL ---
        try:
            logger.info("Creating embeddings with models/embedding-001...")
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",  # ✅ CORRECT
                google_api_key=google_api_key,
                task_type="retrieval_document"
            )
            logger.info("✅ Embeddings initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize embeddings: {str(e)}")
            raise ValueError(f"Embedding initialization failed: {str(e)}")
        
        # --- FIX: USE CORRECT LLM MODEL NAME ---
        try:
            logger.info("Creating Gemini LLM...")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest",  # ✅ CORRECT (not gemini-1.5-flash)
                google_api_key=google_api_key,
                temperature=0.1,
                convert_system_message_to_human=True
            )
            logger.info("✅ LLM initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {str(e)}")
            raise ValueError(f"LLM initialization failed: {str(e)}")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        logger.info("✅ RAG Engine initialized successfully")
    
    def ingest_pdfs(self, pdf_files: List[Any]) -> Dict[str, Any]:
        """Ingest PDFs with detailed logging"""
        try:
            logger.info(f"Starting PDF ingestion for {len(pdf_files)} files...")
            all_documents = []
            pdf_stats = {}
            
            # Create temporary directory
            temp_dir = Path("./temp_pdfs")
            temp_dir.mkdir(exist_ok=True)
            logger.info(f"Temp directory: {temp_dir}")
            
            # Process each PDF
            for idx, pdf_file in enumerate(pdf_files, 1):
                try:
                    logger.info(f"[{idx}/{len(pdf_files)}] Processing: {pdf_file.name}")
                    
                    # Save file
                    temp_path = temp_dir / pdf_file.name
                    with open(temp_path, "wb") as f:
                        f.write(pdf_file.getvalue())
                    logger.info(f"  → Saved to: {temp_path}")
                    
                    # Load PDF
                    loader = PyPDFLoader(str(temp_path))
                    documents = loader.load()
                    logger.info(f"  → Extracted {len(documents)} pages")
                    
                    if not documents:
                        logger.warning(f"  ⚠️ No text extracted from {pdf_file.name}")
                        pdf_stats[pdf_file.name] = "0 pages (Empty/Scanned)"
                        continue
                    
                    # Add metadata
                    for doc_idx, doc in enumerate(documents):
                        doc.metadata["source_file"] = pdf_file.name
                        if "page" not in doc.metadata:
                            doc.metadata["page"] = doc_idx
                        logger.debug(f"    Page {doc_idx + 1}: {len(doc.page_content)} chars")
                    
                    all_documents.extend(documents)
                    pdf_stats[pdf_file.name] = len(documents)
                    logger.info(f"  ✅ Successfully loaded {len(documents)} pages")
                    
                except Exception as e:
                    logger.error(f"  ❌ Error loading {pdf_file.name}: {str(e)}")
                    pdf_stats[pdf_file.name] = f"Error: {str(e)}"
            
            # Check if we got any documents
            if not all_documents:
                logger.error("❌ No documents were successfully loaded")
                return {
                    "status": "error",
                    "total_pages": 0,
                    "total_chunks": 0,
                    "pdf_stats": pdf_stats,
                    "message": "No readable text found. PDFs might be scanned images or corrupted."
                }
            
            logger.info(f"✅ Total documents loaded: {len(all_documents)} pages")
            
            # Split into chunks
            logger.info("Splitting documents into chunks...")
            chunks = self.text_splitter.split_documents(all_documents)
            logger.info(f"✅ Created {len(chunks)} chunks")
            
            if not chunks:
                logger.error("❌ No chunks created (text splitting failed)")
                return {
                    "status": "error",
                    "total_pages": len(all_documents),
                    "total_chunks": 0,
                    "pdf_stats": pdf_stats,
                    "message": "Text chunking failed. Try different PDFs."
                }
            
            # Create embeddings and store in ChromaDB
            logger.info("Creating vector store with embeddings...")
            try:
                if self.vector_store is None:
                    logger.info("Creating NEW ChromaDB collection...")
                    self.vector_store = Chroma.from_documents(
                        documents=chunks,
                        embedding=self.embeddings,
                        persist_directory=self.persist_directory,
                        collection_name="rag_documents"
                    )
                    logger.info("✅ Vector store created")
                else:
                    logger.info("Adding to EXISTING ChromaDB collection...")
                    self.vector_store.add_documents(chunks)
                    logger.info("✅ Documents added to existing store")
                
            except Exception as e:
                logger.error(f"❌ Vector store creation failed: {str(e)}")
                raise ValueError(f"ChromaDB error: {str(e)}")
            
            # Create retriever
            logger.info("Creating retriever...")
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6}
            )
            logger.info("✅ Retriever created")
            
            # Create QA chain
            logger.info("Creating QA chain...")
            self._create_qa_chain()
            logger.info("✅ QA chain created")
            
            result = {
                "status": "success",
                "total_pages": len(all_documents),
                "total_chunks": len(chunks),
                "pdf_stats": pdf_stats
            }
            
            logger.info(f"🎉 PDF ingestion complete! {len(chunks)} chunks ready for queries")
            return result
            
        except Exception as e:
            logger.error(f"❌ FATAL ERROR during PDF ingestion: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise ValueError(f"Processing failed: {str(e)}")
    
    def _create_qa_chain(self):
        """Create QA chain with detailed prompt"""
        system_prompt = """You are an expert research assistant analyzing uploaded PDF documents.

CRITICAL RULES:
1. Answer ONLY based on the provided context from the PDFs
2. If the answer is not in the context, say "I cannot answer this based on the provided documents"
3. ALWAYS cite your sources using [Filename:Page X] format
4. Provide step-by-step reasoning for your answer
5. For multi-document questions, synthesize information and cite all relevant sources
6. Never make up information or use external knowledge

FORMAT YOUR RESPONSE AS:

**Step-by-Step Reasoning:**
[Explain your thought process step by step]

**Answer:**
[Your complete answer with inline citations like [Filename:Page X]]

**Sources:**
- Filename:Page X - [Brief description of what this source contributed]
- Filename:Page Y - [Brief description]

Context from PDFs:
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
        
        logger.info("QA chain created successfully")
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask question with detailed logging"""
        if self.qa_chain is None:
            logger.warning("⚠️ Question asked but no documents loaded")
            return {
                "status": "error",
                "question": question,
                "answer": "Please upload and process PDFs first before asking questions",
                "sources": [],
                "num_sources": 0
            }
        
        try:
            logger.info(f"Processing question: {question}")
            
            # Invoke chain
            response = self.qa_chain.invoke({"input": question})
            retrieved_docs = response.get("context", [])
            answer_text = response.get("answer", "")
            
            logger.info(f"✅ Retrieved {len(retrieved_docs)} relevant chunks")
            logger.info(f"✅ Generated answer: {len(answer_text)} chars")
            
            # Extract sources
            sources = []
            seen_sources = set()
            
            for doc in retrieved_docs:
                source_file = doc.metadata.get("source_file", "Unknown")
                page = doc.metadata.get("page", 0) + 1
                source_key = f"{source_file}:Page {page}"
                
                if source_key not in seen_sources:
                    sources.append({
                        "file": source_file,
                        "page": page,
                        "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    })
                    seen_sources.add(source_key)
            
            logger.info(f"✅ Extracted {len(sources)} unique sources")
            
            return {
                "status": "success",
                "question": question,
                "answer": answer_text,
                "sources": sources,
                "num_sources": len(sources)
            }
            
        except Exception as e:
            logger.error(f"❌ Error during question answering: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "question": question,
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "num_sources": 0
            }

    def clear_database(self):
        """Clear database with logging"""
        try:
            logger.info("Clearing database...")
            
            if self.vector_store is not None:
                self.vector_store = None
                self.retriever = None
                self.qa_chain = None
            
            import shutil
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                logger.info(f"✅ Removed directory: {self.persist_directory}")
            
            logger.info("✅ Database cleared successfully")
            return {"status": "success", "message": "Database cleared"}
            
        except Exception as e:
            logger.error(f"❌ Error clearing database: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database stats"""
        if self.vector_store is None:
            return {
                "status": "empty",
                "total_documents": 0,
                "message": "No documents loaded"
            }
        
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            logger.info(f"Database stats: {count} chunks")
            
            return {
                "status": "ready",
                "total_documents": count,
                "message": f"Ready with {count} document chunks"
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }