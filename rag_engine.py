"""
RAG Engine for NotebookLM-like PDF Question Answering
Handles PDF ingestion, chunking, embedding, retrieval, and citation-grounded answers.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Production-ready RAG Engine using LangChain, ChromaDB, and Gemini.
    
    Features:
    - Multi-PDF ingestion with metadata tracking
    - Semantic chunking with overlap
    - Gemini embeddings (models/embedding-001)
    - ChromaDB vector storage (persistent)
    - Citation-grounded answers with reasoning
    """
    
    def __init__(self, google_api_key: str, persist_directory: str = "./chroma_db"):
        """
        Initialize RAG Engine.
        
        Args:
            google_api_key: Google API key for Gemini models
            persist_directory: Directory for ChromaDB persistence
        """
        self.google_api_key = google_api_key
        self.persist_directory = persist_directory
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        
        # Set API key in environment
        os.environ["GOOGLE_API_KEY"] = google_api_key
        
        # Initialize embeddings model
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        
        # Initialize LLM (Gemini Flash)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            google_api_key=google_api_key,
            temperature=0.1,  # Low temperature for factual accuracy
            convert_system_message_to_human=True
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Optimal for context window
            chunk_overlap=200,  # Preserve context across chunks
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        logger.info("RAG Engine initialized successfully")
    
    def ingest_pdfs(self, pdf_files: List[Any]) -> Dict[str, Any]:
        """
        Ingest multiple PDF files into the vector store.
        
        Args:
            pdf_files: List of uploaded PDF file objects (Streamlit UploadedFile)
            
        Returns:
            Dictionary with ingestion statistics
        """
        try:
            all_documents = []
            pdf_stats = {}
            
            # Create temporary directory for PDF processing
            temp_dir = Path("./temp_pdfs")
            temp_dir.mkdir(exist_ok=True)
            
            for pdf_file in pdf_files:
                try:
                    # Save uploaded file temporarily
                    temp_path = temp_dir / pdf_file.name
                    with open(temp_path, "wb") as f:
                        f.write(pdf_file.getvalue())
                    
                    # Load PDF with page metadata
                    loader = PyPDFLoader(str(temp_path))
                    documents = loader.load()
                    
                    # Add source filename to metadata
                    for doc in documents:
                        doc.metadata["source_file"] = pdf_file.name
                        # Ensure page number is in metadata (0-indexed by PyPDFLoader)
                        if "page" not in doc.metadata:
                            doc.metadata["page"] = 0
                    
                    all_documents.extend(documents)
                    pdf_stats[pdf_file.name] = len(documents)
                    
                    logger.info(f"Loaded {len(documents)} pages from {pdf_file.name}")
                    
                except Exception as e:
                    logger.error(f"Error loading {pdf_file.name}: {str(e)}")
                    pdf_stats[pdf_file.name] = f"Error: {str(e)}"
            
            if not all_documents:
                raise ValueError("No documents were successfully loaded")
            
            # Split documents into chunks
            logger.info(f"Splitting {len(all_documents)} pages into chunks...")
            chunks = self.text_splitter.split_documents(all_documents)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Create or update vector store
            if self.vector_store is None:
                logger.info("Creating new ChromaDB vector store...")
                
                # Use Chroma with explicit collection name
                self.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                    collection_name="rag_documents"
                )
            else:
                logger.info("Adding to existing vector store...")
                self.vector_store.add_documents(chunks)
            
            # Initialize retriever with high k for better context
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6}  # Retrieve top 6 most relevant chunks
            )
            
            # Create QA chain
            self._create_qa_chain()
            
            return {
                "status": "success",
                "total_pages": len(all_documents),
                "total_chunks": len(chunks),
                "pdf_stats": pdf_stats
            }
            
        except Exception as e:
            logger.error(f"Error during PDF ingestion: {str(e)}")
            raise
    
    def _create_qa_chain(self):
        """
        Create the question-answering chain with citation grounding.
        """
        # System prompt for strict grounding and citation
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
        
        # Create document combination chain
        combine_docs_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt
        )
        
        # Create retrieval chain
        self.qa_chain = create_retrieval_chain(
            retriever=self.retriever,
            combine_docs_chain=combine_docs_chain
        )
        
        logger.info("QA chain created successfully")
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get a grounded answer with citations.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing answer, reasoning, and sources
        """
        if self.qa_chain is None:
            raise ValueError("Please upload PDFs first before asking questions")
        
        try:
            # Invoke the chain
            response = self.qa_chain.invoke({"input": question})
            
            # Extract retrieved documents for citation verification
            retrieved_docs = response.get("context", [])
            
            # Parse the response
            answer_text = response.get("answer", "")
            
            # Extract sources from retrieved documents
            sources = []
            seen_sources = set()
            
            for doc in retrieved_docs:
                source_file = doc.metadata.get("source_file", "Unknown")
                page = doc.metadata.get("page", 0) + 1  # Convert to 1-indexed
                source_key = f"{source_file}:Page {page}"
                
                if source_key not in seen_sources:
                    sources.append({
                        "file": source_file,
                        "page": page,
                        "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    })
                    seen_sources.add(source_key)
            
            return {
                "status": "success",
                "question": question,
                "answer": answer_text,
                "sources": sources,
                "num_sources": len(sources)
            }
            
        except Exception as e:
            logger.error(f"Error during question answering: {str(e)}")
            return {
                "status": "error",
                "question": question,
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "num_sources": 0
            }
    
    def clear_database(self):
        """Clear the vector database and reset the engine."""
        try:
            if self.vector_store is not None:
                self.vector_store = None
                self.retriever = None
                self.qa_chain = None
            
            # Remove ChromaDB directory
            import shutil
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
            
            logger.info("Database cleared successfully")
            return {"status": "success", "message": "Database cleared"}
            
        except Exception as e:
            logger.error(f"Error clearing database: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the current vector store."""
        if self.vector_store is None:
            return {
                "status": "empty",
                "total_documents": 0,
                "message": "No documents loaded"
            }
        
        try:
            # Get collection stats
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                "status": "ready",
                "total_documents": count,
                "message": f"Ready with {count} document chunks"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }