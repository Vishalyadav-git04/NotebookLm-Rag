"""
Configuration settings for NotebookLM-like RAG System
Modify these values to customize system behavior
"""

import os
from pathlib import Path

class Config:
    """Central configuration for RAG system."""
    
    # ==================== API Settings ====================
    
    # Google API Key (required)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    
    # ==================== Model Settings ====================
    
    # Embedding model
    EMBEDDING_MODEL = "models/text-embedding-004"
    
    # LLM model for generation
    # Options: "gemini-flash-latest", "gemini-1.5-pro-latest"
    LLM_MODEL = "gemini-flash-latest"
    
    # LLM temperature (0.0 = deterministic, 1.0 = creative)
    LLM_TEMPERATURE = 0.1
    
    # ==================== Chunking Settings ====================
    
    # Target chunk size in characters
    CHUNK_SIZE = 1000
    
    # Overlap between chunks (helps preserve context)
    CHUNK_OVERLAP = 200
    
    # Separators for text splitting (hierarchical)
    CHUNK_SEPARATORS = ["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    
    # ==================== Retrieval Settings ====================
    
    # Number of chunks to retrieve per query
    RETRIEVAL_K = 6
    
    # Search type: "similarity" or "mmr" (maximal marginal relevance)
    SEARCH_TYPE = "similarity"
    
    # Minimum similarity score threshold (0.0-1.0)
    # Set to None to disable threshold filtering
    SIMILARITY_THRESHOLD = None
    
    # ==================== Vector Store Settings ====================
    
    # Directory for ChromaDB persistence
    PERSIST_DIRECTORY = "./chroma_db"
    
    # Collection name in ChromaDB
    COLLECTION_NAME = "rag_documents"
    
    # ==================== File Upload Settings ====================
    
    # Maximum file size in MB
    MAX_FILE_SIZE_MB = 100
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = [".pdf"]
    
    # Temporary directory for PDF processing
    TEMP_PDF_DIR = "./temp_pdfs"
    
    # ==================== UI Settings ====================
    
    # Application title
    APP_TITLE = "NotebookLM-like RAG System"
    
    # Page icon emoji
    PAGE_ICON = "📚"
    
    # Sidebar width (wide or auto)
    SIDEBAR_STATE = "expanded"
    
    # ==================== Prompt Settings ====================
    
    # System prompt template
    SYSTEM_PROMPT = """You are an expert research assistant analyzing uploaded PDF documents.

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
    
    # ==================== Logging Settings ====================
    
    # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Log file path (None = console only)
    LOG_FILE = None
    
    # ==================== Performance Settings ====================
    
    # Enable batch processing for embeddings
    BATCH_EMBEDDINGS = True
    
    # Batch size for embedding generation
    EMBEDDING_BATCH_SIZE = 100
    
    # Enable caching for repeated queries
    ENABLE_QUERY_CACHE = False
    
    # ==================== Advanced Settings ====================
    
    # Use incremental updates for vector store
    INCREMENTAL_UPDATES = True
    
    # Clear vector store on startup
    CLEAR_DB_ON_STARTUP = False
    
    # Enable detailed error messages
    DETAILED_ERRORS = True
    
    # ==================== Feature Flags ====================
    
    # Enable chat history export
    ENABLE_EXPORT = True
    
    # Enable database clear button
    ENABLE_CLEAR_DB = True
    
    # Show processing details by default
    SHOW_PROCESSING_DETAILS = True
    
    # Show retrieved context by default
    SHOW_RETRIEVED_CONTEXT = True
    
    # ==================== Production Settings ====================
    
    # Enable authentication (requires additional setup)
    ENABLE_AUTH = False
    
    # Rate limiting (requests per minute)
    RATE_LIMIT = None
    
    # Maximum concurrent users
    MAX_CONCURRENT_USERS = None
    
    # ==================== Helper Methods ====================
    
    @classmethod
    def validate(cls):
        """Validate configuration settings."""
        errors = []
        
        # Check API key
        if not cls.GOOGLE_API_KEY:
            errors.append("GOOGLE_API_KEY not set")
        
        # Check chunk settings
        if cls.CHUNK_SIZE <= cls.CHUNK_OVERLAP:
            errors.append("CHUNK_SIZE must be greater than CHUNK_OVERLAP")
        
        # Check retrieval settings
        if cls.RETRIEVAL_K < 1:
            errors.append("RETRIEVAL_K must be at least 1")
        
        # Check directories
        Path(cls.PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)
        Path(cls.TEMP_PDF_DIR).mkdir(parents=True, exist_ok=True)
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        return True
    
    @classmethod
    def get_retrieval_kwargs(cls):
        """Get retrieval search kwargs."""
        kwargs = {"k": cls.RETRIEVAL_K}
        
        if cls.SIMILARITY_THRESHOLD:
            kwargs["score_threshold"] = cls.SIMILARITY_THRESHOLD
        
        return kwargs
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("="*60)
        print("RAG System Configuration")
        print("="*60)
        print(f"LLM Model: {cls.LLM_MODEL}")
        print(f"Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"Temperature: {cls.LLM_TEMPERATURE}")
        print(f"Chunk Size: {cls.CHUNK_SIZE}")
        print(f"Chunk Overlap: {cls.CHUNK_OVERLAP}")
        print(f"Retrieval K: {cls.RETRIEVAL_K}")
        print(f"Search Type: {cls.SEARCH_TYPE}")
        print(f"Persist Directory: {cls.PERSIST_DIRECTORY}")
        print("="*60)


# ==================== Preset Configurations ====================

class ProductionConfig(Config):
    """Production-ready configuration with enhanced security."""
    
    LLM_TEMPERATURE = 0.0  # Maximum determinism
    DETAILED_ERRORS = False  # Hide internal errors
    LOG_LEVEL = "WARNING"  # Reduce log verbosity
    ENABLE_AUTH = True  # Enable authentication
    RATE_LIMIT = 100  # 100 requests per minute
    CLEAR_DB_ON_STARTUP = False


class DevelopmentConfig(Config):
    """Development configuration for testing."""
    
    LLM_TEMPERATURE = 0.1
    DETAILED_ERRORS = True
    LOG_LEVEL = "DEBUG"
    SHOW_PROCESSING_DETAILS = True
    SHOW_RETRIEVED_CONTEXT = True


class HighAccuracyConfig(Config):
    """Configuration optimized for accuracy."""
    
    LLM_MODEL = "gemini-flash-latest"  # More powerful model
    RETRIEVAL_K = 10  # More context
    CHUNK_SIZE = 1500  # Larger chunks
    CHUNK_OVERLAP = 300  # More overlap
    LLM_TEMPERATURE = 0.0  # Deterministic


class FastConfig(Config):
    """Configuration optimized for speed."""
    
    LLM_MODEL = "gemini-flash-latest"
    RETRIEVAL_K = 4  # Fewer chunks
    CHUNK_SIZE = 800  # Smaller chunks
    CHUNK_OVERLAP = 100  # Less overlap
    BATCH_EMBEDDINGS = True


# ==================== Active Configuration ====================

# Choose which configuration to use
# Options: Config, ProductionConfig, DevelopmentConfig, HighAccuracyConfig, FastConfig
ACTIVE_CONFIG = Config

# Validate configuration on import
if __name__ != "__main__":
    try:
        ACTIVE_CONFIG.validate()
    except ValueError as e:
        print(f"Configuration validation failed: {e}")