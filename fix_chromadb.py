"""
Diagnostic script to fix ChromaDB onnxruntime issue
Run this before starting the app
"""

import os
import sys

print("=" * 60)
print("ChromaDB Diagnostic & Fix Script")
print("=" * 60)
print()

# Step 1: Check ChromaDB installation
print("1. Checking ChromaDB installation...")
try:
    import chromadb
    print(f"   ✓ ChromaDB version: {chromadb.__version__}")
except ImportError as e:
    print(f"   ✗ ChromaDB not installed: {e}")
    sys.exit(1)

# Step 2: Check onnxruntime
print("\n2. Checking onnxruntime...")
try:
    import onnxruntime
    print(f"   ✓ onnxruntime version: {onnxruntime.__version__}")
except ImportError:
    print("   ✗ onnxruntime not installed")
    print("   Installing onnxruntime...")
    os.system("pip install onnxruntime")

# Step 3: Set environment variable to disable default embedding
print("\n3. Setting ChromaDB environment variables...")
os.environ["CHROMA_ENABLE_DEFAULT_EMBEDDING"] = "false"
print("   ✓ Disabled default embedding function")

# Step 4: Test ChromaDB initialization
print("\n4. Testing ChromaDB initialization...")
try:
    from chromadb.config import Settings
    
    # Try to create a test client
    client = chromadb.Client(Settings(
        anonymized_telemetry=False,
        allow_reset=True
    ))
    
    print("   ✓ ChromaDB client created successfully")
    
    # Clean up test
    client.reset()
    print("   ✓ ChromaDB reset successful")
    
except Exception as e:
    print(f"   ✗ ChromaDB initialization failed: {e}")
    print("\n   SOLUTION: Try this workaround:")
    print("   1. Delete chroma_db folder if it exists")
    print("   2. Reinstall chromadb:")
    print("      pip uninstall chromadb -y")
    print("      pip install chromadb==0.4.24")
    sys.exit(1)

# Step 5: Test LangChain integration
print("\n5. Testing LangChain-ChromaDB integration...")
try:
    from langchain_community.vectorstores import Chroma
    from langchain.embeddings.base import Embeddings
    
    # Create a dummy embedding function
    class DummyEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [[0.0] * 768 for _ in texts]
        
        def embed_query(self, text):
            return [0.0] * 768
    
    # Test creation
    test_docs = [{"page_content": "test", "metadata": {}}]
    from langchain_core.documents import Document
    docs = [Document(page_content="test", metadata={})]
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=DummyEmbeddings(),
        collection_name="test_collection"
    )
    
    print("   ✓ LangChain-ChromaDB integration working")
    
    # Clean up
    vectorstore.delete_collection()
    
except Exception as e:
    print(f"   ✗ LangChain integration failed: {e}")
    print("\n   This might be a version compatibility issue.")
    print("   Try running:")
    print("   pip install --upgrade langchain-community")

print("\n" + "=" * 60)
print("Diagnostic Complete!")
print("=" * 60)
print("\nIf all tests passed, you can now run:")
print("  streamlit run app.py")
print("\nIf tests failed, follow the solutions printed above.")
print("=" * 60)