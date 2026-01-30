import sys
import subprocess
import os

def install_package(package):
    print(f"🔧 Installing {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully!")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}. Please check your internet.")

def patch_rag_engine():
    file_path = "rag_engine.py"
    print(f"🔧 Patching {file_path} to handle empty PDFs...")
    
    if not os.path.exists(file_path):
        print(f"❌ Could not find {file_path}!")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # The code snippet to search for
    search_code = 'chunks = self.text_splitter.split_documents(all_documents)\n            logger.info(f"Created {len(chunks)} chunks")'
    
    # The fix to insert (checks for 0 chunks)
    replacement_code = """chunks = self.text_splitter.split_documents(all_documents)
            logger.info(f"Created {len(chunks)} chunks")
            
            if not chunks:
                return {
                    "status": "error",
                    "answer": "The PDF appears to be empty or scanned (no text found). I cannot process it.",
                    "total_pages": len(all_documents),
                    "total_chunks": 0,
                    "pdf_stats": pdf_stats
                }"""

    if "if not chunks:" in content:
        print("✅ rag_engine.py is already patched!")
    elif search_code in content:
        new_content = content.replace(search_code, replacement_code)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print("✅ rag_engine.py patched successfully!")
    else:
        print("⚠️ Could not find the exact code block to patch. You may need to edit it manually.")

if __name__ == "__main__":
    print("="*40)
    print("      NOTEBOOKLM RAG FIXER      ")
    print("="*40)
    
    # 1. Force install the missing package
    install_package("onnxruntime")
    
    # 2. Patch the code to prevent "0 chunk" crashes
    patch_rag_engine()
    
    print("\n" + "="*40)
    print("✅ REPAIR COMPLETE")
    print("Please restart your app with:")
    print("streamlit run app.py")
    print("="*40)