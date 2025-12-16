import json
import os
import sys

# Make sure you have installed these: pip install langchain-ollama faiss-cpu
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Configuration
# Note: Ensure you have pulled this model in Ollama (e.g., `ollama pull nomic-embed-text`)
# "nomic-embed-text" is generally recommended over LLMs like Qwen for embeddings.
model_name = "nomic-embed-text" 
#company_name = "Amazon" 
company_name = sys.argv[2]
CHUNKS_FILE = f"files/Chunks/chunks_{company_name}.json" 
INDEX_NAME = f"files/faiss/faiss_{company_name}"

def create_vector_store():
    # 1. Check if the chunks file exists
    if not os.path.exists(CHUNKS_FILE):
        print(f"❌ Error: File {CHUNKS_FILE} not found.")
        sys.exit(1)

    print(f"--- Found {CHUNKS_FILE}. Loading chunks... ---")
    
    # 2. Load the data
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    loaded_chunks = []
    
    # 3. Process chunks
    for i, item in enumerate(data, 1):
        numbered_content = ""
        metadata = {}

        # Handle Dict format (Standard)
        if isinstance(item, dict):
            content = item.get("page_content", "")
            metadata = item.get("metadata", {})
            numbered_content = f"Chunk number {i}: {content}"
        
        # Handle String format (Legacy)
        elif isinstance(item, str):
            numbered_content = f"Chunk number {i}: {item}"
        
        if numbered_content:
            loaded_chunks.append(
                Document(page_content=numbered_content, metadata=metadata)
            )

    print(f"✅ Loaded {len(loaded_chunks)} chunks.")

    # 4. Initialize Ollama Embeddings
    print(f"--- Generating embeddings using model: {model_name} ---")
    print("(This may take a few minutes depending on data size...)")
    
    embeddings = OllamaEmbeddings(model=model_name)

    # 5. Create and Save FAISS Index
    try:
        vectorstore = FAISS.from_documents(loaded_chunks, embeddings)
        vectorstore.save_local(INDEX_NAME)
        print(f"🎉 Success! FAISS index saved to folder: '{INDEX_NAME}'")
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")

if __name__ == "__main__":
    create_vector_store()