import json
from langchain_core.documents import Document
import ollama
import sys
import os
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

model_name = "qwen3-embedding:8b"
text_input = "The quick brown fox jumps over the lazy dog."


company_name = "Amazon"
CHUNKS_FILE = f"files/Chunks/chunks_{company_name}.json" 


if os.path.exists(CHUNKS_FILE):
        print(f"--- Found {CHUNKS_FILE}. Loading chunks from cache... ---")
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        loaded_chunks = []
        # We use enumerate(data, 1) to get an index 'i' starting at 1
        for i, item in enumerate(data, 1):

            # CHECK 1: If item is a dictionary (The correct format)
            if isinstance(item, dict):
                original_content = item.get("page_content")
                # Prepend the chunk number to the content
                numbered_content = f"Chunk number {i}: {original_content}"

                loaded_chunks.append(
                    Document(page_content=numbered_content, metadata=item.get("metadata"))
                )

            # CHECK 2: If item is a string (The legacy/wrong format)
            elif isinstance(item, str):
                # Prepend the chunk number to the content
                numbered_content = f"Chunk number {i}: {item}"

                loaded_chunks.append(
                    Document(page_content=numbered_content, metadata={})
                )

        print(f"✅ Loaded {len(loaded_chunks)} chunks from cache.")
        
        
embeddings = OllamaEmbeddings(model=model_name)
#time: 3 minutes aprox
vectorstore = FAISS.from_documents(loaded_chunks, embeddings)        

# Save the vector store to a folder named "faiss_index_local"
vectorstore.save_local(f"faiss_{company_name}")