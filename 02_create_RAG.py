import argparse
import json
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# --- New Imports for RAG ---
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

def main():
    parser = argparse.ArgumentParser(description="RAG Inference with vLLM and Ollama")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Hugging Face model for generation.")
    parser.add_argument("--input-file", type=str, required=True, help="JSON file containing text chunks.")
    parser.add_argument("--output-file", type=str, required=True, help="File to save the result.")
    args = parser.parse_args()

    # ==========================================
    # 1. LOAD CHUNKS (Your Custom Logic)
    # ==========================================
    CHUNKS_FILE = args.input_file
    loaded_chunks = []

    if os.path.exists(CHUNKS_FILE):
        print(f"--- Found {CHUNKS_FILE}. Loading chunks... ---")
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # We use enumerate(data, 1) to get an index 'i' starting at 1
        for i, item in enumerate(data, 1):
            # CHECK 1: If item is a dictionary (The correct format)
            if isinstance(item, dict):
                original_content = item.get("page_content") or item.get("text") # Added fallback
                # Prepend the chunk number to the content
                numbered_content = f"Chunk number {i}: {original_content}"

                loaded_chunks.append(
                    Document(page_content=numbered_content, metadata=item.get("metadata", {}))
                )

            # CHECK 2: If item is a string (The legacy/wrong format)
            elif isinstance(item, str):
                # Prepend the chunk number to the content
                numbered_content = f"Chunk number {i}: {item}"

                loaded_chunks.append(
                    Document(page_content=numbered_content, metadata={})
                )
        
        print(f"✅ Loaded {len(loaded_chunks)} chunks.")
    else:
        raise FileNotFoundError(f"Input file {CHUNKS_FILE} not found.")

    # ==========================================
    # 2. EMBEDDING & RETRIEVAL (Ollama + FAISS)
    # ==========================================
    print("--- Initializing Embeddings (Ollama: nomic-embed-text) ---")
    # Ensure 'ollama serve' is running in the background or accessible on the node
    embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
    
    print("Building FAISS Vector Store...")
    vectorstore = FAISS.from_documents(loaded_chunks, embeddings_model)
    
    print("Running Retrieval...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Hardcoded query as requested
    retriever_query = "carbon emission reduction target"
    retrieved_docs = retriever.invoke(retriever_query)
    
    # Combine retrieved docs into a single context string
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    print(f"Context retrieved (Length: {len(context_text)} chars)")

    # ==========================================
    # 3. GENERATION (vLLM)
    # ==========================================
    print(f"--- Loading vLLM Model: {args.model} ---")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Construct the RAG Prompt
    question = "What is the carbon emission reduction target?"
    
    rag_prompt_content = (
        f"You are a helpful assistant. Answer the question based ONLY on the context provided below.\n\n"
        f"### Context:\n{context_text}\n\n"
        f"### Question:\n{question}"
    )

    messages = [{"role": "user", "content": rag_prompt_content}]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        enable_thinking=True 
    )

    # Initialize Engine
    llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=8192)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)

    print("Generating Answer...")
    outputs = llm.generate([formatted_prompt], sampling_params)

    # Decode
    generated_ids = outputs[0].outputs[0].token_ids
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    print("\n--- Final Answer ---")
    print(generated_text)
    print("--------------------")

    # ==========================================
    # 4. SAVE RESULTS
    # ==========================================
    result = {
        "query": question,
        "retrieved_context": context_text,
        "generated_answer": generated_text
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(result, f, indent=4)
        
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()