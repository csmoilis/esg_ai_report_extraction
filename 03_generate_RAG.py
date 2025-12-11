import os
import sys
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Ollama Imports ---
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings

# --- CONFIGURATION ---
LLM_MODEL = "llama3.1" 

from langchain_community.vectorstores import FAISS
# specific embedding model used during creation is required
from langchain_openai import OpenAIEmbeddings 

embeddings = OpenAIEmbeddings()

# Load the saved index
new_vectorstore = FAISS.load_local(
    "faiss_index_local", 
    embeddings, 
    allow_dangerous_deserialization=True
)

# Now you can use it as normal
vectorstore = new_vectorstore.similarity_search("your query here")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 
from langchain_core.output_parsers import JsonOutputParser

# ... [Previous code for loading chunks and vector store remains the same] ...

# 4. Create Chain with JSON Output
print("--- Building JSON RAG Chain ---")

# IMPORTANT: Set format="json" to force the model to output valid JSON

llm = ChatOllama(model=LLM_MODEL, temperature=0, format="json")

# Update prompt to ask for specific JSON keys
template = """You are a helpful AI assistant.
Answer the user's question based ONLY on the provided context.

Output your response as a JSON object with the following keys:
- "answer": The direct answer to the question.
- "source_text": The exact text from the context that contains the answer.

Context:
{context}

Question: {input}
"""
prompt = ChatPromptTemplate.from_template(template)

# Use JsonOutputParser to automatically convert the text response into a Python dictionary
generation_chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
    | JsonOutputParser()
)

#================================
# Generate function
#====================
query = "initiatives to reach the carbon reduction goal"

retrieved_docs = retriever.invoke(query)
print(f"\n--- Retrieved {len(retrieved_docs)} relevant chunks for the query: ---")
print(f"Query: '{query}'\n")

# Format 'retrieved_docs' into a single string
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 3. Invoke the chain
# IMPORTANT: We use "input" here because your prompt template uses {input}
query = "What are the initiatives to reach the carbon reduction goal?"

print("🤔 Generating answer from provided docs...")
response_stream = generation_chain.stream({
    "context": context_text,
    "input": query
})

print("✅ Answer:")
for chunk in response_stream:
    print(chunk, end="", flush=True)
print("\n")