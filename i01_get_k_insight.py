import os
import sys
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import json

# --- Ollama Imports ---
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import JsonOutputParser

#ollama run qwen2.5:1.5b
# --- PARAMETERS ---
LLM_MODEL = "qwen2.5:1.5b" 

company_name = "Google" 

model_name = "nomic-embed-text" 
embeddings = OllamaEmbeddings(model=model_name)


new_vectorstore = FAISS.load_local(
    f"files/faiss/faiss_{company_name}", 
    embeddings, 
    allow_dangerous_deserialization=True
)


# Question 1
retriever_query = "Scope 1 (Direct Operations) GHG Gross emissions"#carbon footprint Gross Greenhouse gas
query_1 = "What is the total amount of Scope 1 (Direct) GHG emissions reported?"


#delete:

retriever = new_vectorstore.as_retriever(search_kwargs={"k": 9}) 
query = "What is the total amount of Scope 1 (Direct Operations) GHG emissions reported on the latest fiscal year?"
retrieved_docs = retriever.invoke(retriever_query)
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
context_text


# Assuming you already have your text in this variable
# context_text = " ... your long text ... "

target_number = "79,400"

if target_number in context_text:
    print(f"Yes, {target_number} is found in the text.")
else:
    print(f"No, {target_number} is NOT in the text.")
    