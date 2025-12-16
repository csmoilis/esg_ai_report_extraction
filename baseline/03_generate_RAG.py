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
import openai
#ollama run qwen2.5:1.5b
# --- PARAMETERS ---
LLM_MODEL = "qwen2.5:1.5b" 

HISTORY_FILE = "RAG_results_TINY.json"

company_name = sys.argv[2]

client = openai.OpenAI() if os.getenv("OPENAI_API_KEY") else None
model_name = "nomic-embed-text" 
embeddings = OllamaEmbeddings(model=model_name)


new_vectorstore = FAISS.load_local(
    f"files/faiss/faiss_{company_name}", 
    embeddings, 
    allow_dangerous_deserialization=True
)


retriever = new_vectorstore.as_retriever(search_kwargs={"k": 9}) 


print("--- Building JSON RAG Chain ---")

llm = ChatOllama(model=LLM_MODEL, temperature=0, format="json")

template = """You are a helpful AI assistant.
Answer the user's question based ONLY on the provided context.

Output your response as a JSON object with the following keys:
- "answer": The direct answer to the question. If the answer is not found answer "Not found".
- "source_text": The exact text from the context that contains the answer.

Context:
{context}

Question: {input}
"""
prompt = ChatPromptTemplate.from_template(template)

# Use JsonOutputParser to automatically convert the text response into a Python dictionary
generation_chain = (
    prompt
    | llm
    | JsonOutputParser()
)
#================================
# Generate function
#================================


def generate_answer(retriever_query,query):
    
    retrieved_docs = retriever.invoke(retriever_query)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    response = generation_chain.invoke({
        "context": context_text,
        "input": query
    })

    return response



# ================================
# =========== Questions 
# ================================


print(f"--- Querying about {company_name}... ---")

# Question 1
retriever_query = "Scope 1 (Direct Operations) GHG Gross emissions"#carbon footprint Gross Greenhouse gas
query_1 = "What is the total amount of Scope 1 (Direct) GHG emissions reported?"
response_data_1 = generate_answer(retriever_query, query_1)

# Question 2
retriever_query = "year achieve Net Zero emissions"
query_2 = "By which year does the company aim to achieve Net Zero emissions?"
response_data_2 = generate_answer(retriever_query, query_2)

# Question 3
retriever_query = "current progress carbon emission reduction"
query_3 = "List the initiatives to reach the carbon reduction goal?"
response_data_3 = generate_answer(retriever_query, query_3)

# 3. Prepare Data to Save
entry = {
    "Company": company_name,
    "question1": query_1,
    "answer1": response_data_1.get("answer"),
    "source_text_1": response_data_1.get("source_text"),
    "question2": query_2,
    "answer2": response_data_2.get("answer"),
    "source_text_2": response_data_2.get("source_text"),
    "question3": query_3,
    "answer3": response_data_3.get("answer"),
    "source_text_3": response_data_3.get("source_text")
}

# 4. Save to JSON File
if os.path.exists(HISTORY_FILE):
    # If file exists, load it first
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        try:
            history = json.load(f)
        except json.JSONDecodeError:
            history = [] # Handle empty or corrupt files
else:
    # If file doesn't exist, start a new list
    history = []

history.append(entry)

with open(HISTORY_FILE, "w", encoding="utf-8") as f:
    json.dump(history, f, indent=4, ensure_ascii=False)

# 5. Print Result for User

print(f"📁 Data saved to: {HISTORY_FILE}")

