import json
import subprocess
import os
import fitz  # PyMuPDF
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Inputs
##ollama run qwen2.5:1.5b
JSON_FILE = 'files/reports_companies.json' 
WORKER_SCRIPT =  '03_generate_RAG.py'
#'00_formating.py'
#'02_generate_vectorstore_ollama.py'
#'01_generate_chunks_from_md.py'
#'03_generate_RAG.py'
def run_pipeline():
    # 1. Load the list of reports and companies
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {JSON_FILE}")
        return

    print(f"Found {len(data)} reports to process.\n")

    # 2. Iterate through each item
    for entry in data:
        report_name = entry['report_name']
        company_name = entry['company_name']
        

        print(f"--- Processing {company_name} ---")
        
        try:

            subprocess.run(
                [sys.executable, WORKER_SCRIPT, report_name, company_name], 
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ Error processing {company_name}: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            
    print("\nPipeline execution finished.")

if __name__ == "__main__":
    run_pipeline()