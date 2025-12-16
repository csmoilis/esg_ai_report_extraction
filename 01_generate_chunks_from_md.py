import sys
import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Argument Safety Check ---
if len(sys.argv) < 3:
    print("Error: Please provide both a filename and a company name.")
    print("Usage: python script.py <filename.md> <company_name>")
    sys.exit(1)

name_path = sys.argv[1]
company_name = sys.argv[2]

# name_path = "FY2024-NVIDIA-Corporate-Sustainability-Report"
# company_name = "NVIDIA"

file_path = f"files/MD_reports/{company_name}.md"
print(f"Processing file: {file_path} ")

# --- Step 1: Open and Read the MD File ---
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        md_text = f.read()
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    sys.exit(1)

# --- Step 2: Configure Splitter ---
# We use standard separators but prioritize newlines to keep paragraphs together
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

# --- Step 3: Split the Text ---
splits = text_splitter.split_text(md_text)

print(f"Split the document into {len(splits)} string chunks.")

# --- Step 4: Save to JSON ---
# Ensure the 'files' directory exists
os.makedirs('files', exist_ok=True)

output_filename = f'files/Chunks/chunks_{company_name}.json'

with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(splits, f, ensure_ascii=False, indent=4)

print(f'Saved in {output_filename}')