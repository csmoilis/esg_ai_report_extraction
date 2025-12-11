import fitz  # PyMuPDF
import sys
import json
# sys.argv[0] is always the script name itself
# sys.argv[1] is the first parameter passed
if len(sys.argv) > 1:
    filename = sys.argv[1]
    print(f"Processing file: {filename}")
else:
    print("Please provide a filename.")

file_name = "2024-amazon-sustainability-report.pdf"
company_name = "Amazon"


def extract_text_return(pdf_path):
    doc = fitz.open(pdf_path)
    
    # Initialize a list to hold text chunks
    text_content = []
    
    # Loop through the PDF
    for page in doc:
        text = page.get_text()
        
        # Add the text and your separator to the list
        text_content.append(text + "\n")
        text_content.append("--- End of Page ---\n")
    
    # Join the list into a single string and return it
    return "".join(text_content)

# Call the function and save the result to a variable
pdf_text = extract_text_return(f"files/{file_name}")

# You can now print it or manipulate it in memory
print(pdf_text)

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

splits = text_splitter.split_text(pdf_text)

print(f"Split the document into {len(splits)} string chunks.")

with open(f'files/chunks_{company_name}.json', 'w', encoding='utf-8') as f:
    json.dump(splits, f, ensure_ascii=False, indent=4)

print(f'saved in files/chunks_{company_name}.json')