import pymupdf4llm
import markdown
from bs4 import BeautifulSoup
import os
import sys 

report_name = sys.argv[1]
company_name = sys.argv[2]


# --- Configuration ---
input_path = f"files/reports/{report_name}.pdf"  
output_path = f"files/MD_reports/{company_name}.md"


print(f"Processing: {input_path}...")

# --- Step 1: Convert PDF to Markdown ---
# Uses PyMuPDF4LLM to extract text and layout
md_content = pymupdf4llm.to_markdown(input_path)

# Save the Markdown file
with open(output_path, "w", encoding="utf-8") as f:
    f.write(md_content)
print(f"Saved: {output_path}")

