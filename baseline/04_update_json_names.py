import json
import os

# --- CONFIGURATION ---
INPUT_FILE = "golden_data.jsonl"
OUTPUT_FILE = "golden_data_fixed.jsonl"

print("--- 🛠️ Starting Golden Dataset Name Fix ---")

# 1. Define the Mapping (Ticker -> RAG Company Name)
# This maps the identifiers in your golden data to the names in your RAG results
name_mapping = {
    "AMZN": "Amazon",
    "MSFT": "Microsoft",
    "MICRON": "Micron",
    "MORGAN STANLEY": "Morgan Stanley",
    "NFLX": "Netflix",
    "PFIZER": "Pfizer",
    "QCOM": "Qualcomm",
    "SALESFORCE": "Salesforce",
    "WALT DISNEY": "Disney",
    "SHELL": "Shell",
    "TSLA": "Tesla",
    "ANALOG DEVICES": "Analog Devices",
    "AAPL": "Apple",
    "ASML": "ASML",
    "CARGILL": "Cargill",
    "CHEVRON": "Chevron",
    "COCA COLA": "Coca-Cola",
    "CVS HEALTH": "CVS",
    "NVDA": "NVIDIA",
    "WMT": "Walmart",
    "GOOG": "Google",
    "IKEA": "IKEA",
    "INTUIT": "Intuit",
    "LEGO": "LEGO Group",
    "META": "Meta"
}

# 2. Process the File
updated_records = []
count_fixed = 0

if os.path.exists(INPUT_FILE):
    print(f"Reading from: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                # Parse JSON line
                record = json.loads(line)
                
                # Check and fix name
                old_name = record.get("company_name")
                if old_name in name_mapping:
                    new_name = name_mapping[old_name]
                    record["company_name"] = new_name
                    count_fixed += 1
                
                updated_records.append(record)
                
            except json.JSONDecodeError:
                print(f"⚠️ Warning: Could not parse line {line_num}")

    # 3. Save the Fixed File
    if updated_records:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for record in updated_records:
                f.write(json.dumps(record) + "\n")
        
        print(f"✅ Success! Fixed {count_fixed} company names.")
        print(f"📁 Saved new file to: {OUTPUT_FILE}")
        print("👉 You can now use this file in your evaluation script.")
    else:
        print("⚠️ No records found to save.")

else:
    print(f"❌ Error: Input file '{INPUT_FILE}' not found.")