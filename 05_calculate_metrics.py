import json
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import re
# --- CONFIGURATION ---
GOLDEN_DATA_PATH = "golden_dataset_V3.json"
RAG_RESULTS_PATH = "RAG_results_TINY.json"
MODEL_NAME = 'all-MiniLM-L6-v2'  # Small, fast model for semantic similarity

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
golden_records = []
rag_results = []

# Load Golden Data (JSONL)
import json
import os

if os.path.exists(GOLDEN_DATA_PATH):
    with open(GOLDEN_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)       
        if isinstance(data, list):           
            golden_records.extend(data) 
        else:
            golden_records.append(data)
else:
    print(f"Error: {GOLDEN_DATA_PATH} not found.")

# Load RAG Results (JSON)
if os.path.exists(RAG_RESULTS_PATH):
    with open(RAG_RESULTS_PATH, 'r', encoding='utf-8') as f:
        rag_results = json.load(f)
else:
    print(f"Error: {RAG_RESULTS_PATH} not found. Run your generation script first.")


rag_map = {item.get('Company', 'Unknown'): item for item in rag_results}

# ---------------------------------------------------------
# 2. INITIALIZE MODEL
# ---------------------------------------------------------
print(f"Loading embedding model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

analysis_results = []

# ---------------------------------------------------------
# 3. EVALUATION LOOP
# ---------------------------------------------------------

#recall answer Q1 and Q2: see if the number match
#recall source Q1 and Q2: see if the number match golden dataset answer
#semantic similarity of answer of Q3
#semantic similarity of source Q2
#semantic similarity of source Q3
if golden_records and rag_results:
    print(f"Starting analysis on {len(golden_records)} records...")

    for gold in golden_records:
        company = gold.get('company_name') or gold.get('ticker_name')
        generated = rag_map.get(company)
        
        if not generated:
            print(f"⚠️ Warning: No RAG result found for {company}")
            continue

        # =========================================================
        # QUESTION 1: SCOPE 1 EMISSIONS
        # =========================================================
        
        q1_rag_ans = str(generated.get('answer1', ''))
        q1_gold_ans = str(gold.get('answer1', ''))
        
        # 1. Numeric Match (RAG Answer vs Golden Answer)
        raw_matches_rag = re.findall(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?', q1_rag_ans)
        nums_rag_q1 = {float(m.replace(',', '')) for m in raw_matches_rag if m.replace(',', '').replace('.', '').lstrip('-').isdigit()}
        
        raw_matches_gold = re.findall(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?', q1_gold_ans)
        nums_gold_q1 = {float(m.replace(',', '')) for m in raw_matches_gold if m.replace(',', '').replace('.', '').lstrip('-').isdigit()}

        q1_num_match = (nums_rag_q1 == nums_gold_q1)

        # 2. Numeric Containment (RAG Answer vs Golden SOURCE)
        q1_gold_src = str(gold.get('source_answer1', ''))
        raw_matches_src = re.findall(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?', q1_gold_src)
        nums_src_q1 = {float(m.replace(',', '')) for m in raw_matches_src if m.replace(',', '').replace('.', '').lstrip('-').isdigit()}

        q1_num_contained = nums_rag_q1.issubset(nums_src_q1)

        # 3. Source Similarity (RAG Source vs Golden Source)
        q1_rag_src = str(generated.get('source_text_1', ''))
        
        if q1_gold_src and q1_rag_src:
            emb = model.encode([q1_gold_src, q1_rag_src])
            q1_src_sim = float(cosine_similarity([emb[0]], [emb[1]])[0][0])
        else:
            q1_src_sim = 0.0

        # =========================================================
        # QUESTION 2: NET ZERO YEAR
        # =========================================================

        q2_rag_ans = str(generated.get('answer2', ''))
        q2_gold_ans = str(gold.get('answer2', ''))

        # 1. Numeric Match
        raw_matches_rag = re.findall(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?', q2_rag_ans)
        nums_rag_q2 = {float(m.replace(',', '')) for m in raw_matches_rag if m.replace(',', '').replace('.', '').lstrip('-').isdigit()}

        raw_matches_gold = re.findall(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?', q2_gold_ans)
        nums_gold_q2 = {float(m.replace(',', '')) for m in raw_matches_gold if m.replace(',', '').replace('.', '').lstrip('-').isdigit()}

        q2_num_match = (nums_rag_q2 == nums_gold_q2)

        # 2. Numeric Containment
        q2_gold_src = str(gold.get('source_answer2', ''))
        raw_matches_src = re.findall(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?', q2_gold_src)
        nums_src_q2 = {float(m.replace(',', '')) for m in raw_matches_src if m.replace(',', '').replace('.', '').lstrip('-').isdigit()}
        
        q2_num_contained = nums_rag_q2.issubset(nums_src_q2)

        # 3. Source Similarity (RAG Source vs Golden Source)
        q2_rag_src = str(generated.get('source_text_2', ''))

        if q2_gold_src and q2_rag_src:
            emb = model.encode([q2_gold_src, q2_rag_src])
            q2_src_sim = float(cosine_similarity([emb[0]], [emb[1]])[0][0])
        else:
            q2_src_sim = 0.0

        # =========================================================
        # QUESTION 3: INITIATIVES (Semantics Only)
        # =========================================================
        
        q3_gold_ans = str(gold.get('answer3', ''))
        q3_rag_ans = str(generated.get('answer3', ''))
        q3_gold_src = str(gold.get('source_answer3', ''))
        q3_rag_src = str(generated.get('source_text_3', ''))

        # 1. Answer Similarity
        if q3_gold_ans and q3_rag_ans:
            emb = model.encode([q3_gold_ans, q3_rag_ans])
            q3_ans_sim = float(cosine_similarity([emb[0]], [emb[1]])[0][0])
        else:
            q3_ans_sim = 0.0

        # 2. Source Similarity
        if q3_gold_src and q3_rag_src:
            emb = model.encode([q3_gold_src, q3_rag_src])
            q3_src_sim = float(cosine_similarity([emb[0]], [emb[1]])[0][0])
        else:
            q3_src_sim = 0.0

        # ---------------------------------------------------------
        # AGGREGATE
        # ---------------------------------------------------------
        
        analysis_results.append({
            "Company": company,
            "Q1_Num_Match": q1_num_match,
            "Q1_In_Source": q1_num_contained,
            "Q1_Source_Sim": round(q1_src_sim, 3),
            
            "Q2_Num_Match": q2_num_match,
            "Q2_In_Source": q2_num_contained,
            "Q2_Source_Sim": round(q2_src_sim, 3),

            "Q3_Ans_Sim": round(q3_ans_sim, 3),
            "Q3_Source_Sim": round(q3_src_sim, 3)
        })

    # Output Results
    df_results = pd.DataFrame(analysis_results)
    print("\nAnalysis Complete:")
    print(df_results.to_string())

else:
    print("Error: Either golden_records or rag_results is empty.")
    
df_results.to_csv('rag_analysis_results.csv', index=False)

print("✅ Data successfully saved to 'rag_analysis_results.csv'")