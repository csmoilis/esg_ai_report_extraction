import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os 

output_dir = os.path.join('files', 'Results')
os.makedirs(output_dir, exist_ok=True)
print(f"📂 Saving plots to: {output_dir}")
# Load the data back into a DataFrame

#'rag_analysis_results.csv'


df_results = pd.read_csv('Ai_lab_RAG_results.csv')

companies_to_remove = [
        "Meta", "Salesforce", "Disney", "Qualcomm", 
        "Intuit", "Tesla", "Netflix", "Cargill"
    ]
    
    # Keep only rows where the company is NOT in the removal list
    # The '~' symbol means "NOT"
df_results = df_results[~df_results['company'].isin(companies_to_remove)] 

df_results.columns

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# ---------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------
output_dir = os.path.join('files', 'Results')
os.makedirs(output_dir, exist_ok=True)
print(f"📂 Saving plots to: {output_dir}")

# Define Academic Colors
academic_blue = "#003366"  # Navy Blue
academic_red = "#8B0000"   # Dark Red

# Set Plotting Style
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["font.family"] = "serif"



# ---------------------------------------------------------
# 3. PLOT TYPE A: RECALL METRICS (Q1, Q2, Q3 Combined)
# ---------------------------------------------------------
# We combine all three recalls into one grouped bar chart for comparison
fig, ax = plt.subplots(figsize=(10, 6))

# Prepare data counts
s1 = df_results['recall_1'].value_counts()
s2 = df_results['recall_2'].value_counts()
s3 = df_results['recall_3'].value_counts()

# Create DataFrame for plotting
counts = pd.DataFrame({
    'Q1 Recall': s1, 
    'Q2 Recall': s2, 
    'Q3 Recall': s3
}).T.fillna(0)


# Plot
counts.plot(kind='bar', ax=ax, color=[academic_blue, academic_red], 
            edgecolor='black', width=0.8, rot=0)

ax.set_title('Recall Metrics: Comparison Across Questions', fontsize=16, weight='bold', pad=20)
ax.set_ylabel('Number of Companies', fontsize=12)
ax.legend(title='Outcome', labels=['True (Found)', 'False (Not Found)'], 
          frameon=True, fancybox=False, edgecolor='black', loc='best')
ax.grid(axis='y', linestyle='--', alpha=0.6)
sns.despine(left=True)

save_path_recall = os.path.join(output_dir, 'Recall_Metrics_Combined.png')
plt.tight_layout()
plt.savefig(save_path_recall, dpi=300)
plt.close()
print(f"✅ Saved: {save_path_recall}")


# ---------------------------------------------------------
# 4. PLOT TYPE B: LLM JUDGE (Frequency Count)
# ---------------------------------------------------------
judge_cols = [('llm_judge_1', 'Q1'), ('llm_judge_2', 'Q2'), ('llm_judge_3', 'Q3')]

for col, label in judge_cols:
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get counts
    counts_judge = df_results[col].value_counts()
    
    # Plot
    counts_judge.plot(kind='bar', ax=ax, color=academic_blue, edgecolor='black', rot=0, width=0.6)
    
    ax.set_title(f'LLM Judge Verdict: {label}', fontsize=16, weight='bold', pad=20)
    ax.set_xlabel('Verdict', fontsize=12)
    ax.set_ylabel('Number of Companies', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    sns.despine(left=True)
    
    save_path_judge = os.path.join(output_dir, f'{label}_LLM_Judge.png')
    plt.tight_layout()
    plt.savefig(save_path_judge, dpi=300)
    plt.close()
    print(f"✅ Saved: {save_path_judge}")

# ---------------------------------------------------------
# 5. PLOT TYPE C: SIMILARITY (Horizontal Bar)
# ---------------------------------------------------------
sim_cols = [('similarity_1', 'Q1'), ('similarity_2', 'Q2'), ('similarity_3', 'Q3')]

for col, label in sim_cols:
    # Sort data for this specific column
    df_sorted = df_results.sort_values(col, ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(df_sorted['company'], df_sorted[col], 
                   color=academic_blue, edgecolor='black', height=0.65)

    ax.set_title(f'Semantic Similarity: {label}', fontsize=16, weight='bold', pad=20)
    ax.set_xlabel('Cosine Similarity Score', fontsize=12)
    ax.set_xlim(0, 1.1)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    sns.despine(left=True)

    # Add numeric values to bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                va='center', fontsize=10, color='black')

    save_path_sim = os.path.join(output_dir, f'{label}_Semantic_Similarity.png')
    plt.tight_layout()
    plt.savefig(save_path_sim, dpi=300)
    plt.close()
    print(f"✅ Saved: {save_path_sim}")

print("\nAll plots generated successfully.")