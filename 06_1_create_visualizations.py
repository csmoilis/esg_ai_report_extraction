import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os 

output_dir = os.path.join('files', 'Results')
os.makedirs(output_dir, exist_ok=True)
print(f"📂 Saving plots to: {output_dir}")
# Load the data back into a DataFrame

#'rag_analysis_results.csv'


df_results = pd.read_csv('rag_analysis_results.csv')
df_results.columns
# 2. Setup Academic Style & Colors
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["font.family"] = "serif"  # Enforce serif font for academic look

academic_blue = "#003366"  # Navy Blue (for True)
academic_red = "#8B0000"   # Dark Red (for False)

# 3. Process Data
s1 = df_results['Q1_Num_Match'].value_counts()
s2 = df_results['Q1_In_Source'].value_counts()
counts = pd.DataFrame({'Numeric Match': s1, 'In Source': s2}).T.fillna(0)


counts = counts.reindex(columns=[True, False], fill_value=0)

# 4. Create the Output Directory
output_dir = 'files/Results'
os.makedirs(output_dir, exist_ok=True)

# 5. Plotting
ax = counts.plot(
    kind='bar', 
    figsize=(8, 6), 
    rot=0, 
    color=[academic_blue, academic_red], # Maps to [True, False]
    edgecolor='black',                   # Adds a clean border to bars
    width=0.7
)

# 6. Customization
plt.title("Comparison of Matches", fontsize=14, weight='bold')
plt.ylabel("Amount")
plt.xlabel("Match Type")
plt.legend(title="Match Result", frameon=False)

# 7. Save High-Quality Image
save_path = os.path.join(output_dir, 'Q1_Accuracy_Metrics.png')
plt.tight_layout()
plt.savefig(save_path, dpi=300)  # 300 DPI is standard for publication

print(f"Plot saved successfully to: {save_path}")


#### ========   question 2
# 2. Setup Academic Style & Colors
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["font.family"] = "serif"  # Enforce serif font for academic look


# 3. Process Data
s1 = df_results['Q2_Num_Match'].value_counts()
s2 = df_results['Q2_In_Source'].value_counts()
counts = pd.DataFrame({'Numeric Match': s1, 'In Source': s2}).T.fillna(0)


counts = counts.reindex(columns=[True, False], fill_value=0)

# 4. Create the Output Directory
output_dir = 'files/Results'
os.makedirs(output_dir, exist_ok=True)

# 5. Plotting
ax = counts.plot(
    kind='bar', 
    figsize=(8, 6), 
    rot=0, 
    color=[academic_blue, academic_red], # Maps to [True, False]
    edgecolor='black',                   # Adds a clean border to bars
    width=0.7
)

# 6. Customization
plt.title("Comparison of Matches", fontsize=14, weight='bold')
plt.ylabel("Amount")
plt.xlabel("Match Type")
plt.legend(title="Match Result", frameon=False)

# 7. Save High-Quality Image
save_path = os.path.join(output_dir, 'Q2_Accuracy_Metrics.png')
plt.tight_layout()
plt.savefig(save_path, dpi=300)  # 300 DPI is standard for publication

print(f"Plot saved successfully to: {save_path}")

# ---------------------------------------------------------
# 4. PLOT 2: SEMANTIC SIMILARITY
# ---------------------------------------------------------
df_sorted = df_results.sort_values('Q1_Source_Sim', ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(df_sorted['Company'], df_sorted['Q1_Source_Sim'], 
               color=academic_blue, edgecolor='black', height=0.65)

ax.set_title('Semantic Similarity: Generated Source vs. Golden Source (Q1)', fontsize=16, weight='bold', pad=20)
ax.set_xlabel('Cosine Similarity Score', fontsize=12)
ax.set_xlim(0, 1.1)
ax.grid(axis='x', linestyle='--', alpha=0.6)
sns.despine(left=True)

# Add numeric values to bars
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
            va='center', fontsize=10, color='black')

save_path_2 = os.path.join(output_dir, 'Q1_Semantic_Similarity.png')
plt.tight_layout()
plt.savefig(save_path_2, dpi=300)
plt.close()
print(f"✅ Saved: {save_path_2}")

## ========= Question2 

df_sorted = df_results.sort_values('Q2_Source_Sim', ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(df_sorted['Company'], df_sorted['Q2_Source_Sim'], 
               color=academic_blue, edgecolor='black', height=0.65)

ax.set_title('Semantic Similarity: Generated Source vs. Golden Source (Q2)', fontsize=16, weight='bold', pad=20)
ax.set_xlabel('Cosine Similarity Score', fontsize=12)
ax.set_xlim(0, 1.1)
ax.grid(axis='x', linestyle='--', alpha=0.6)
sns.despine(left=True)

# Add numeric values to bars
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
            va='center', fontsize=10, color='black')

save_path_2 = os.path.join(output_dir, 'Q2_Semantic_Similarity.png')
plt.tight_layout()
plt.savefig(save_path_2, dpi=300)
plt.close()
print(f"✅ Saved: {save_path_2}")

## ========= Question 3 - Answer Similarity

df_sorted = df_results.sort_values('Q3_Ans_Sim', ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(df_sorted['Company'], df_sorted['Q3_Ans_Sim'], 
               color=academic_blue, edgecolor='black', height=0.65)

ax.set_title('Semantic Similarity: Generated Answer vs. Golden Answer (Q3)', fontsize=16, weight='bold', pad=20)
ax.set_xlabel('Cosine Similarity Score', fontsize=12)
ax.set_xlim(0, 1.1)
ax.grid(axis='x', linestyle='--', alpha=0.6)
sns.despine(left=True)

# Add numeric values to bars
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
            va='center', fontsize=10, color='black')

save_path = os.path.join(output_dir, 'Q3_Semantic_Similarity_answer.png')
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()
print(f"✅ Saved: {save_path}")

## ========= Question 3 - Source Similarity

df_sorted = df_results.sort_values('Q3_Source_Sim', ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(df_sorted['Company'], df_sorted['Q3_Source_Sim'], 
               color=academic_blue, edgecolor='black', height=0.65)

ax.set_title('Semantic Similarity: Generated Source vs. Golden Source (Q3)', fontsize=16, weight='bold', pad=20)
ax.set_xlabel('Cosine Similarity Score', fontsize=12)
ax.set_xlim(0, 1.1)
ax.grid(axis='x', linestyle='--', alpha=0.6)
sns.despine(left=True)

# Add numeric values to bars
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
            va='center', fontsize=10, color='black')

save_path = os.path.join(output_dir, 'Q3_Semantic_Similarity_Source.png')
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()
print(f"✅ Saved: {save_path}")