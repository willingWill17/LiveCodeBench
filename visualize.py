import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_insights(df):
    easy, medium, hard = 0, 0, 0
    easy_pass, medium_pass, hard_pass = 0, 0, 0
    for index, row in df.iterrows():
        if row['difficulty'].lower() == 'easy':
            easy += 1
            if row['pass@1'] == 1.0:
                easy_pass += 1
        elif row['difficulty'].lower() == 'medium':
            medium += 1
            if row['pass@1'] == 1.0:
                medium_pass += 1
        elif row['difficulty'].lower() == 'hard':
            hard += 1
            if row['pass@1'] == 1.0:
                hard_pass += 1

    # Calculate percentages
    easy_pct = (easy_pass / easy) * 100 if easy > 0 else 0
    medium_pct = (medium_pass / medium) * 100 if medium > 0 else 0
    hard_pct = (hard_pass / hard) * 100 if hard > 0 else 0
    pass1_pct = ((easy_pass + medium_pass + hard_pass) /
                 (easy + medium + hard)) * 100 if (easy + medium + hard) > 0 else 0

    return pass1_pct, easy_pct, medium_pct, hard_pct

# Load JSON
df_old = pd.read_json("output/GPT-5-Nano/Scenario.codegeneration_1_0.2_eval_all.json")
df_new = pd.read_json("output/GPT-5-Nano-with-Cipher/Scenario.codegeneration_1_0.2_codegeneration_output_eval_all.json")

# Get model stats
old_stats = get_insights(df_old)
new_stats = get_insights(df_new)

# Labels and data
labels = ["Pass@1", "Easy", "Medium", "Hard"]
models = ["GPT-5-nano", "GPT-5-nano with Cipher"]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8,5))
rects1 = ax.bar(x - width/2, old_stats, width, label=models[0])
rects2 = ax.bar(x + width/2, new_stats, width, label=models[1])

ax.set_ylabel('Percentage (%)')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add % labels above bars
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(rects1)
add_labels(rects2)

plt.ylim(0, 110)
plt.tight_layout()
plt.savefig('visualize_result.png')

# Count difficulty distribution (from new model dataset or combined)
difficulty_counts = df_new['difficulty'].str.lower().value_counts()

# Pie chart settings
colors = {
    'easy': '#FF6B6B',    
    'medium': '#FFC94D',  
    'hard': '#4DABFF'     
}

fig, ax = plt.subplots(figsize=(5,5))
wedges, texts, autotexts = ax.pie(
    difficulty_counts,
    labels=difficulty_counts.index.str.capitalize(),
    autopct=lambda pct: f"{int(round(pct/100.*sum(difficulty_counts)))}",
    startangle=140,
    colors=[colors[d] for d in difficulty_counts.index.str.lower()],
    textprops={'color':'white', 'weight':'bold'}
)

# Improve style
for t in texts:
    t.set_color('black')
for at in autotexts:
    at.set_color('white')
    at.set_weight('bold')

ax.set_title('Question Difficulty Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('difficulty_distribution_pie.png', dpi=300)