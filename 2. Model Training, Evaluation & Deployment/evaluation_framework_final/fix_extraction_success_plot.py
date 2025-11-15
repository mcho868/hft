#!/usr/bin/env python3
"""
Fix the No RAG vs With RAG plot to use correct extraction success calculation.
Extraction success = 1 - (total_failures / cases_evaluated)
where total_failures includes UNKNOWN triage results.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the JSON source data
json_path = Path('/Users/choemanseung/789/hft/comprehensive_evaluation_results_20250930_045843.json')
with open(json_path, 'r') as f:
    data = json.load(f)

print(f"✅ Loaded {len(data)} configurations from JSON")

# Extract metrics with correct extraction success calculation
results = []
for entry in data:
    config = entry['config']

    # Determine if RAG is used
    has_rag = config['rag_config'] is not None

    # CORRECT extraction success calculation
    total_failures = entry.get('total_failures', 0)
    cases_evaluated = entry['cases_evaluated']
    extraction_success = 1 - (total_failures / cases_evaluated)

    results.append({
        'config_name': config['test_name'],
        'has_rag': has_rag,
        'triage_accuracy': entry['triage_accuracy'],
        'f1_score': entry['f1_score'],
        'f2_score': entry['f2_score'],
        'cases_evaluated': cases_evaluated,
        'success_count': entry['success_count'],
        'total_failures': total_failures,
        'unknown_count': entry.get('unknown_triage_count', 0),
        'extraction_success_rate': extraction_success,
        'avg_inference_time': entry['avg_inference_time_per_case']
    })

df = pd.DataFrame(results)

# Save corrected CSV
output_dir = Path('/Users/choemanseung/789/hft/evaluation_framework_final/plots_unknown_unknown')
corrected_csv = output_dir / 'comprehensive_results_analysis_corrected.csv'
df.to_csv(corrected_csv, index=False)
print(f"\n💾 Saved corrected CSV: {corrected_csv.name}")

# ==================== REGENERATE NO RAG VS WITH RAG PLOT ====================
print("\n📊 Regenerating No RAG vs With RAG comparison plot...")

rag_comparison = df.groupby('has_rag').agg({
    'triage_accuracy': 'mean',
    'extraction_success_rate': 'mean',
    'f2_score': 'mean',
    'config_name': 'count'
}).reset_index()

print("\n=== Corrected RAG Comparison ===")
print(rag_comparison)

# Create plot
fig, ax = plt.subplots(figsize=(12, 7))

x = np.array([0, 1])
width = 0.25

bars1 = ax.bar(x - width, rag_comparison['triage_accuracy'] * 100, width,
              label='Accuracy', color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x, rag_comparison['extraction_success_rate'] * 100, width,
              label='Extraction Success', color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
bars3 = ax.bar(x + width, rag_comparison['f2_score'] * 100, width,
              label='F2 Score', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xlabel('Configuration Type', fontsize=14, fontweight='bold')
ax.set_ylabel('Performance (%)', fontsize=14, fontweight='bold')
ax.set_title('No RAG vs With RAG: Performance Comparison\n(Corrected Extraction Success Calculation)',
            fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(['No RAG', 'With RAG'], fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 105)

plt.tight_layout()

# Save corrected plot
plot_file = output_dir / 'comprehensive_analysis' / 'plot_4_norag_vs_rag_comparison_CORRECTED.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"✅ Saved corrected plot: {plot_file.name}")

# Print LaTeX table format
print("\n=== LaTeX Table Format ===")
print("\\begin{table}[h!]")
print("\\centering")
print("\\caption{RAG impact on performance metrics ($n=48$ configurations per condition).}")
print("\\label{tab:rag_impact}")
print("\\begin{tabular}{l|c|c|c}")
print("\\textbf{Condition} & \\textbf{Mean Accuracy} & \\textbf{Extraction Success} & \\textbf{F2 Score} \\\\ \\hline")

for _, row in rag_comparison.iterrows():
    condition = "With RAG" if row['has_rag'] else "Without RAG"
    print(f"{condition} & {row['triage_accuracy']:.3f} & {row['extraction_success_rate']:.3f} & {row['f2_score']:.3f} \\\\")

print("\\hline")
diff_acc = rag_comparison[rag_comparison['has_rag'] == True]['triage_accuracy'].values[0] - \
           rag_comparison[rag_comparison['has_rag'] == False]['triage_accuracy'].values[0]
diff_ext = rag_comparison[rag_comparison['has_rag'] == True]['extraction_success_rate'].values[0] - \
           rag_comparison[rag_comparison['has_rag'] == False]['extraction_success_rate'].values[0]
diff_f2 = rag_comparison[rag_comparison['has_rag'] == True]['f2_score'].values[0] - \
          rag_comparison[rag_comparison['has_rag'] == False]['f2_score'].values[0]

print(f"\\textbf{{Difference}} & {diff_acc:+.3f} & {diff_ext:+.3f} & {diff_f2:+.3f} \\\\")
print("\\end{tabular}")
print("\\end{table}")

print("\n✅ Done!")
