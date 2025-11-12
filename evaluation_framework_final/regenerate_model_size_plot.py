#!/usr/bin/env python3
"""
Regenerate Model Size Impact plot with line graph showing average vs top config
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load data
data_file = "/Users/choemanseung/789/hft/evaluation_framework_final/plots_unknown_unknown/comprehensive_results_analysis.csv"
output_file = "/Users/choemanseung/789/hft/evaluation_framework_final/plots_unknown_unknown/plots_unknown_tracking/individual_plots_unknown_tracking/07_model_size_impact.png"

print(f"📂 Loading data from: {data_file}")
df = pd.read_csv(data_file)

print(f"✅ Loaded {len(df)} configurations")
print(f"📊 Columns: {df.columns.tolist()}")

# Extract actual model size from config_name (e.g., "SmolLM2-135M_4bit" -> "135M_4bit")
df['full_model_size'] = df['config_name'].str.extract(r'(SmolLM2-\d+M_\d+bit|Gemma-\d+M_\d+bit)')[0]
df['full_model_size'] = df['full_model_size'].str.replace('SmolLM2-', '').str.replace('Gemma-', '')

print(f"\n📋 Extracted model sizes:")
print(df['full_model_size'].value_counts())

# Calculate average and max performance per model size
size_performance = df.groupby('full_model_size').agg({
    'triage_accuracy': ['mean', 'std', 'max'],
}).reset_index()

# Flatten column names
size_performance.columns = ['model_size', 'acc_mean', 'acc_std', 'acc_max']

# Sort by model size for proper line plotting
size_performance['size_num'] = pd.to_numeric(size_performance['model_size'].str.extract(r'(\d+)M')[0], errors='coerce')
size_performance['quant'] = pd.to_numeric(size_performance['model_size'].str.extract(r'(\d+)bit')[0], errors='coerce')

# Remove rows with NaN (couldn't extract size)
size_performance = size_performance.dropna(subset=['size_num', 'quant'])
size_performance['size_num'] = size_performance['size_num'].astype(int)
size_performance['quant'] = size_performance['quant'].astype(int)

size_performance = size_performance.sort_values(['size_num', 'quant']).reset_index(drop=True)

print("\n📈 Performance by Model Size:")
print(size_performance[['model_size', 'acc_mean', 'acc_max']])

# Create plot
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(size_performance))

# Plot lines
ax.plot(x, size_performance['acc_mean'],
        marker='o', linewidth=3, markersize=12,
        label='Average Accuracy', color='#e74c3c',
        markerfacecolor='white', markeredgewidth=2, markeredgecolor='#e74c3c')

ax.plot(x, size_performance['acc_max'],
        marker='s', linewidth=3, markersize=12,
        label='Top Config Accuracy', color='#2ecc71',
        markerfacecolor='white', markeredgewidth=2, markeredgecolor='#2ecc71')

# Add error bars for average accuracy
ax.errorbar(x, size_performance['acc_mean'], yerr=size_performance['acc_std'],
           fmt='none', ecolor='#e74c3c', alpha=0.3, capsize=5, capthick=2)

# Add value labels
for i, row in size_performance.iterrows():
    # Average accuracy label
    ax.text(i, row['acc_mean'] + 0.04, f"{row['acc_mean']:.1%}",
           ha='center', va='bottom', fontsize=10, fontweight='bold',
           color='#c0392b', bbox=dict(boxstyle='round,pad=0.3',
           facecolor='white', edgecolor='#e74c3c', alpha=0.7))

    # Top config accuracy label
    ax.text(i, row['acc_max'] + 0.04, f"{row['acc_max']:.1%}",
           ha='center', va='bottom', fontsize=10, fontweight='bold',
           color='#27ae60', bbox=dict(boxstyle='round,pad=0.3',
           facecolor='white', edgecolor='#2ecc71', alpha=0.7))

# Add gap indicator between average and top
for i, row in size_performance.iterrows():
    gap = row['acc_max'] - row['acc_mean']
    ax.annotate('', xy=(i, row['acc_max']), xytext=(i, row['acc_mean']),
               arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5, alpha=0.5))
    ax.text(i + 0.15, (row['acc_mean'] + row['acc_max']) / 2,
           f"Δ{gap:.1%}", fontsize=9, color='gray', style='italic')

# Styling
ax.set_xlabel('Model Size', fontsize=13, fontweight='bold')
ax.set_ylabel('Triage Accuracy', fontsize=13, fontweight='bold')
ax.set_title('Model Size Impact: Average vs Best Configuration Performance\n(Comparing all configurations per model size)',
            fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(size_performance['model_size'], rotation=45, ha='right', fontsize=11)
ax.legend(fontsize=12, loc='best', framealpha=0.9)
ax.set_ylim(0, max(size_performance['acc_max']) + 0.15)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Plot saved to: {output_file}")
plt.close()

print("\n" + "="*80)
print("🎉 Model Size Impact plot regenerated successfully!")
print("="*80)
