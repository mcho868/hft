#!/usr/bin/env python3
"""
Generate comprehensive model size analysis:
1. Table showing average vs top config performance
2. Plot comparing 4bit vs 8bit models across all metrics
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load data
data_file = "/Users/choemanseung/789/hft/evaluation_framework_final/plots_unknown_unknown/comprehensive_results_analysis.csv"
output_dir = Path("/Users/choemanseung/789/hft/evaluation_framework_final/plots_unknown_unknown/plots_unknown_tracking/individual_plots_unknown_tracking")

print(f"📂 Loading data from: {data_file}")
df = pd.read_csv(data_file)

print(f"✅ Loaded {len(df)} configurations")

# Extract actual model size from config_name
df['full_model_size'] = df['config_name'].str.extract(r'(SmolLM2-\d+M_\d+bit|Gemma-\d+M_\d+bit)')[0]
df['full_model_size'] = df['full_model_size'].str.replace('SmolLM2-', '').str.replace('Gemma-', '')

# ==================== TABLE 1: Average vs Top Config ====================
print("\n" + "="*100)
print("TABLE 1: MODEL SIZE PERFORMANCE SUMMARY (Average vs Top Configuration)")
print("="*100)

size_performance = df.groupby('full_model_size').agg({
    'triage_accuracy': ['mean', 'std', 'min', 'max'],
    'f1_score': ['mean', 'max'],
    'f2_score': ['mean', 'max'],
    'avg_inference_time': ['mean', 'min', 'max']
}).reset_index()

# Flatten column names
size_performance.columns = [
    'model_size',
    'acc_mean', 'acc_std', 'acc_min', 'acc_max',
    'f1_mean', 'f1_max',
    'f2_mean', 'f2_max',
    'time_mean', 'time_min', 'time_max'
]

# Sort
size_performance['size_num'] = pd.to_numeric(size_performance['model_size'].str.extract(r'(\d+)M')[0], errors='coerce')
size_performance['quant'] = pd.to_numeric(size_performance['model_size'].str.extract(r'(\d+)bit')[0], errors='coerce')
size_performance = size_performance.dropna(subset=['size_num', 'quant'])
size_performance['size_num'] = size_performance['size_num'].astype(int)
size_performance['quant'] = size_performance['quant'].astype(int)
size_performance = size_performance.sort_values(['size_num', 'quant']).reset_index(drop=True)

# Calculate gap
size_performance['acc_gap'] = size_performance['acc_max'] - size_performance['acc_mean']

# Format for display
display_table = size_performance[[
    'model_size', 'acc_mean', 'acc_max', 'acc_gap',
    'f1_mean', 'f1_max', 'f2_mean', 'f2_max',
    'time_mean', 'time_min', 'time_max'
]].copy()

display_table['acc_mean'] = (display_table['acc_mean'] * 100).round(1).astype(str) + '%'
display_table['acc_max'] = (display_table['acc_max'] * 100).round(1).astype(str) + '%'
display_table['acc_gap'] = (display_table['acc_gap'] * 100).round(1).astype(str) + '%'
display_table['f1_mean'] = (display_table['f1_mean'] * 100).round(1).astype(str) + '%'
display_table['f1_max'] = (display_table['f1_max'] * 100).round(1).astype(str) + '%'
display_table['f2_mean'] = (display_table['f2_mean'] * 100).round(1).astype(str) + '%'
display_table['f2_max'] = (display_table['f2_max'] * 100).round(1).astype(str) + '%'
display_table['time_mean'] = display_table['time_mean'].round(3).astype(str) + 's'
display_table['time_min'] = display_table['time_min'].round(3).astype(str) + 's'
display_table['time_max'] = display_table['time_max'].round(3).astype(str) + 's'

display_table.columns = [
    'Model Size', 'Avg Accuracy', 'Top Accuracy', 'Gap',
    'Avg F1', 'Top F1', 'Avg F2', 'Top F2',
    'Avg Time', 'Min Time', 'Max Time'
]

print(display_table.to_string(index=False))

# Save table
table_file = output_dir / 'model_size_performance_table.csv'
size_performance.to_csv(table_file, index=False)
print(f"\n💾 Table saved to: {table_file}")

# ==================== TABLE 2: 4bit vs 8bit Comparison ====================
print("\n" + "="*100)
print("TABLE 2: 4-BIT vs 8-BIT QUANTIZATION COMPARISON")
print("="*100)

df['base_model'] = df['full_model_size'].str.replace('_4bit', '').str.replace('_8bit', '')
df['quantization_type'] = df['full_model_size'].str.extract(r'(\d+bit)')[0]

quant_comparison = df.groupby('quantization_type').agg({
    'triage_accuracy': ['mean', 'std', 'max'],
    'f1_score': ['mean', 'max'],
    'f2_score': ['mean', 'max'],
    'avg_inference_time': ['mean', 'std']
}).reset_index()

quant_comparison.columns = [
    'Quantization',
    'Accuracy Mean', 'Accuracy Std', 'Accuracy Max',
    'F1 Mean', 'F1 Max',
    'F2 Mean', 'F2 Max',
    'Inference Time Mean', 'Inference Time Std'
]

print(quant_comparison.to_string(index=False))

quant_table_file = output_dir / 'quantization_comparison_table.csv'
quant_comparison.to_csv(quant_table_file, index=False)
print(f"\n💾 Table saved to: {quant_table_file}")

# ==================== PLOT: 4bit vs 8bit Comparison ====================
print("\n📊 Generating 4-bit vs 8-bit comparison plot...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('4-bit vs 8-bit Quantization: Comprehensive Performance Comparison',
             fontsize=18, fontweight='bold', y=1.00)

# Group by base model and quantization
comparison_data = df.groupby(['base_model', 'quantization_type']).agg({
    'triage_accuracy': 'mean',
    'f1_score': 'mean',
    'f2_score': 'mean',
    'avg_inference_time': 'mean'
}).reset_index()

# Pivot for easier plotting
pivot_acc = comparison_data.pivot(index='base_model', columns='quantization_type', values='triage_accuracy')
pivot_f1 = comparison_data.pivot(index='base_model', columns='quantization_type', values='f1_score')
pivot_f2 = comparison_data.pivot(index='base_model', columns='quantization_type', values='f2_score')
pivot_time = comparison_data.pivot(index='base_model', columns='quantization_type', values='avg_inference_time')

# Also calculate extraction success rate if available
if 'success_count' in df.columns and 'cases_evaluated' in df.columns:
    df['extraction_success_rate'] = df['success_count'] / df['cases_evaluated']
    extraction_data = df.groupby(['base_model', 'quantization_type']).agg({
        'extraction_success_rate': 'mean'
    }).reset_index()
    pivot_extraction = extraction_data.pivot(index='base_model', columns='quantization_type', values='extraction_success_rate')
else:
    pivot_extraction = None

models = pivot_acc.index
x = np.arange(len(models))
width = 0.35

# Plot 1: Accuracy
ax1 = axes[0, 0]
bars1 = ax1.bar(x - width/2, pivot_acc['4bit'] * 100, width, label='4-bit', color='#3498db', alpha=0.8)
bars2 = ax1.bar(x + width/2, pivot_acc['8bit'] * 100, width, label='8-bit', color='#e74c3c', alpha=0.8)
ax1.set_ylabel('Accuracy (%)', fontweight='bold')
ax1.set_title('Triage Accuracy', fontweight='bold', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# Plot 2: F1 Score
ax2 = axes[0, 1]
bars1 = ax2.bar(x - width/2, pivot_f1['4bit'] * 100, width, label='4-bit', color='#3498db', alpha=0.8)
bars2 = ax2.bar(x + width/2, pivot_f1['8bit'] * 100, width, label='8-bit', color='#e74c3c', alpha=0.8)
ax2.set_ylabel('F1 Score (%)', fontweight='bold')
ax2.set_title('F1 Score', fontweight='bold', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# Plot 3: F2 Score
ax3 = axes[0, 2]
bars1 = ax3.bar(x - width/2, pivot_f2['4bit'] * 100, width, label='4-bit', color='#3498db', alpha=0.8)
bars2 = ax3.bar(x + width/2, pivot_f2['8bit'] * 100, width, label='8-bit', color='#e74c3c', alpha=0.8)
ax3.set_ylabel('F2 Score (%)', fontweight='bold')
ax3.set_title('F2 Score (Recall-weighted)', fontweight='bold', fontsize=14)
ax3.set_xticks(x)
ax3.set_xticklabels(models, rotation=45, ha='right')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# Plot 4: Extraction Success Rate (if available)
ax4 = axes[1, 0]
if pivot_extraction is not None:
    bars1 = ax4.bar(x - width/2, pivot_extraction['4bit'] * 100, width, label='4-bit', color='#3498db', alpha=0.8)
    bars2 = ax4.bar(x + width/2, pivot_extraction['8bit'] * 100, width, label='8-bit', color='#e74c3c', alpha=0.8)
    ax4.set_ylabel('Extraction Success (%)', fontweight='bold')
    ax4.set_title('Extraction Success Rate', fontweight='bold', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
else:
    ax4.text(0.5, 0.5, 'Extraction data not available', ha='center', va='center', transform=ax4.transAxes)
    ax4.axis('off')

# Plot 5: Inference Latency
ax5 = axes[1, 1]
bars1 = ax5.bar(x - width/2, pivot_time['4bit'], width, label='4-bit', color='#3498db', alpha=0.8)
bars2 = ax5.bar(x + width/2, pivot_time['8bit'], width, label='8-bit', color='#e74c3c', alpha=0.8)
ax5.set_ylabel('Inference Time (s)', fontweight='bold')
ax5.set_title('Average Inference Latency', fontweight='bold', fontsize=14)
ax5.set_xticks(x)
ax5.set_xticklabels(models, rotation=45, ha='right')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}s', ha='center', va='bottom', fontsize=9)

# Plot 6: Summary metrics (radar/spider chart or summary text)
ax6 = axes[1, 2]
ax6.axis('off')

# Calculate overall statistics
bit4_stats = df[df['quantization_type'] == '4bit'].agg({
    'triage_accuracy': 'mean',
    'f1_score': 'mean',
    'f2_score': 'mean',
    'avg_inference_time': 'mean'
})

bit8_stats = df[df['quantization_type'] == '8bit'].agg({
    'triage_accuracy': 'mean',
    'f1_score': 'mean',
    'f2_score': 'mean',
    'avg_inference_time': 'mean'
})

summary_text = f"""
QUANTIZATION SUMMARY
{'='*40}

4-BIT MODELS:
  Accuracy: {bit4_stats['triage_accuracy']*100:.1f}%
  F1 Score: {bit4_stats['f1_score']*100:.1f}%
  F2 Score: {bit4_stats['f2_score']*100:.1f}%
  Avg Time: {bit4_stats['avg_inference_time']:.3f}s

8-BIT MODELS:
  Accuracy: {bit8_stats['triage_accuracy']*100:.1f}%
  F1 Score: {bit8_stats['f1_score']*100:.1f}%
  F2 Score: {bit8_stats['f2_score']*100:.1f}%
  Avg Time: {bit8_stats['avg_inference_time']:.3f}s

RELATIVE PERFORMANCE:
  Accuracy Diff: {(bit4_stats['triage_accuracy']-bit8_stats['triage_accuracy'])*100:+.1f}%
  F1 Diff: {(bit4_stats['f1_score']-bit8_stats['f1_score'])*100:+.1f}%
  F2 Diff: {(bit4_stats['f2_score']-bit8_stats['f2_score'])*100:+.1f}%
  Time Diff: {(bit4_stats['avg_inference_time']-bit8_stats['avg_inference_time']):+.3f}s

WINNER: {'4-bit' if bit4_stats['triage_accuracy'] > bit8_stats['triage_accuracy'] else '8-bit'}
(Based on accuracy)
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plot_file = output_dir / '08_4bit_vs_8bit_comprehensive.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"✅ Plot saved to: {plot_file}")
plt.close()

print("\n" + "="*100)
print("✅ MODEL SIZE ANALYSIS COMPLETE!")
print("="*100)
print(f"\n📁 Generated files:")
print(f"  • {table_file.name}")
print(f"  • {quant_table_file.name}")
print(f"  • {plot_file.name}")
