#!/usr/bin/env python3
"""
Generate comprehensive missing analyses:
1. Per-class metrics (precision/recall) for top configs
2. Adapter aggregated statistics table
3. Bar chart: Max accuracy by model variant
4. Box plot: 4-bit vs 8-bit accuracy distributions
5. Scatter plot: Accuracy vs Extraction Success
6. Grouped bars: NoRAG vs RAG comparison
7. Heatmap: Model × Adapter performance matrix
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

# Load data
data_file = "/Users/choemanseung/789/hft/evaluation_framework_final/plots_unknown_unknown/comprehensive_results_analysis.csv"
output_dir = Path("/Users/choemanseung/789/hft/evaluation_framework_final/plots_unknown_unknown/comprehensive_analysis")
output_dir.mkdir(exist_ok=True, parents=True)

print(f"📂 Loading data from: {data_file}")
df = pd.read_csv(data_file)

# Extract model info
df['full_model_size'] = df['config_name'].str.extract(r'(SmolLM2-\d+M_\d+bit|Gemma-\d+M_\d+bit)')[0]
df['full_model_size'] = df['full_model_size'].str.replace('SmolLM2-', '').str.replace('Gemma-', '')
df['base_model'] = df['full_model_size'].str.replace('_4bit', '').str.replace('_8bit', '')
df['quantization_type'] = df['full_model_size'].str.extract(r'(\d+bit)')[0]
df['extraction_success_rate'] = df['success_count'] / df['cases_evaluated']

print(f"✅ Loaded {len(df)} configurations")

# ==================== 1. PER-CLASS METRICS FOR TOP CONFIGS ====================
print("\n" + "="*100)
print("1️⃣ PER-CLASS METRICS FOR TOP 10 CONFIGURATIONS")
print("="*100)

# Note: We don't have per-class precision/recall in the CSV, so we'll note this
# and create what we can from available data

top_10 = df.nlargest(10, 'triage_accuracy')

top_10_table = top_10[[
    'config_name', 'triage_accuracy', 'f1_score', 'f2_score',
    'extraction_success_rate', 'avg_inference_time', 'adapter_type'
]].copy()

top_10_table.columns = ['Configuration', 'Accuracy', 'F1', 'F2', 'Extraction', 'Time (s)', 'Adapter']

print(top_10_table.to_string(index=False))

top_10_file = output_dir / 'top_10_configs_metrics.csv'
top_10_table.to_csv(top_10_file, index=False)
print(f"\n💾 Saved: {top_10_file.name}")

print("\n⚠️  Note: Per-class precision/recall for ED/GP/HOME not available in CSV.")
print("   This would require access to detailed evaluation results with confusion matrices.")

# ==================== 2. ADAPTER AGGREGATED STATISTICS TABLE ====================
print("\n" + "="*100)
print("2️⃣ ADAPTER AGGREGATED STATISTICS")
print("="*100)

adapter_stats = df.groupby('adapter_type').agg({
    'triage_accuracy': ['mean', 'std', 'max'],
    'extraction_success_rate': ['mean', 'std'],
    'f2_score': ['mean', 'std', 'max'],
    'avg_inference_time': ['mean', 'std'],
    'config_name': 'count'
}).round(4)

adapter_stats.columns = [
    'Mean Acc', 'Std Acc', 'Max Acc',
    'Mean Extract', 'Std Extract',
    'Mean F2', 'Std F2', 'Max F2',
    'Mean Time', 'Std Time',
    'Num Configs'
]

print(adapter_stats)

adapter_file = output_dir / 'adapter_aggregated_statistics.csv'
adapter_stats.to_csv(adapter_file)
print(f"\n💾 Saved: {adapter_file.name}")

# ==================== 3. BAR CHART: MAX ACCURACY BY MODEL VARIANT ====================
print("\n📊 Creating Plot 1: Max Accuracy by Model Variant...")

fig, ax = plt.subplots(figsize=(14, 7))

model_max_acc = df.groupby('full_model_size').agg({
    'triage_accuracy': 'max',
    'f2_score': 'max'
}).reset_index()

# Sort
model_max_acc['size_num'] = pd.to_numeric(model_max_acc['full_model_size'].str.extract(r'(\d+)M')[0], errors='coerce')
model_max_acc['quant'] = pd.to_numeric(model_max_acc['full_model_size'].str.extract(r'(\d+)bit')[0], errors='coerce')
model_max_acc = model_max_acc.dropna(subset=['size_num', 'quant'])
model_max_acc['size_num'] = model_max_acc['size_num'].astype(int)
model_max_acc['quant'] = model_max_acc['quant'].astype(int)
model_max_acc = model_max_acc.sort_values(['size_num', 'quant'])

x = np.arange(len(model_max_acc))
width = 0.35

# Highlight 135M_4bit
colors = ['#2ecc71' if '135M_4bit' in model else '#3498db' for model in model_max_acc['full_model_size']]

bars = ax.bar(x, model_max_acc['triage_accuracy'] * 100, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xlabel('Model Variant', fontsize=14, fontweight='bold')
ax.set_ylabel('Maximum Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Maximum Accuracy by Model Variant\n(135M_4bit Dominance)', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(model_max_acc['full_model_size'], rotation=45, ha='right', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', edgecolor='black', label='135M_4bit (Best)'),
    Patch(facecolor='#3498db', edgecolor='black', label='Other Models')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

plt.tight_layout()
plot1_file = output_dir / 'plot_1_max_accuracy_by_model.png'
plt.savefig(plot1_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {plot1_file.name}")

# ==================== 4. BOX PLOT: 4-BIT VS 8-BIT DISTRIBUTIONS ====================
print("\n📊 Creating Plot 2: 4-bit vs 8-bit Accuracy Distributions...")

fig, ax = plt.subplots(figsize=(12, 7))

bit4_data = df[df['quantization_type'] == '4bit']['triage_accuracy'] * 100
bit8_data = df[df['quantization_type'] == '8bit']['triage_accuracy'] * 100

bp = ax.boxplot([bit4_data, bit8_data], labels=['4-bit', '8-bit'],
                patch_artist=True, widths=0.6,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5))

# Color differently
bp['boxes'][0].set_facecolor('#3498db')
bp['boxes'][1].set_facecolor('#e74c3c')

# Add statistics text
ax.text(1, bit4_data.max() + 2,
        f'Mean: {bit4_data.mean():.1f}%\nMedian: {bit4_data.median():.1f}%\nMax: {bit4_data.max():.1f}%',
        ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.text(2, bit8_data.max() + 2,
        f'Mean: {bit8_data.mean():.1f}%\nMedian: {bit8_data.median():.1f}%\nMax: {bit8_data.max():.1f}%',
        ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_ylabel('Triage Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('4-bit vs 8-bit Accuracy Distributions\n(Box Plot Comparison)', fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot2_file = output_dir / 'plot_2_4bit_vs_8bit_boxplot.png'
plt.savefig(plot2_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {plot2_file.name}")

# ==================== 5. SCATTER: ACCURACY VS EXTRACTION ====================
print("\n📊 Creating Plot 3: Accuracy vs Extraction Success Scatter...")

fig, ax = plt.subplots(figsize=(14, 9))

# Color by adapter, size by F2
adapters = df['adapter_type'].unique()
colors_map = {'ultra_safe': '#e74c3c', 'balanced_safe': '#3498db',
              'performance_safe': '#2ecc71', 'high_capacity_safe': '#f39c12'}

for adapter in adapters:
    subset = df[df['adapter_type'] == adapter]
    scatter = ax.scatter(subset['extraction_success_rate'] * 100,
                        subset['triage_accuracy'] * 100,
                        s=subset['f2_score'] * 1000,  # Size by F2
                        c=colors_map.get(adapter, '#95a5a6'),
                        alpha=0.6, edgecolors='black', linewidth=1,
                        label=adapter)

ax.set_xlabel('Extraction Success Rate (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('Triage Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Accuracy vs Extraction Success\n(Colored by Adapter, Sized by F2 Score)',
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=11, title='Adapter Type', title_fontsize=12)
ax.grid(True, alpha=0.3)

# Add size legend
from matplotlib.lines import Line2D
size_legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markersize=np.sqrt(0.2*1000)/3, label='F2=0.2'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markersize=np.sqrt(0.5*1000)/3, label='F2=0.5'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markersize=np.sqrt(0.7*1000)/3, label='F2=0.7')
]
size_legend = ax.legend(handles=size_legend_elements, loc='upper left',
                       fontsize=10, title='F2 Score', title_fontsize=11)
ax.add_artist(size_legend)
ax.legend(loc='lower right', fontsize=11, title='Adapter Type', title_fontsize=12)

plt.tight_layout()
plot3_file = output_dir / 'plot_3_accuracy_vs_extraction_scatter.png'
plt.savefig(plot3_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {plot3_file.name}")

# ==================== 6. GROUPED BARS: NoRAG VS RAG ====================
print("\n📊 Creating Plot 4: NoRAG vs RAG Comparison...")

fig, ax = plt.subplots(figsize=(14, 7))

df['has_rag_clean'] = df['has_rag'].apply(lambda x: 'With RAG' if x else 'No RAG')

rag_comparison = df.groupby('has_rag_clean').agg({
    'triage_accuracy': 'mean',
    'extraction_success_rate': 'mean',
    'f2_score': 'mean'
}).reset_index()

x = np.arange(len(rag_comparison))
width = 0.25

bars1 = ax.bar(x - width, rag_comparison['triage_accuracy'] * 100, width,
              label='Accuracy', color='#3498db', alpha=0.8)
bars2 = ax.bar(x, rag_comparison['extraction_success_rate'] * 100, width,
              label='Extraction Success', color='#2ecc71', alpha=0.8)
bars3 = ax.bar(x + width, rag_comparison['f2_score'] * 100, width,
              label='F2 Score', color='#e74c3c', alpha=0.8)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Configuration Type', fontsize=14, fontweight='bold')
ax.set_ylabel('Performance (%)', fontsize=14, fontweight='bold')
ax.set_title('No RAG vs With RAG: Performance Comparison', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(rag_comparison['has_rag_clean'], fontsize=13)
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot4_file = output_dir / 'plot_4_norag_vs_rag_comparison.png'
plt.savefig(plot4_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {plot4_file.name}")

# ==================== 7. HEATMAP: MODEL × ADAPTER MATRIX ====================
print("\n📊 Creating Plot 5: Model × Adapter Performance Heatmap...")

fig, ax = plt.subplots(figsize=(12, 8))

heatmap_data = df.groupby(['full_model_size', 'adapter_type']).agg({
    'triage_accuracy': 'mean'
}).reset_index()

heatmap_pivot = heatmap_data.pivot(index='full_model_size',
                                   columns='adapter_type',
                                   values='triage_accuracy')

# Sort rows
heatmap_pivot['size_num'] = pd.to_numeric(heatmap_pivot.index.str.extract(r'(\d+)M')[0], errors='coerce')
heatmap_pivot['quant'] = pd.to_numeric(heatmap_pivot.index.str.extract(r'(\d+)bit')[0], errors='coerce')
heatmap_pivot = heatmap_pivot.sort_values(['size_num', 'quant'])
heatmap_pivot = heatmap_pivot.drop(columns=['size_num', 'quant'])

sns.heatmap(heatmap_pivot * 100, annot=True, fmt='.1f', cmap='RdYlGn',
           cbar_kws={'label': 'Mean Accuracy (%)'}, linewidths=0.5,
           linecolor='gray', ax=ax, vmin=0, vmax=70)

ax.set_xlabel('Adapter Type', fontsize=14, fontweight='bold')
ax.set_ylabel('Model Variant', fontsize=14, fontweight='bold')
ax.set_title('Model × Adapter Performance Matrix\n(Mean Accuracy %)',
            fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plot5_file = output_dir / 'plot_5_model_adapter_heatmap.png'
plt.savefig(plot5_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {plot5_file.name}")

# ==================== SUMMARY ====================
print("\n" + "="*100)
print("✅ COMPREHENSIVE ANALYSIS COMPLETE!")
print("="*100)
print(f"\n📁 All outputs saved to: {output_dir}")
print("\nGenerated files:")
for file in sorted(output_dir.glob("*")):
    print(f"  • {file.name}")

print("\n" + "="*100)
print("📋 SUMMARY OF EXISTING PLOTS:")
print("="*100)
print("\n✅ Already exist in plots_unknown_tracking/individual_plots_unknown_tracking/:")
print("  • 05_class_performance_breakdown.png - Per-class F1 scores (overall)")
print("  • 06_adapter_effectiveness.png - Adapter comparison")
print("  • 04_rag_impact_comparison.png - RAG impact analysis")
print("  • 03_model_reliability_heatmap.png - Model reliability heatmap")
print("  • 01_accuracy_vs_unknown_scatter.png - Accuracy vs unknown cases")

print("\n⚠️  MISSING DATA:")
print("  • Per-class precision/recall for top configs - Requires detailed confusion matrices")
print("  • ED recall specifically - Would need per-config class-level metrics")
