#!/usr/bin/env python3
"""
Create confusion matrix plots for all configurations in final_test_results.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Load JSON
json_path = Path('/Users/choemanseung/789/hft/evaluation_framework_final/testing_framework_final/final_test_results_20250930_211951.json')
with open(json_path, 'r') as f:
    data = json.load(f)

# Extract results
results = data['results']
print(f"✅ Loaded {len(results)} configurations")

# Create output directory
output_dir = Path('/Users/choemanseung/789/hft/evaluation_framework_final/plots_unknown_unknown/confusion_matrices_all')
output_dir.mkdir(exist_ok=True, parents=True)

# Define class labels
class_labels = ['ED', 'GP', 'HOME', 'UNKNOWN']

# ==================== CREATE COMBINED PLOT ====================
print("\n📊 Creating combined confusion matrix plot...")

n_configs = len(results)
n_cols = min(3, n_configs)
n_rows = (n_configs + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 7*n_rows))
axes = axes.flatten() if n_configs > 1 else [axes]

for idx, entry in enumerate(results):
    ax = axes[idx]

    # Get confusion matrix
    cm = np.array(entry['confusion_matrix'])

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=class_labels, yticklabels=class_labels,
                cbar=True, square=True, linewidths=1, linecolor='black',
                ax=ax, cbar_kws={'label': 'Count'},
                annot_kws={'fontsize': 11, 'fontweight': 'bold'})

    # Extract config info
    config_name = entry['config']['test_name']
    accuracy = entry['triage_accuracy']
    f2_score = entry['f2_score']

    # Simplified title
    has_rag = entry['config']['rag_config'] is not None
    rag_status = "RAG" if has_rag else "NoRAG"

    # Extract adapter name
    adapter_path = entry['config'].get('adapter_path', '')
    if 'high_capacity_safe' in adapter_path:
        adapter_name = 'high_capacity_safe'
    elif 'balanced_safe' in adapter_path:
        adapter_name = 'balanced_safe'
    elif 'performance_safe' in adapter_path:
        adapter_name = 'performance_safe'
    elif 'ultra_safe' in adapter_path:
        adapter_name = 'ultra_safe'
    else:
        adapter_name = 'unknown'

    title = f"{adapter_name} - {rag_status}\n"
    title += f"Acc: {accuracy:.1%} | F2: {f2_score:.3f}"

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax.set_ylabel('True', fontsize=11, fontweight='bold')
    ax.tick_params(axis='both', labelsize=10)

# Hide extra subplots
for idx in range(n_configs, len(axes)):
    axes[idx].axis('off')

plt.suptitle('Confusion Matrices - All Configurations',
             fontsize=18, fontweight='bold', y=1.0)
plt.tight_layout()

# Save combined plot
combined_file = output_dir / 'all_confusion_matrices_combined.png'
plt.savefig(combined_file, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {combined_file.name}")
plt.close()

# ==================== CREATE INDIVIDUAL PLOTS ====================
print("\n📊 Creating individual confusion matrix plots...")

for idx, entry in enumerate(results):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get confusion matrix
    cm = np.array(entry['confusion_matrix'])

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=class_labels, yticklabels=class_labels,
                cbar=True, square=True, linewidths=2, linecolor='black',
                ax=ax, cbar_kws={'label': 'Number of Cases', 'pad': 0.02},
                annot_kws={'fontsize': 16, 'fontweight': 'bold'})

    # Extract config info
    config_name = entry['config']['test_name']
    accuracy = entry['triage_accuracy']
    f2_score = entry['f2_score']
    f1_score = entry['f1_score']
    has_rag = entry['config']['rag_config'] is not None

    # Extract adapter name
    adapter_path = entry['config'].get('adapter_path', '')
    if 'high_capacity_safe' in adapter_path:
        adapter_name = 'High Capacity Safe'
    elif 'balanced_safe' in adapter_path:
        adapter_name = 'Balanced Safe'
    elif 'performance_safe' in adapter_path:
        adapter_name = 'Performance Safe'
    elif 'ultra_safe' in adapter_path:
        adapter_name = 'Ultra Safe'
    else:
        adapter_name = 'Unknown Adapter'

    # Title
    rag_status = "With RAG" if has_rag else "No RAG"
    title = f"Confusion Matrix: {adapter_name} ({rag_status})\n"
    title += f"Accuracy: {accuracy:.1%} | F1: {f1_score:.3f} | F2: {f2_score:.3f}"

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)

    # Add performance metrics text
    report = entry['classification_report']
    metrics_text = "Class Performance:\n" + "="*25 + "\n"

    for cls in ['ED', 'GP', 'HOME']:
        if cls in report:
            prec = report[cls]['precision']
            rec = report[cls]['recall']
            f1 = report[cls]['f1-score']
            supp = report[cls]['support']
            metrics_text += f"{cls:6s}: P={prec:.2f} R={rec:.2f}\n"
            metrics_text += f"       F1={f1:.2f} N={supp:.0f}\n"

    # Add text box with metrics
    ax.text(1.02, 0.5, metrics_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='center', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, pad=1))

    plt.tight_layout()

    # Save individual plot
    # Create safe filename
    safe_name = config_name.replace('/', '_').replace(' ', '_')
    individual_file = output_dir / f'cm_{idx+1:02d}_{safe_name[:80]}.png'
    plt.savefig(individual_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: cm_{idx+1:02d}_{adapter_name.replace(' ', '_')}_{rag_status.replace(' ', '_')}.png")
    plt.close()

print("\n✅ All confusion matrix plots created!")
print(f"📁 Output directory: {output_dir}")
