#!/usr/bin/env python3
"""
Create confusion matrix plots for high_capacity_safe adapter configurations.
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

# Find the specific adapter configuration
target_adapter = "adapter_safe_triage_SmolLM2-135M_4bit_high_capacity_safe"
matching_configs = []

for entry in results:
    adapter_path = entry['config'].get('adapter_path', '')
    if target_adapter in adapter_path:
        matching_configs.append({
            'config_name': entry['config']['test_name'],
            'confusion_matrix': entry['confusion_matrix'],
            'accuracy': entry['triage_accuracy'],
            'f2_score': entry['f2_score'],
            'classification_report': entry['classification_report'],
            'has_rag': entry['config']['rag_config'] is not None,
            'rag_config': entry['config']['rag_config']
        })

print(f"\n✅ Found {len(matching_configs)} configurations with {target_adapter}")

# Show all matching configs
for i, config in enumerate(matching_configs, 1):
    rag_status = "With RAG" if config['has_rag'] else "No RAG"
    print(f"\n{i}. {rag_status}")
    print(f"   Config: {config['config_name']}")
    print(f"   Accuracy: {config['accuracy']:.1%}, F2: {config['f2_score']:.3f}")
    if config['has_rag']:
        rag = config['rag_config']
        print(f"   RAG: {rag.get('chunking_method', 'N/A')}, {rag.get('retrieval_type', 'N/A')}")

# Create output directory
output_dir = Path('/Users/choemanseung/789/hft/evaluation_framework_final/plots_unknown_unknown/confusion_matrices')
output_dir.mkdir(exist_ok=True, parents=True)

# Define class labels
class_labels = ['ED', 'GP', 'HOME', 'UNKNOWN']

# ==================== PLOT CONFUSION MATRICES ====================

# Create subplots for all matching configs
n_configs = len(matching_configs)
fig, axes = plt.subplots(1, n_configs, figsize=(7*n_configs, 6))

if n_configs == 1:
    axes = [axes]

for idx, config in enumerate(matching_configs):
    ax = axes[idx]

    # Get confusion matrix
    cm = np.array(config['confusion_matrix'])

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=class_labels, yticklabels=class_labels,
                cbar=True, square=True, linewidths=1, linecolor='black',
                ax=ax, cbar_kws={'label': 'Count'})

    # Titles and labels
    rag_status = "With RAG" if config['has_rag'] else "No RAG"
    title = f"High Capacity Safe Adapter - {rag_status}\n"
    title += f"Accuracy: {config['accuracy']:.1%} | F2: {config['f2_score']:.3f}"

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
    ax.tick_params(axis='both', labelsize=11)

plt.tight_layout()

# Save plot
plot_file = output_dir / 'high_capacity_safe_confusion_matrices.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved confusion matrix plot: {plot_file.name}")
plt.close()

# ==================== CREATE INDIVIDUAL PLOTS ====================
print("\n📊 Creating individual confusion matrix plots...")

for config in matching_configs:
    fig, ax = plt.subplots(figsize=(8, 7))

    # Get confusion matrix
    cm = np.array(config['confusion_matrix'])

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=class_labels, yticklabels=class_labels,
                cbar=True, square=True, linewidths=2, linecolor='black',
                ax=ax, cbar_kws={'label': 'Number of Cases'},
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})

    # Enhanced title
    rag_status = "With RAG" if config['has_rag'] else "No RAG"
    title = f"Confusion Matrix: High Capacity Safe Adapter ({rag_status})\n"
    title += f"Accuracy: {config['accuracy']:.1%} | F2 Score: {config['f2_score']:.3f}"

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)

    # Add performance metrics text
    report = config['classification_report']
    metrics_text = f"Class Performance:\n"
    for cls in ['ED', 'GP', 'HOME']:
        if cls in report:
            prec = report[cls]['precision']
            rec = report[cls]['recall']
            f1 = report[cls]['f1-score']
            metrics_text += f"{cls}: P={prec:.2f} R={rec:.2f} F1={f1:.2f}\n"

    # Add text box with metrics
    ax.text(1.35, 0.5, metrics_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save individual plot
    rag_suffix = "with_rag" if config['has_rag'] else "no_rag"
    individual_file = output_dir / f'high_capacity_safe_{rag_suffix}_confusion_matrix.png'
    plt.savefig(individual_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {individual_file.name}")
    plt.close()

print("\n✅ All confusion matrix plots created!")
