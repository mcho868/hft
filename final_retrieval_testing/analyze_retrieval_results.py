#!/usr/bin/env python3
"""
Analyze retrieval results and generate comprehensive visualizations
- Average pass@k per configuration type
- Top performers per configuration
- Pass@k curves showing accuracy vs k
- Sweet spot analysis for context length vs accuracy
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import defaultdict

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_results(json_file):
    """Load results from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def extract_chunking_type(chunking_method):
    """Extract chunking type from method name"""
    method = chunking_method.lower()
    if 'contextual' in method:
        return 'contextual'
    elif 'sentence' in method:
        return 'sentence'
    elif 'fixed' in method:
        return 'fixed'
    elif 'paragraph' in method:
        return 'paragraph'
    elif 'structured_agent' in method:
        return 'structured_agent'
    elif 'agent' in method:
        return 'agent'
    else:
        return 'other'

def extract_chunk_size(chunking_method):
    """Extract approximate chunk size from method name"""
    import re
    method = chunking_method.lower()

    # Look for patterns like c512, c768, t384, etc.
    match = re.search(r'[ct](\d+)', method)
    if match:
        return int(match.group(1))

    # Default sizes for different types
    if 'contextual' in method:
        return 1024  # typical contextual size
    elif 'sentence' in method:
        return 512
    elif 'fixed' in method:
        return 512
    elif 'paragraph' in method:
        return 768
    else:
        return 512  # default

def create_summary_tables(df, output_dir):
    """Create summary tables for average pass@k per configuration type"""

    # Add derived columns
    df['chunking_type'] = df['chunking_method'].apply(extract_chunking_type)
    df['chunk_size'] = df['chunking_method'].apply(extract_chunk_size)

    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    # Table 1: Average Pass@k by Chunking Type
    print("\n📊 Table 1: Average Pass@k by Chunking Type")
    print("-"*80)

    chunking_summary = df.groupby('chunking_type').agg({
        'pass_at_1': 'mean',
        'pass_at_2': 'mean',
        'pass_at_3': 'mean',
        'pass_at_4': 'mean',
        'pass_at_5': 'mean',
        'pass_at_10': 'mean',
        'pass_at_20': 'mean',
        'avg_retrieval_time': 'mean',
        'config_name': 'count'
    }).round(4)

    chunking_summary.columns = ['Pass@1', 'Pass@2', 'Pass@3', 'Pass@4', 'Pass@5', 'Pass@10', 'Pass@20', 'Avg Time (s)', 'Count']
    print(chunking_summary)
    chunking_summary.to_csv(f"{output_dir}/avg_passk_by_chunking_type.csv")

    # Table 2: Average Pass@k by Retrieval Type
    print("\n📊 Table 2: Average Pass@k by Retrieval Type")
    print("-"*80)

    retrieval_summary = df.groupby('retrieval_type').agg({
        'pass_at_1': 'mean',
        'pass_at_2': 'mean',
        'pass_at_3': 'mean',
        'pass_at_4': 'mean',
        'pass_at_5': 'mean',
        'pass_at_10': 'mean',
        'pass_at_20': 'mean',
        'avg_retrieval_time': 'mean',
        'config_name': 'count'
    }).round(4)

    retrieval_summary.columns = ['Pass@1', 'Pass@2', 'Pass@3', 'Pass@4', 'Pass@5', 'Pass@10', 'Pass@20', 'Avg Time (s)', 'Count']
    print(retrieval_summary)
    retrieval_summary.to_csv(f"{output_dir}/avg_passk_by_retrieval_type.csv")

    # Table 3: Average Pass@k by Bias Config
    print("\n📊 Table 3: Average Pass@k by Bias Configuration")
    print("-"*80)

    bias_summary = df.groupby('bias_config').agg({
        'pass_at_1': 'mean',
        'pass_at_2': 'mean',
        'pass_at_3': 'mean',
        'pass_at_4': 'mean',
        'pass_at_5': 'mean',
        'pass_at_10': 'mean',
        'pass_at_20': 'mean',
        'avg_retrieval_time': 'mean',
        'config_name': 'count'
    }).round(4)

    bias_summary.columns = ['Pass@1', 'Pass@2', 'Pass@3', 'Pass@4', 'Pass@5', 'Pass@10', 'Pass@20', 'Avg Time (s)', 'Count']
    print(bias_summary)
    bias_summary.to_csv(f"{output_dir}/avg_passk_by_bias_config.csv")

    # Table 4: Top 20 Performers Overall
    print("\n🏆 Table 4: Top 20 Overall Performers (by Pass@5)")
    print("-"*80)

    top_performers = df.nlargest(20, 'pass_at_5')[[
        'config_name', 'chunking_type', 'retrieval_type', 'bias_config',
        'pass_at_1', 'pass_at_2', 'pass_at_3', 'pass_at_4', 'pass_at_5', 'pass_at_10', 'pass_at_20',
        'avg_retrieval_time'
    ]].copy()

    top_performers[['pass_at_1', 'pass_at_2', 'pass_at_3', 'pass_at_4', 'pass_at_5', 'pass_at_10', 'pass_at_20']] = \
        top_performers[['pass_at_1', 'pass_at_2', 'pass_at_3', 'pass_at_4', 'pass_at_5', 'pass_at_10', 'pass_at_20']].round(4)
    top_performers['avg_retrieval_time'] = top_performers['avg_retrieval_time'].round(4)

    print(top_performers.to_string(index=False))
    top_performers.to_csv(f"{output_dir}/top_20_overall_performers.csv", index=False)

    return chunking_summary, retrieval_summary, bias_summary, top_performers

def plot_passk_curves(df, output_dir):
    """Plot Pass@k curves showing sweet spots"""

    k_values = [1, 2, 3, 4, 5, 10, 20]

    # Add derived columns
    df['chunking_type'] = df['chunking_method'].apply(extract_chunking_type)
    df['chunk_size'] = df['chunking_method'].apply(extract_chunk_size)

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Pass@k Performance Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Pass@k curves by chunking type
    ax1 = axes[0, 0]
    chunking_types = df['chunking_type'].unique()

    for chunking_type in chunking_types:
        subset = df[df['chunking_type'] == chunking_type]
        avg_passk = [
            subset['pass_at_1'].mean(),
            subset['pass_at_2'].mean(),
            subset['pass_at_3'].mean(),
            subset['pass_at_4'].mean(),
            subset['pass_at_5'].mean(),
            subset['pass_at_10'].mean(),
            subset['pass_at_20'].mean()
        ]
        ax1.plot(k_values, avg_passk, marker='o', linewidth=2, markersize=8, label=chunking_type)

    ax1.set_xlabel('k (Number of Retrieved Chunks)', fontsize=12)
    ax1.set_ylabel('Pass@k Accuracy', fontsize=12)
    ax1.set_title('Pass@k by Chunking Type', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_xticks(k_values)
    ax1.set_xticklabels(k_values)

    # Plot 2: Pass@k curves by retrieval type
    ax2 = axes[0, 1]
    retrieval_types = df['retrieval_type'].unique()

    for ret_type in retrieval_types:
        subset = df[df['retrieval_type'] == ret_type]
        avg_passk = [
            subset['pass_at_1'].mean(),
            subset['pass_at_2'].mean(),
            subset['pass_at_3'].mean(),
            subset['pass_at_4'].mean(),
            subset['pass_at_5'].mean(),
            subset['pass_at_10'].mean(),
            subset['pass_at_20'].mean()
        ]
        ax2.plot(k_values, avg_passk, marker='s', linewidth=2, markersize=8, label=ret_type)

    ax2.set_xlabel('k (Number of Retrieved Chunks)', fontsize=12)
    ax2.set_ylabel('Pass@k Accuracy', fontsize=12)
    ax2.set_title('Pass@k by Retrieval Type', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_xticks(k_values)
    ax2.set_xticklabels(k_values)

    # Plot 3: Top 10 configurations Pass@k curves
    ax3 = axes[1, 0]
    top_10 = df.nlargest(10, 'pass_at_5')

    for idx, row in top_10.iterrows():
        config_label = f"{row['chunking_type']}_{row['retrieval_type']}"
        passk_values = [
            row['pass_at_1'],
            row['pass_at_2'],
            row['pass_at_3'],
            row['pass_at_4'],
            row['pass_at_5'],
            row['pass_at_10'],
            row['pass_at_20']
        ]
        ax3.plot(k_values, passk_values, marker='o', linewidth=1.5, alpha=0.7, label=config_label[:30])

    ax3.set_xlabel('k (Number of Retrieved Chunks)', fontsize=12)
    ax3.set_ylabel('Pass@k Accuracy', fontsize=12)
    ax3.set_title('Top 10 Configurations Pass@k Curves', fontsize=14, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_xticks(k_values)
    ax3.set_xticklabels(k_values)

    # Plot 4: Chunk size vs accuracy sweet spot
    ax4 = axes[1, 1]

    # Create chunk size bins
    df['chunk_size_bin'] = pd.cut(df['chunk_size'], bins=[0, 384, 512, 768, 1024, 2048],
                                   labels=['<384', '384-512', '512-768', '768-1024', '>1024'])

    chunk_size_summary = df.groupby('chunk_size_bin').agg({
        'pass_at_1': 'mean',
        'pass_at_5': 'mean',
        'pass_at_10': 'mean',
        'pass_at_20': 'mean'
    })

    x_pos = np.arange(len(chunk_size_summary.index))
    width = 0.2

    ax4.bar(x_pos - 1.5*width, chunk_size_summary['pass_at_1'], width, label='Pass@1', alpha=0.8)
    ax4.bar(x_pos - 0.5*width, chunk_size_summary['pass_at_5'], width, label='Pass@5', alpha=0.8)
    ax4.bar(x_pos + 0.5*width, chunk_size_summary['pass_at_10'], width, label='Pass@10', alpha=0.8)
    ax4.bar(x_pos + 1.5*width, chunk_size_summary['pass_at_20'], width, label='Pass@20', alpha=0.8)

    ax4.set_xlabel('Chunk Size (tokens)', fontsize=12)
    ax4.set_ylabel('Pass@k Accuracy', fontsize=12)
    ax4.set_title('Chunk Size vs Accuracy Sweet Spot', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(chunk_size_summary.index)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/passk_curves_analysis.png", dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved: {output_dir}/passk_curves_analysis.png")

    # Additional plot: Marginal gains analysis
    fig2, ax = plt.subplots(figsize=(12, 7))

    # Calculate marginal gains for top configurations
    top_configs = df.nlargest(15, 'pass_at_5')

    for idx, row in top_configs.iterrows():
        config_label = f"{row['chunking_type'][:8]}_{row['retrieval_type'][:8]}"

        # Calculate marginal gains
        marginal_gains = [
            row['pass_at_1'],
            row['pass_at_2'] - row['pass_at_1'],
            row['pass_at_3'] - row['pass_at_2'],
            row['pass_at_4'] - row['pass_at_3'],
            row['pass_at_5'] - row['pass_at_4'],
            row['pass_at_10'] - row['pass_at_5'],
            row['pass_at_20'] - row['pass_at_10']
        ]

        # Cumulative for stacking
        cumulative = np.cumsum(marginal_gains)

        ax.plot(k_values, cumulative, marker='o', linewidth=2, alpha=0.7, label=config_label)

    ax.set_xlabel('k (Number of Retrieved Chunks)', fontsize=12)
    ax.set_ylabel('Cumulative Accuracy', fontsize=12)
    ax.set_title('Cumulative Accuracy Growth: Finding the Sweet Spot', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xticks(k_values)
    ax.set_xticklabels(k_values)

    # Add sweet spot annotation
    ax.axvline(x=5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Typical Sweet Spot (k=5)')
    ax.axvline(x=10, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Extended Sweet Spot (k=10)')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/cumulative_accuracy_sweet_spot.png", dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {output_dir}/cumulative_accuracy_sweet_spot.png")

def analyze_sweet_spot(df, output_dir):
    """Analyze and identify the sweet spot for k"""

    k_values = [1, 2, 3, 4, 5, 10, 20]

    # Calculate marginal gains for all configurations
    marginal_gains = []

    for idx, row in df.iterrows():
        gains = {
            'config': row['config_name'],
            'k1_to_k2': row['pass_at_2'] - row['pass_at_1'],
            'k2_to_k3': row['pass_at_3'] - row['pass_at_2'],
            'k3_to_k4': row['pass_at_4'] - row['pass_at_3'],
            'k4_to_k5': row['pass_at_5'] - row['pass_at_4'],
            'k5_to_k10': row['pass_at_10'] - row['pass_at_5'],
            'k10_to_k20': row['pass_at_20'] - row['pass_at_10']
        }
        marginal_gains.append(gains)

    marginal_df = pd.DataFrame(marginal_gains)

    # Calculate average marginal gains
    avg_marginal = {
        'k1→k2': marginal_df['k1_to_k2'].mean(),
        'k2→k3': marginal_df['k2_to_k3'].mean(),
        'k3→k4': marginal_df['k3_to_k4'].mean(),
        'k4→k5': marginal_df['k4_to_k5'].mean(),
        'k5→k10': marginal_df['k5_to_k10'].mean(),
        'k10→k20': marginal_df['k10_to_k20'].mean()
    }

    print("\n📈 Sweet Spot Analysis: Marginal Accuracy Gains")
    print("="*80)
    for transition, gain in avg_marginal.items():
        print(f"{transition:10s}: {gain:+.4f} ({gain*100:+.2f}%)")

    # Identify diminishing returns point
    print("\n🎯 Diminishing Returns Analysis:")
    print("-"*80)
    print(f"Largest marginal gain: {max(avg_marginal, key=avg_marginal.get)} = {max(avg_marginal.values()):.4f}")
    print(f"Smallest marginal gain: {min(avg_marginal, key=avg_marginal.get)} = {min(avg_marginal.values()):.4f}")

    # Save marginal gains
    marginal_summary = pd.DataFrame([avg_marginal])
    marginal_summary.to_csv(f"{output_dir}/marginal_gains_analysis.csv", index=False)
    print(f"\n💾 Saved marginal gains to: {output_dir}/marginal_gains_analysis.csv")

def main():
    # Configuration
    results_file = "/Users/choemanseung/789/hft/final_retrieval_testing/results/optimized_comprehensive/optimized_summary_20251010_095218.json"
    output_dir = Path("/Users/choemanseung/789/hft/analysis_output")
    output_dir.mkdir(exist_ok=True, parents=True)

    print("🚀 Retrieval Results Analysis")
    print("="*80)
    print(f"📂 Input: {results_file}")
    print(f"📁 Output: {output_dir}")
    print("="*80)

    # Load data
    df = load_results(results_file)
    print(f"\n✅ Loaded {len(df)} configurations")

    # Generate summary tables
    chunking_summary, retrieval_summary, bias_summary, top_performers = create_summary_tables(df, output_dir)

    # Plot Pass@k curves
    print("\n📊 Generating Pass@k curve visualizations...")
    plot_passk_curves(df, output_dir)

    # Analyze sweet spot
    analyze_sweet_spot(df, output_dir)

    print("\n" + "="*80)
    print("✅ Analysis Complete!")
    print("="*80)
    print(f"📁 All outputs saved to: {output_dir}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob("*")):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()
