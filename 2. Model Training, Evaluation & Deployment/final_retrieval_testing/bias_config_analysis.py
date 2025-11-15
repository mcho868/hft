#!/usr/bin/env python3
"""
Source Bias Configuration Analysis
Analyzes performance differences between different source bias configurations
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

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

def create_bias_analysis_plots(df, output_dir):
    """Create comprehensive bias configuration analysis plots"""

    # Add chunking type
    df['chunking_type'] = df['chunking_method'].apply(extract_chunking_type)

    # Create output directory
    bias_output_dir = output_dir / "bias_configuration_analysis"
    bias_output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n📊 Creating Bias Configuration Analysis Plots")
    print("="*80)

    # Plot 1: Overall comparison by bias config
    fig, ax = plt.subplots(figsize=(12, 6))

    bias_configs = df['bias_config'].unique()
    k_metrics = ['pass_at_1', 'pass_at_2', 'pass_at_3', 'pass_at_4', 'pass_at_5', 'pass_at_10', 'pass_at_20']
    x = np.arange(len(k_metrics))
    width = 0.35

    for i, bias in enumerate(bias_configs):
        bias_data = df[df['bias_config'] == bias]
        means = [bias_data[metric].mean() * 100 for metric in k_metrics]
        offset = (i - len(bias_configs)/2 + 0.5) * width
        ax.bar(x + offset, means, width, label=bias, alpha=0.8)

    ax.set_xlabel('Pass@k Metric', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Performance Comparison: Bias Configurations', fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Pass@1', 'Pass@2', 'Pass@3', 'Pass@4', 'Pass@5', 'Pass@10', 'Pass@20'])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{bias_output_dir}/plot_1_bias_overall_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot 1: Overall bias comparison")

    # Plot 2: Bias config by chunking type
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Bias Configuration Performance by Chunking Type', fontsize=16, fontweight='bold')

    chunking_types = sorted(df['chunking_type'].unique())
    for idx, chunking_type in enumerate(chunking_types):
        ax = axes[idx // 3, idx % 3]

        chunking_data = df[df['chunking_type'] == chunking_type]

        for bias in bias_configs:
            bias_chunking_data = chunking_data[chunking_data['bias_config'] == bias]
            if len(bias_chunking_data) > 0:
                k_values = [1, 2, 3, 4, 5, 10, 20]
                accuracies = [bias_chunking_data[f'pass_at_{k}'].mean() * 100 for k in k_values]
                ax.plot(k_values, accuracies, marker='o', linewidth=2, markersize=8, label=bias, alpha=0.8)

        ax.set_xlabel('k', fontsize=10, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
        ax.set_title(f'{chunking_type.replace("_", " ").title()}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_xticks(k_values)
        ax.set_xticklabels(k_values)

    plt.tight_layout()
    plt.savefig(f"{bias_output_dir}/plot_2_bias_by_chunking_type.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot 2: Bias by chunking type")

    # Plot 3: Bias config by retrieval type
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Bias Configuration Performance by Retrieval Type', fontsize=16, fontweight='bold')

    retrieval_types = df['retrieval_type'].unique()
    for idx, ret_type in enumerate(retrieval_types):
        ax = axes[idx]

        ret_data = df[df['retrieval_type'] == ret_type]

        for bias in bias_configs:
            bias_ret_data = ret_data[ret_data['bias_config'] == bias]
            k_values = [1, 2, 3, 4, 5, 10, 20]
            accuracies = [bias_ret_data[f'pass_at_{k}'].mean() * 100 for k in k_values]
            ax.plot(k_values, accuracies, marker='s', linewidth=3, markersize=10, label=bias, alpha=0.8)

        ax.set_xlabel('k (Number of Retrieved Chunks)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{ret_type.replace("_", " ").title()}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_xticks(k_values)
        ax.set_xticklabels(k_values)

    plt.tight_layout()
    plt.savefig(f"{bias_output_dir}/plot_3_bias_by_retrieval_type.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot 3: Bias by retrieval type")

    # Plot 4: Heatmap showing bias impact across configurations
    fig, ax = plt.subplots(figsize=(14, 8))

    # Calculate difference between bias configs for each chunking+retrieval combo
    pivot_data = []

    for chunking_type in chunking_types:
        for ret_type in retrieval_types:
            subset = df[(df['chunking_type'] == chunking_type) & (df['retrieval_type'] == ret_type)]

            if len(subset) > 0:
                for bias in bias_configs:
                    bias_subset = subset[subset['bias_config'] == bias]
                    if len(bias_subset) > 0:
                        pivot_data.append({
                            'Configuration': f"{chunking_type}\n{ret_type}",
                            'Bias': bias,
                            'Pass@5': bias_subset['pass_at_5'].mean() * 100,
                            'Pass@10': bias_subset['pass_at_10'].mean() * 100
                        })

    pivot_df = pd.DataFrame(pivot_data)

    # Create heatmap for Pass@10
    heatmap_data = pivot_df.pivot_table(values='Pass@10', index='Configuration', columns='Bias', aggfunc='mean')

    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
                cbar_kws={'label': 'Pass@10 Accuracy (%)'}, linewidths=0.5, linecolor='gray')
    ax.set_title('Bias Configuration Impact Heatmap (Pass@10)', fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel('Bias Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Configuration Type', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{bias_output_dir}/plot_4_bias_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot 4: Bias impact heatmap")

    # Plot 5: Difference plot (healthify_focused vs diverse)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance Difference: healthify_focused vs diverse', fontsize=16, fontweight='bold')

    # Calculate differences
    differences = []

    for idx, row in df.iterrows():
        if row['bias_config'] == 'healthify_focused':
            # Find matching diverse config
            matching = df[
                (df['chunking_method'] == row['chunking_method']) &
                (df['retrieval_type'] == row['retrieval_type']) &
                (df['bias_config'] == 'diverse')
            ]

            if len(matching) > 0:
                diverse_row = matching.iloc[0]
                differences.append({
                    'chunking_type': extract_chunking_type(row['chunking_method']),
                    'retrieval_type': row['retrieval_type'],
                    'diff_pass_at_1': (row['pass_at_1'] - diverse_row['pass_at_1']) * 100,
                    'diff_pass_at_5': (row['pass_at_5'] - diverse_row['pass_at_5']) * 100,
                    'diff_pass_at_10': (row['pass_at_10'] - diverse_row['pass_at_10']) * 100,
                    'diff_pass_at_20': (row['pass_at_20'] - diverse_row['pass_at_20']) * 100
                })

    diff_df = pd.DataFrame(differences)

    # Plot 5a: By chunking type
    ax1 = axes[0, 0]
    chunking_diffs = diff_df.groupby('chunking_type').agg({
        'diff_pass_at_1': 'mean',
        'diff_pass_at_5': 'mean',
        'diff_pass_at_10': 'mean',
        'diff_pass_at_20': 'mean'
    })

    x = np.arange(len(chunking_diffs.index))
    width = 0.2

    ax1.bar(x - 1.5*width, chunking_diffs['diff_pass_at_1'], width, label='Pass@1', alpha=0.8)
    ax1.bar(x - 0.5*width, chunking_diffs['diff_pass_at_5'], width, label='Pass@5', alpha=0.8)
    ax1.bar(x + 0.5*width, chunking_diffs['diff_pass_at_10'], width, label='Pass@10', alpha=0.8)
    ax1.bar(x + 1.5*width, chunking_diffs['diff_pass_at_20'], width, label='Pass@20', alpha=0.8)

    ax1.set_xlabel('Chunking Type', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Difference (%)', fontsize=11, fontweight='bold')
    ax1.set_title('By Chunking Type', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(chunking_diffs.index, rotation=45, ha='right')
    ax1.legend(fontsize=9)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 5b: By retrieval type
    ax2 = axes[0, 1]
    retrieval_diffs = diff_df.groupby('retrieval_type').agg({
        'diff_pass_at_1': 'mean',
        'diff_pass_at_5': 'mean',
        'diff_pass_at_10': 'mean',
        'diff_pass_at_20': 'mean'
    })

    x = np.arange(len(retrieval_diffs.index))

    ax2.bar(x - 1.5*width, retrieval_diffs['diff_pass_at_1'], width, label='Pass@1', alpha=0.8)
    ax2.bar(x - 0.5*width, retrieval_diffs['diff_pass_at_5'], width, label='Pass@5', alpha=0.8)
    ax2.bar(x + 0.5*width, retrieval_diffs['diff_pass_at_10'], width, label='Pass@10', alpha=0.8)
    ax2.bar(x + 1.5*width, retrieval_diffs['diff_pass_at_20'], width, label='Pass@20', alpha=0.8)

    ax2.set_xlabel('Retrieval Type', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Difference (%)', fontsize=11, fontweight='bold')
    ax2.set_title('By Retrieval Type', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(retrieval_diffs.index, rotation=45, ha='right')
    ax2.legend(fontsize=9)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 5c: Distribution of differences
    ax3 = axes[1, 0]

    metrics = ['diff_pass_at_1', 'diff_pass_at_5', 'diff_pass_at_10', 'diff_pass_at_20']
    metric_labels = ['Pass@1', 'Pass@5', 'Pass@10', 'Pass@20']

    bp = ax3.boxplot([diff_df[m] for m in metrics], labels=metric_labels, patch_artist=True)

    for patch, color in zip(bp['boxes'], ['#3498db', '#2ecc71', '#e67e22', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax3.set_ylabel('Difference (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Distribution of Differences', fontsize=12, fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 5d: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
    BIAS CONFIGURATION COMPARISON
    {'='*50}

    Average Differences (healthify_focused - diverse):

    Pass@1:  {diff_df['diff_pass_at_1'].mean():+.2f}%
    Pass@5:  {diff_df['diff_pass_at_5'].mean():+.2f}%
    Pass@10: {diff_df['diff_pass_at_10'].mean():+.2f}%
    Pass@20: {diff_df['diff_pass_at_20'].mean():+.2f}%

    Standard Deviations:

    Pass@1:  {diff_df['diff_pass_at_1'].std():.2f}%
    Pass@5:  {diff_df['diff_pass_at_5'].std():.2f}%
    Pass@10: {diff_df['diff_pass_at_10'].std():.2f}%
    Pass@20: {diff_df['diff_pass_at_20'].std():.2f}%

    Interpretation:
    {'='*50}

    Positive: healthify_focused performs better
    Negative: diverse performs better
    Near zero: No significant difference

    Total configurations compared: {len(diff_df)}
    """

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{bias_output_dir}/plot_5_bias_difference_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot 5: Bias difference analysis")

    # Plot 6: Source distribution analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Source Distribution by Bias Configuration', fontsize=16, fontweight='bold')

    for idx, bias in enumerate(bias_configs):
        ax = axes[idx]

        bias_data = df[df['bias_config'] == bias]

        # Calculate average source distribution
        avg_healthify = bias_data['source_distribution'].apply(lambda x: x['healthify']).mean()
        avg_mayo = bias_data['source_distribution'].apply(lambda x: x['mayo']).mean()
        avg_nhs = bias_data['source_distribution'].apply(lambda x: x['nhs']).mean()

        total = avg_healthify + avg_mayo + avg_nhs

        sources = ['Healthify', 'Mayo', 'NHS']
        percentages = [avg_healthify/total*100, avg_mayo/total*100, avg_nhs/total*100]
        colors = ['#3498db', '#e67e22', '#2ecc71']

        wedges, texts, autotexts = ax.pie(percentages, labels=sources, autopct='%1.1f%%',
                                          colors=colors, startangle=90, textprops={'fontsize': 11})

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax.set_title(f'{bias.replace("_", " ").title()}\n({total:.0f} total chunks)',
                    fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{bias_output_dir}/plot_6_source_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot 6: Source distribution")

    print(f"\n{'='*80}")
    print(f"✅ All bias configuration plots created!")
    print(f"📁 Output: {bias_output_dir}")

    return bias_output_dir

def create_bias_summary_table(df, output_dir):
    """Create summary table for bias configurations"""

    bias_output_dir = output_dir / "bias_configuration_analysis"
    bias_output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n📊 Creating Bias Configuration Summary Tables")
    print("="*80)

    # Table 1: Overall comparison
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

    print("\nTable 1: Overall Performance by Bias Configuration")
    print("-"*80)
    print(bias_summary)

    bias_summary.to_csv(f"{bias_output_dir}/bias_overall_summary.csv")
    print(f"\n💾 Saved: bias_overall_summary.csv")

    # Table 2: By chunking type and bias
    df['chunking_type'] = df['chunking_method'].apply(extract_chunking_type)

    chunking_bias_summary = df.groupby(['chunking_type', 'bias_config']).agg({
        'pass_at_5': 'mean',
        'pass_at_10': 'mean',
        'pass_at_20': 'mean'
    }).round(4)

    print("\n\nTable 2: Performance by Chunking Type and Bias")
    print("-"*80)
    print(chunking_bias_summary)

    chunking_bias_summary.to_csv(f"{bias_output_dir}/bias_by_chunking_type.csv")
    print(f"\n💾 Saved: bias_by_chunking_type.csv")

def main():
    # Configuration
    results_file = "/Users/choemanseung/789/hft/final_retrieval_testing/results/optimized_comprehensive/optimized_summary_20251010_095218.json"
    output_dir = Path("/Users/choemanseung/789/hft/analysis_output")

    print("="*80)
    print("🚀 Bias Configuration Analysis")
    print("="*80)
    print(f"📂 Input: {results_file}")
    print(f"📁 Output: {output_dir}")

    # Load data
    df = load_results(results_file)
    print(f"\n✅ Loaded {len(df)} configurations")
    print(f"📊 Bias configurations: {df['bias_config'].unique()}")

    # Create tables
    create_bias_summary_table(df, output_dir)

    # Create plots
    bias_dir = create_bias_analysis_plots(df, output_dir)

    print("\n" + "="*80)
    print("✅ BIAS CONFIGURATION ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n📁 All outputs in: {bias_dir}")
    print("\nGenerated files:")
    for file in sorted(bias_dir.glob("*")):
        print(f"  • {file.name}")

if __name__ == "__main__":
    main()
