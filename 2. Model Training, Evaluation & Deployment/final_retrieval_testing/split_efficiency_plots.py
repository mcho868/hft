#!/usr/bin/env python3
"""
Split the comprehensive efficiency analysis into 8 separate plot files
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(json_file):
    """Load results from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def calculate_efficiency_metrics(df):
    """Calculate efficiency-adjusted metrics"""
    avg_passk = {
        1: df['pass_at_1'].mean(),
        2: df['pass_at_2'].mean(),
        3: df['pass_at_3'].mean(),
        4: df['pass_at_4'].mean(),
        5: df['pass_at_5'].mean(),
        10: df['pass_at_10'].mean(),
        20: df['pass_at_20'].mean()
    }

    transitions = [
        ("k=1→k=2", 1, 2),
        ("k=2→k=3", 2, 3),
        ("k=3→k=4", 3, 4),
        ("k=4→k=5", 4, 5),
        ("k=5→k=10", 5, 10),
        ("k=10→k=20", 10, 20)
    ]

    efficiency_data = []
    for name, k_from, k_to in transitions:
        chunks_added = k_to - k_from
        absolute_gain = avg_passk[k_to] - avg_passk[k_from]
        gain_per_chunk = absolute_gain / chunks_added

        efficiency_data.append({
            'Transition': name,
            'k_from': k_from,
            'k_to': k_to,
            'Chunks Added': chunks_added,
            'Absolute Gain': absolute_gain,
            'Absolute Gain (%)': absolute_gain * 100,
            'Gain per Chunk': gain_per_chunk,
            'Gain per Chunk (%)': gain_per_chunk * 100,
            'Pass@k_from': avg_passk[k_from],
            'Pass@k_to': avg_passk[k_to]
        })

    return pd.DataFrame(efficiency_data), avg_passk

def plot_1_efficiency_per_chunk(efficiency_df, output_dir):
    """Plot 1: Efficiency per chunk (bar chart)"""
    fig, ax = plt.subplots(figsize=(10, 6))

    transitions = efficiency_df['Transition'].values
    efficiency = efficiency_df['Gain per Chunk (%)'].values
    colors = ['#2ecc71' if e == max(efficiency) else '#3498db' if e > 2 else '#e74c3c' if e == 0 else '#95a5a6'
              for e in efficiency]

    bars = ax.bar(range(len(transitions)), efficiency, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(transitions)))
    ax.set_xticklabels(transitions, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Gain per Chunk Added (%)', fontsize=13, fontweight='bold')
    ax.set_title('Efficiency: Gain per Chunk Added', fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.8)

    for i, (bar, val) in enumerate(zip(bars, efficiency)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom' if val > 0 else 'top',
                fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot_1_efficiency_per_chunk.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot 1 saved: plot_1_efficiency_per_chunk.png")

def plot_2_absolute_gains(efficiency_df, output_dir):
    """Plot 2: Absolute gains (bar chart)"""
    fig, ax = plt.subplots(figsize=(10, 6))

    transitions = efficiency_df['Transition'].values
    absolute_gains = efficiency_df['Absolute Gain (%)'].values
    chunks_added = efficiency_df['Chunks Added'].values

    colors = ['#e67e22' if g == max(absolute_gains) else '#3498db' if g > 5 else '#95a5a6'
               for g in absolute_gains]

    bars = ax.bar(range(len(transitions)), absolute_gains, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(transitions)))
    ax.set_xticklabels(transitions, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Absolute Gain (%)', fontsize=13, fontweight='bold')
    ax.set_title('Absolute Gains (Total Performance Increase)', fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')

    for i, (bar, val, chunks) in enumerate(zip(bars, absolute_gains, chunks_added)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%\n({chunks} chunk{"s" if chunks > 1 else ""})',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot_2_absolute_gains.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot 2 saved: plot_2_absolute_gains.png")

def plot_3_cumulative_accuracy(avg_passk, output_dir):
    """Plot 3: Cumulative accuracy curve"""
    fig, ax = plt.subplots(figsize=(10, 6))

    k_values = [1, 2, 3, 4, 5, 10, 20]
    accuracies = [avg_passk[k] * 100 for k in k_values]

    ax.plot(k_values, accuracies, marker='o', linewidth=3, markersize=12,
             color='#2c3e50', markerfacecolor='#e74c3c', markeredgewidth=2, markeredgecolor='#c0392b')
    ax.fill_between(k_values, accuracies, alpha=0.2, color='#3498db')

    ax.axvline(x=5, color='#2ecc71', linestyle='--', linewidth=2.5, alpha=0.7, label='k=5 (Best Efficiency)')
    ax.axvline(x=10, color='#f39c12', linestyle='--', linewidth=2.5, alpha=0.7, label='k=10 (Balanced)')

    ax.set_xlabel('k (Number of Retrieved Chunks)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Pass@k Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Cumulative Accuracy vs Context Length', fontsize=15, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.set_xticks(k_values)
    ax.set_xticklabels(k_values)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=11)

    for k, acc in zip(k_values, accuracies):
        ax.annotate(f'{acc:.1f}%', (k, acc), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot_3_cumulative_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot 3 saved: plot_3_cumulative_accuracy.png")

def plot_4_efficiency_vs_chunks(efficiency_df, output_dir):
    """Plot 4: Efficiency vs Chunks Added scatter"""
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter_sizes = efficiency_df['Absolute Gain (%)'] * 30
    scatter = ax.scatter(efficiency_df['Chunks Added'], efficiency_df['Gain per Chunk (%)'],
                         s=scatter_sizes, alpha=0.6, c=efficiency_df['Gain per Chunk (%)'],
                         cmap='RdYlGn', edgecolors='black', linewidth=2)

    for idx, row in efficiency_df.iterrows():
        ax.annotate(row['Transition'],
                    (row['Chunks Added'], row['Gain per Chunk (%)']),
                    textcoords="offset points", xytext=(8,8), ha='left', fontsize=10, fontweight='bold')

    ax.set_xlabel('Chunks Added', fontsize=13, fontweight='bold')
    ax.set_ylabel('Efficiency (% per chunk)', fontsize=13, fontweight='bold')
    ax.set_title('Efficiency vs Context Cost', fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax, label='Efficiency (%)')
    cbar.set_label('Efficiency (%)', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot_4_efficiency_vs_chunks.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot 4 saved: plot_4_efficiency_vs_chunks.png")

def plot_5_efficiency_heatmap(efficiency_df, output_dir):
    """Plot 5: Efficiency heatmap"""
    fig, ax = plt.subplots(figsize=(12, 6))

    transitions = efficiency_df['Transition'].values
    heatmap_data = []
    metrics = ['Chunks Added', 'Absolute Gain (%)', 'Gain per Chunk (%)']

    for metric in metrics:
        values = efficiency_df[metric].values
        normalized = (values - values.min()) / (values.max() - values.min() + 1e-10)
        heatmap_data.append(normalized)

    heatmap_data = np.array(heatmap_data)

    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(transitions)))
    ax.set_xticklabels(transitions, rotation=45, ha='right', fontsize=10)
    ax.set_title('Normalized Metrics Comparison', fontsize=15, fontweight='bold', pad=20)
    cbar = plt.colorbar(im, ax=ax, label='Normalized Score')
    cbar.set_label('Normalized Score', fontsize=11, fontweight='bold')

    for i in range(len(metrics)):
        for j in range(len(transitions)):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot_5_efficiency_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot 5 saved: plot_5_efficiency_heatmap.png")

def plot_6_roi_analysis(efficiency_df, output_dir):
    """Plot 6: ROI (Return on Investment) analysis"""
    fig, ax = plt.subplots(figsize=(10, 6))

    transitions = efficiency_df['Transition'].values
    roi = efficiency_df['Absolute Gain (%)'] / efficiency_df['Chunks Added']

    colors_roi = ['#2ecc71' if r == max(roi) else '#3498db' if r > 2 else '#e74c3c' if r == 0 else '#95a5a6'
                  for r in roi]

    bars_roi = ax.barh(range(len(transitions)), roi, color=colors_roi, alpha=0.8,
                        edgecolor='black', linewidth=1.5)
    ax.set_yticks(range(len(transitions)))
    ax.set_yticklabels(transitions, fontsize=11)
    ax.set_xlabel('ROI (% gain per chunk)', fontsize=13, fontweight='bold')
    ax.set_title('Return on Investment (ROI)', fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')

    for i, (bar, val) in enumerate(zip(bars_roi, roi)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' {val:.2f}%', ha='left', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot_6_roi_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot 6 saved: plot_6_roi_analysis.png")

def plot_7_marginal_utility(efficiency_df, output_dir):
    """Plot 7: Marginal utility curve"""
    fig, ax = plt.subplots(figsize=(12, 6))

    k_plot = []
    marginal_gains = []

    for idx, row in efficiency_df.iterrows():
        k_plot.append(row['k_from'])
        marginal_gains.append(row['Gain per Chunk (%)'])

    ax.plot(k_plot, marginal_gains, marker='s', linewidth=3, markersize=14,
             color='#34495e', markerfacecolor='#e74c3c', markeredgewidth=2.5,
             markeredgecolor='#c0392b', label='Marginal Efficiency')

    ax.axhline(y=4, color='#2ecc71', linestyle=':', linewidth=2.5, alpha=0.6, label='High Efficiency (>4%)')
    ax.axhline(y=2, color='#f39c12', linestyle=':', linewidth=2.5, alpha=0.6, label='Medium Efficiency (>2%)')

    ax.fill_between(k_plot, marginal_gains, 4, where=[m >= 4 for m in marginal_gains],
                     alpha=0.3, color='#2ecc71', label='High Efficiency Zone')
    ax.fill_between(k_plot, marginal_gains, 0, where=[m < 2 for m in marginal_gains],
                     alpha=0.3, color='#e74c3c', label='Low Efficiency Zone')

    ax.set_xlabel('Starting k value', fontsize=13, fontweight='bold')
    ax.set_ylabel('Marginal Efficiency (% per chunk)', fontsize=13, fontweight='bold')
    ax.set_title('Marginal Utility Curve: Diminishing Returns Analysis', fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xticks(k_plot)
    ax.set_xticklabels(k_plot)

    for k, gain in zip(k_plot, marginal_gains):
        ax.annotate(f'{gain:.2f}%', (k, gain), textcoords="offset points",
                    xytext=(0,12), ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot_7_marginal_utility.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot 7 saved: plot_7_marginal_utility.png")

def plot_8_recommendations(efficiency_df, output_dir):
    """Plot 8: Recommendations panel"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    best_efficiency = efficiency_df.loc[efficiency_df['Gain per Chunk (%)'].idxmax()]
    best_absolute = efficiency_df.loc[efficiency_df['Absolute Gain (%)'].idxmax()]

    recommendations_text = f"""
OPTIMAL RETRIEVAL STRATEGY
{'='*50}

BEST EFFICIENCY (Per Chunk)
{best_efficiency['Transition']}
• Efficiency: {best_efficiency['Gain per Chunk (%)']:.2f}% per chunk
• Chunks Added: {best_efficiency['Chunks Added']}
• Total Gain: {best_efficiency['Absolute Gain (%)']:.2f}%
• From {best_efficiency['Pass@k_from']*100:.1f}% → {best_efficiency['Pass@k_to']*100:.1f}%

LARGEST ABSOLUTE GAIN
{best_absolute['Transition']}
• Total Gain: {best_absolute['Absolute Gain (%)']:.2f}%
• Chunks Added: {best_absolute['Chunks Added']}
• Efficiency: {best_absolute['Gain per Chunk (%)']:.2f}% per chunk
• From {best_absolute['Pass@k_from']*100:.1f}% → {best_absolute['Pass@k_to']*100:.1f}%

RECOMMENDATIONS BY CONTEXT BUDGET
{'='*50}

Tight Context Budget (≤5 chunks):
  → Recommended: k=5
  → Accuracy: 38.22%
  → Best efficiency at k=4→k=5 (+8.82% per chunk)
  → Provides good accuracy with minimal context

Moderate Context Budget (≤10 chunks):
  → Recommended: k=10
  → Accuracy: 58.61%
  → Additional 5 chunks for +20.39% gain
  → Efficiency: 4.08% per chunk (still good ROI)

Large Context Budget (>10 chunks):
  → Recommended: STOP at k=10
  → No improvement beyond k=10
  → k=10→k=20 provides 0% gain
  → Avoid wasting precious context window

EFFICIENCY SUMMARY
{'='*50}
Rank 1: k=4→k=5  (8.82% per chunk)
Rank 2: k=1→k=2  (4.43% per chunk)
Rank 3: k=5→k=10 (4.08% per chunk)
Rank 4: k=2→k=3  (2.50% per chunk)
Rank 5: k=3→k=4  (1.95% per chunk)
Rank 6: k=10→k=20 (0.00% per chunk) - AVOID!
    """

    ax.text(0.05, 0.95, recommendations_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#fffacd', alpha=0.9, edgecolor='black', linewidth=2))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot_8_recommendations.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot 8 saved: plot_8_recommendations.png")

def main():
    # Configuration
    results_file = "/Users/choemanseung/789/hft/final_retrieval_testing/results/optimized_comprehensive/optimized_summary_20251010_095218.json"
    output_dir = Path("/Users/choemanseung/789/hft/analysis_output/efficiency_analysis_comprehensive")
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*80)
    print("🚀 Creating Individual Efficiency Analysis Plots")
    print("="*80)
    print(f"📂 Input: {results_file}")
    print(f"📁 Output Directory: {output_dir}")
    print("="*80)

    # Load data
    df = load_results(results_file)
    print(f"\n✅ Loaded {len(df)} configurations")

    # Calculate efficiency metrics
    print("📊 Calculating efficiency metrics...")
    efficiency_df, avg_passk = calculate_efficiency_metrics(df)

    # Create individual plots
    print("\n📈 Generating individual plots...")
    print("-"*80)

    plot_1_efficiency_per_chunk(efficiency_df, output_dir)
    plot_2_absolute_gains(efficiency_df, output_dir)
    plot_3_cumulative_accuracy(avg_passk, output_dir)
    plot_4_efficiency_vs_chunks(efficiency_df, output_dir)
    plot_5_efficiency_heatmap(efficiency_df, output_dir)
    plot_6_roi_analysis(efficiency_df, output_dir)
    plot_7_marginal_utility(efficiency_df, output_dir)
    plot_8_recommendations(efficiency_df, output_dir)

    print("\n" + "="*80)
    print("✅ ALL PLOTS CREATED SUCCESSFULLY!")
    print("="*80)
    print(f"\n📁 Output directory: {output_dir}")
    print("\nGenerated plots:")
    for i, file in enumerate(sorted(output_dir.glob("plot_*.png")), 1):
        print(f"  {i}. {file.name}")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
