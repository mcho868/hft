#!/usr/bin/env python3
"""
Efficiency-adjusted analysis of retrieval performance
Creates tables and visualizations showing gain per chunk added
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

    # Average pass@k values
    avg_passk = {
        1: df['pass_at_1'].mean(),
        2: df['pass_at_2'].mean(),
        3: df['pass_at_3'].mean(),
        4: df['pass_at_4'].mean(),
        5: df['pass_at_5'].mean(),
        10: df['pass_at_10'].mean(),
        20: df['pass_at_20'].mean()
    }

    # Define transitions
    transitions = [
        ("k=1→k=2", 1, 2),
        ("k=2→k=3", 2, 3),
        ("k=3→k=4", 3, 4),
        ("k=4→k=5", 4, 5),
        ("k=5→k=10", 5, 10),
        ("k=10→k=20", 10, 20)
    ]

    # Calculate efficiency metrics
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

def create_efficiency_tables(efficiency_df, output_dir):
    """Create comprehensive efficiency tables"""

    print("\n" + "="*100)
    print("TABLE 1: EFFICIENCY-ADJUSTED MARGINAL GAINS ANALYSIS")
    print("="*100)

    # Main efficiency table
    table1 = efficiency_df[[
        'Transition', 'Chunks Added', 'Absolute Gain (%)',
        'Gain per Chunk (%)', 'Pass@k_from', 'Pass@k_to'
    ]].copy()

    table1['Pass@k_from'] = (table1['Pass@k_from'] * 100).round(2)
    table1['Pass@k_to'] = (table1['Pass@k_to'] * 100).round(2)
    table1['Absolute Gain (%)'] = table1['Absolute Gain (%)'].round(2)
    table1['Gain per Chunk (%)'] = table1['Gain per Chunk (%)'].round(2)

    table1.columns = ['Transition', 'Chunks Added', 'Total Gain (%)',
                      'Efficiency (% per chunk)', 'From (%)', 'To (%)']

    print(table1.to_string(index=False))
    table1.to_csv(f"{output_dir}/efficiency_marginal_gains.csv", index=False)
    print(f"\n💾 Saved: {output_dir}/efficiency_marginal_gains.csv")

    # Efficiency ranking table
    print("\n" + "="*100)
    print("TABLE 2: EFFICIENCY RANKING (Best to Worst)")
    print("="*100)

    table2 = efficiency_df[['Transition', 'Chunks Added', 'Absolute Gain (%)',
                            'Gain per Chunk (%)']].copy().sort_values(
        'Gain per Chunk (%)', ascending=False
    )

    table2['Absolute Gain (%)'] = table2['Absolute Gain (%)'].round(2)
    table2['Gain per Chunk (%)'] = table2['Gain per Chunk (%)'].round(2)
    table2['Efficiency Rank'] = range(1, len(table2) + 1)

    table2 = table2[['Efficiency Rank', 'Transition', 'Chunks Added',
                     'Absolute Gain (%)', 'Gain per Chunk (%)']]

    print(table2.to_string(index=False))
    table2.to_csv(f"{output_dir}/efficiency_ranking.csv", index=False)
    print(f"\n💾 Saved: {output_dir}/efficiency_ranking.csv")

    # Cumulative efficiency table
    print("\n" + "="*100)
    print("TABLE 3: CUMULATIVE PERFORMANCE BY K VALUE")
    print("="*100)

    cumulative_data = []
    k_values = [1, 2, 3, 4, 5, 10, 20]

    for i, k in enumerate(k_values):
        row = efficiency_df[efficiency_df['k_to'] == k]
        if len(row) > 0:
            cumulative_data.append({
                'k': k,
                'Accuracy (%)': row['Pass@k_to'].values[0] * 100,
                'Total Chunks': k,
                'Gain from k=1 (%)': (row['Pass@k_to'].values[0] - efficiency_df.iloc[0]['Pass@k_from']) * 100,
                'Context Cost': k
            })
        elif k == 1:
            cumulative_data.append({
                'k': k,
                'Accuracy (%)': efficiency_df.iloc[0]['Pass@k_from'] * 100,
                'Total Chunks': k,
                'Gain from k=1 (%)': 0,
                'Context Cost': k
            })

    table3 = pd.DataFrame(cumulative_data)
    table3['Accuracy (%)'] = table3['Accuracy (%)'].round(2)
    table3['Gain from k=1 (%)'] = table3['Gain from k=1 (%)'].round(2)

    print(table3.to_string(index=False))
    table3.to_csv(f"{output_dir}/cumulative_performance_by_k.csv", index=False)
    print(f"\n💾 Saved: {output_dir}/cumulative_performance_by_k.csv")

def plot_efficiency_analysis(efficiency_df, avg_passk, output_dir):
    """Create comprehensive efficiency visualization"""

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Main title
    fig.suptitle('Retrieval Efficiency Analysis: Context Length vs Accuracy Trade-offs',
                 fontsize=18, fontweight='bold', y=0.98)

    # Plot 1: Efficiency per chunk (bar chart)
    ax1 = fig.add_subplot(gs[0, 0])
    transitions = efficiency_df['Transition'].values
    efficiency = efficiency_df['Gain per Chunk (%)'].values
    colors = ['#2ecc71' if e == max(efficiency) else '#3498db' if e > 2 else '#e74c3c' if e == 0 else '#95a5a6'
              for e in efficiency]

    bars = ax1.bar(range(len(transitions)), efficiency, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(transitions)))
    ax1.set_xticklabels(transitions, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel('Gain per Chunk Added (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Efficiency: Gain per Chunk Added', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='black', linewidth=0.8)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, efficiency)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom' if val > 0 else 'top',
                fontsize=9, fontweight='bold')

    # Plot 2: Absolute gains (bar chart)
    ax2 = fig.add_subplot(gs[0, 1])
    absolute_gains = efficiency_df['Absolute Gain (%)'].values
    chunks_added = efficiency_df['Chunks Added'].values

    colors2 = ['#e67e22' if g == max(absolute_gains) else '#3498db' if g > 5 else '#95a5a6'
               for g in absolute_gains]

    bars2 = ax2.bar(range(len(transitions)), absolute_gains, color=colors2, alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(transitions)))
    ax2.set_xticklabels(transitions, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('Absolute Gain (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Absolute Gains (Total Performance Increase)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (bar, val, chunks) in enumerate(zip(bars2, absolute_gains, chunks_added)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%\n({chunks} chunk{"s" if chunks > 1 else ""})',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Plot 3: Cumulative accuracy curve
    ax3 = fig.add_subplot(gs[0, 2])
    k_values = [1, 2, 3, 4, 5, 10, 20]
    accuracies = [avg_passk[k] * 100 for k in k_values]

    ax3.plot(k_values, accuracies, marker='o', linewidth=3, markersize=10,
             color='#2c3e50', markerfacecolor='#e74c3c', markeredgewidth=2, markeredgecolor='#c0392b')
    ax3.fill_between(k_values, accuracies, alpha=0.2, color='#3498db')

    # Highlight sweet spots
    ax3.axvline(x=5, color='#2ecc71', linestyle='--', linewidth=2, alpha=0.7, label='k=5 (Best Efficiency)')
    ax3.axvline(x=10, color='#f39c12', linestyle='--', linewidth=2, alpha=0.7, label='k=10 (Balanced)')

    ax3.set_xlabel('k (Number of Retrieved Chunks)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Pass@k Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Cumulative Accuracy vs Context Length', fontsize=13, fontweight='bold')
    ax3.set_xscale('log')
    ax3.set_xticks(k_values)
    ax3.set_xticklabels(k_values)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='lower right', fontsize=9)

    # Add accuracy labels
    for k, acc in zip(k_values, accuracies):
        ax3.annotate(f'{acc:.1f}%', (k, acc), textcoords="offset points",
                    xytext=(0,8), ha='center', fontsize=9, fontweight='bold')

    # Plot 4: Efficiency vs Chunks Added scatter
    ax4 = fig.add_subplot(gs[1, 0])
    scatter_sizes = efficiency_df['Absolute Gain (%)'] * 20
    scatter = ax4.scatter(efficiency_df['Chunks Added'], efficiency_df['Gain per Chunk (%)'],
                         s=scatter_sizes, alpha=0.6, c=efficiency_df['Gain per Chunk (%)'],
                         cmap='RdYlGn', edgecolors='black', linewidth=2)

    # Add labels
    for idx, row in efficiency_df.iterrows():
        ax4.annotate(row['Transition'],
                    (row['Chunks Added'], row['Gain per Chunk (%)']),
                    textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)

    ax4.set_xlabel('Chunks Added', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Efficiency (% per chunk)', fontsize=12, fontweight='bold')
    ax4.set_title('Efficiency vs Context Cost', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Efficiency (%)')

    # Plot 5: Efficiency heatmap
    ax5 = fig.add_subplot(gs[1, 1])

    # Create data for heatmap
    heatmap_data = []
    metrics = ['Chunks Added', 'Absolute Gain (%)', 'Gain per Chunk (%)']

    for metric in metrics:
        values = efficiency_df[metric].values
        # Normalize to 0-1 scale
        normalized = (values - values.min()) / (values.max() - values.min() + 1e-10)
        heatmap_data.append(normalized)

    heatmap_data = np.array(heatmap_data)

    im = ax5.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
    ax5.set_yticks(range(len(metrics)))
    ax5.set_yticklabels(metrics, fontsize=10)
    ax5.set_xticks(range(len(transitions)))
    ax5.set_xticklabels(transitions, rotation=45, ha='right', fontsize=9)
    ax5.set_title('Normalized Metrics Comparison', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax5, label='Normalized Score')

    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(transitions)):
            text = ax5.text(j, i, f'{heatmap_data[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8, fontweight='bold')

    # Plot 6: ROI (Return on Investment) analysis
    ax6 = fig.add_subplot(gs[1, 2])

    # Calculate ROI as absolute gain / chunks added
    roi = efficiency_df['Absolute Gain (%)'] / efficiency_df['Chunks Added']

    colors_roi = ['#2ecc71' if r == max(roi) else '#3498db' if r > 2 else '#e74c3c' if r == 0 else '#95a5a6'
                  for r in roi]

    bars_roi = ax6.barh(range(len(transitions)), roi, color=colors_roi, alpha=0.8,
                        edgecolor='black', linewidth=1.5)
    ax6.set_yticks(range(len(transitions)))
    ax6.set_yticklabels(transitions, fontsize=10)
    ax6.set_xlabel('ROI (% gain per chunk)', fontsize=12, fontweight='bold')
    ax6.set_title('Return on Investment (ROI)', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars_roi, roi)):
        width = bar.get_width()
        ax6.text(width, bar.get_y() + bar.get_height()/2.,
                f' {val:.2f}%', ha='left', va='center', fontsize=9, fontweight='bold')

    # Plot 7: Marginal utility curve
    ax7 = fig.add_subplot(gs[2, :2])

    k_plot = []
    marginal_gains = []

    for idx, row in efficiency_df.iterrows():
        k_plot.append(row['k_from'])
        marginal_gains.append(row['Gain per Chunk (%)'])

    ax7.plot(k_plot, marginal_gains, marker='s', linewidth=3, markersize=12,
             color='#34495e', markerfacecolor='#e74c3c', markeredgewidth=2,
             markeredgecolor='#c0392b', label='Marginal Efficiency')

    # Add threshold lines
    ax7.axhline(y=4, color='#2ecc71', linestyle=':', linewidth=2, alpha=0.5, label='High Efficiency (>4%)')
    ax7.axhline(y=2, color='#f39c12', linestyle=':', linewidth=2, alpha=0.5, label='Medium Efficiency (>2%)')

    # Shade regions
    ax7.fill_between(k_plot, marginal_gains, 4, where=[m >= 4 for m in marginal_gains],
                     alpha=0.3, color='#2ecc71', label='High Efficiency Zone')
    ax7.fill_between(k_plot, marginal_gains, 0, where=[m < 2 for m in marginal_gains],
                     alpha=0.3, color='#e74c3c', label='Low Efficiency Zone')

    ax7.set_xlabel('Starting k value', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Marginal Efficiency (% per chunk)', fontsize=12, fontweight='bold')
    ax7.set_title('Marginal Utility Curve: Diminishing Returns Analysis', fontsize=14, fontweight='bold')
    ax7.legend(loc='upper right', fontsize=9)
    ax7.grid(True, alpha=0.3)
    ax7.set_xscale('log')
    ax7.set_xticks(k_plot)
    ax7.set_xticklabels(k_plot)

    # Add annotations
    for k, gain in zip(k_plot, marginal_gains):
        ax7.annotate(f'{gain:.2f}%', (k, gain), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=9, fontweight='bold')

    # Plot 8: Recommendations panel
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')

    # Get recommendations
    best_efficiency = efficiency_df.loc[efficiency_df['Gain per Chunk (%)'].idxmax()]
    best_absolute = efficiency_df.loc[efficiency_df['Absolute Gain (%)'].idxmax()]

    recommendations_text = f"""
    OPTIMAL RETRIEVAL STRATEGY
    {'='*40}

    🥇 BEST EFFICIENCY
    {best_efficiency['Transition']}
    • {best_efficiency['Gain per Chunk (%)']:.2f}% per chunk
    • {best_efficiency['Chunks Added']} chunk(s) added
    • Total gain: {best_efficiency['Absolute Gain (%)']:.2f}%

    📊 LARGEST ABSOLUTE GAIN
    {best_absolute['Transition']}
    • Total gain: {best_absolute['Absolute Gain (%)']:.2f}%
    • {best_absolute['Chunks Added']} chunks added
    • Efficiency: {best_absolute['Gain per Chunk (%)']:.2f}%/chunk

    💡 RECOMMENDATIONS
    {'='*40}

    Tight Context (≤5 chunks):
    → Use k=5
    → 38.22% accuracy
    → Best efficiency at k4→k5

    Moderate Context (≤10 chunks):
    → Use k=10
    → 58.61% accuracy
    → Worth 5 extra chunks

    Large Context (>10 chunks):
    → Stop at k=10
    → No improvement beyond k=10
    → Avoid wasting context
    """

    ax8.text(0.05, 0.95, recommendations_text, transform=ax8.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(f"{output_dir}/efficiency_analysis_comprehensive.png", dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved comprehensive plot: {output_dir}/efficiency_analysis_comprehensive.png")

    plt.close()

def main():
    # Configuration
    results_file = "/Users/choemanseung/789/hft/final_retrieval_testing/results/optimized_comprehensive/optimized_summary_20251010_095218.json"
    output_dir = Path("/Users/choemanseung/789/hft/analysis_output")
    output_dir.mkdir(exist_ok=True, parents=True)

    print("🚀 Efficiency-Adjusted Retrieval Analysis")
    print("="*100)
    print(f"📂 Input: {results_file}")
    print(f"📁 Output: {output_dir}")

    # Load data
    df = load_results(results_file)
    print(f"\n✅ Loaded {len(df)} configurations")

    # Calculate efficiency metrics
    efficiency_df, avg_passk = calculate_efficiency_metrics(df)

    # Create tables
    create_efficiency_tables(efficiency_df, output_dir)

    # Create plots
    print("\n" + "="*100)
    print("Creating comprehensive efficiency visualizations...")
    print("="*100)
    plot_efficiency_analysis(efficiency_df, avg_passk, output_dir)

    print("\n" + "="*100)
    print("✅ EFFICIENCY ANALYSIS COMPLETE!")
    print("="*100)
    print(f"\n📁 All outputs saved to: {output_dir}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob("efficiency_*")):
        print(f"  • {file.name}")

if __name__ == "__main__":
    main()
