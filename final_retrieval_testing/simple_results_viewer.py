#!/usr/bin/env python3
"""
Simple Results Viewer for Cases with Low Performance

This script handles results files where most configurations have 0.0 performance
and provides useful analysis even in those cases.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_results(results_file):
    """Analyze results file even with many 0.0 performances"""
    
    print(f"📊 ANALYZING RESULTS: {results_file}")
    print("="*60)
    
    # Load data
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print(f"📈 Total configurations: {len(data)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Separate valid vs invalid results
    valid_results = df[df['pass_at_5'] > 0.0]
    zero_results = df[df['pass_at_5'] == 0.0]
    
    print(f"✅ Configurations with Pass@5 > 0: {len(valid_results)}")
    print(f"❌ Configurations with Pass@5 = 0: {len(zero_results)}")
    print(f"📊 Success Rate: {len(valid_results)/len(df)*100:.1f}%")
    
    # Memory analysis (works even with 0.0 performance)
    print(f"\n💾 MEMORY USAGE ANALYSIS")
    print("-" * 30)
    # Calculate memory delta (additional memory used)
    memory_delta = df['memory_stats'].apply(lambda x: x['peak_memory_mb'] - x['start_memory_mb'])
    print(f"Average Additional Memory Used: {memory_delta.mean():.1f} MB")
    print(f"Additional Memory Range: {memory_delta.min():.1f} - {memory_delta.max():.1f} MB")
    print(f"Average Absolute Peak Memory: {df['memory_stats'].apply(lambda x: x['peak_memory_mb']).mean():.1f} MB")
    
    # Timing analysis
    print(f"\n⏱️  TIMING ANALYSIS")
    print("-" * 20)
    print(f"Average Retrieval Time: {df['avg_retrieval_time'].mean():.4f} seconds")
    print(f"Timing Range: {df['avg_retrieval_time'].min():.4f} - {df['avg_retrieval_time'].max():.4f} seconds")
    
    # Retrieval type breakdown
    print(f"\n🔍 RETRIEVAL TYPE BREAKDOWN")
    print("-" * 30)
    retrieval_breakdown = df.groupby('retrieval_type').agg({
        'pass_at_5': ['count', 'mean', 'max'],
        'avg_retrieval_time': 'mean'
    }).round(4)
    print(retrieval_breakdown)
    
    # Bias configuration breakdown
    print(f"\n⚖️  BIAS CONFIGURATION BREAKDOWN")
    print("-" * 35)
    bias_breakdown = df.groupby('bias_config').agg({
        'pass_at_5': ['count', 'mean', 'max'],
        'avg_retrieval_time': 'mean'
    }).round(4)
    print(bias_breakdown)
    
    # Generate visualizations
    create_basic_plots(df, valid_results, Path(results_file).parent)
    
    # If we have valid results, show top performers
    if len(valid_results) > 0:
        print(f"\n🏆 TOP PERFORMERS")
        print("-" * 20)
        top_5 = valid_results.nlargest(5, 'pass_at_5')
        display_df = top_5[['chunking_method', 'retrieval_type', 'bias_config', 
                           'pass_at_5', 'pass_at_10', 'pass_at_20', 'avg_retrieval_time']].round(3)
        print(display_df.to_string(index=False))
    else:
        print(f"\n⚠️  NO VALID RESULTS FOUND")
        print("All configurations returned 0.0 performance.")
        print("This suggests an issue with the evaluation setup or data matching.")
        
        # Analyze why results might be zero
        print(f"\n🔍 DEBUGGING ZERO RESULTS")
        print("-" * 25)
        
        # Check source distribution
        sample_config = df.iloc[0]
        total_retrieved = sum(sample_config['source_distribution'].values())
        print(f"Total documents retrieved per config: {total_retrieved}")
        print(f"Source distribution: {sample_config['source_distribution']}")
        
        # Check chunking methods
        unique_chunking = df['chunking_method'].unique()
        print(f"Chunking methods tested: {len(unique_chunking)}")
        print(f"Sample chunking methods: {list(unique_chunking)[:5]}")

def create_basic_plots(df, valid_df, output_dir):
    """Create basic visualization plots"""
    
    print(f"\n📈 GENERATING PLOTS")
    print("-" * 20)
    
    # Create output directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    
    # Plot 1: Memory Usage Distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Results Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Memory distribution (additional memory used)
    ax1 = axes[0, 0]
    memory_data = df['memory_stats'].apply(lambda x: x['peak_memory_mb'] - x['start_memory_mb'])
    ax1.hist(memory_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(memory_data.mean(), color='red', linestyle='--', label=f'Mean: {memory_data.mean():.1f} MB')
    ax1.set_title('Additional Memory Usage Distribution')
    ax1.set_xlabel('Additional Memory Used (MB)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Timing distribution
    ax2 = axes[0, 1]
    ax2.hist(df['avg_retrieval_time'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.axvline(df['avg_retrieval_time'].mean(), color='red', linestyle='--', 
               label=f'Mean: {df["avg_retrieval_time"].mean():.4f}s')
    ax2.set_title('Retrieval Time Distribution')
    ax2.set_xlabel('Average Retrieval Time (seconds)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Performance by retrieval type
    ax3 = axes[1, 0]
    perf_by_type = df.groupby('retrieval_type')['pass_at_5'].mean()
    bars = ax3.bar(perf_by_type.index, perf_by_type.values, color=['coral', 'lightblue'])
    ax3.set_title('Average Pass@5 by Retrieval Type')
    ax3.set_ylabel('Pass@5')
    ax3.set_ylim(0, max(0.1, perf_by_type.max() * 1.1))
    
    # Add value labels on bars
    for bar, value in zip(bars, perf_by_type.values):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Performance by bias config
    ax4 = axes[1, 1]
    perf_by_bias = df.groupby('bias_config')['pass_at_5'].mean()
    bars = ax4.bar(perf_by_bias.index, perf_by_bias.values, color=['gold', 'lightcoral'])
    ax4.set_title('Average Pass@5 by Bias Configuration')
    ax4.set_ylabel('Pass@5')
    ax4.set_ylim(0, max(0.1, perf_by_bias.max() * 1.1))
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, perf_by_bias.values):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = plots_dir / "results_analysis_dashboard.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"💾 Dashboard saved to: {plot_file}")
    
    # If we have valid results, create memory vs accuracy plot
    if len(valid_df) > 0:
        plt.figure(figsize=(10, 6))
        
        # Use additional memory (peak - start)
        memory_values = valid_df['memory_stats'].apply(lambda x: x['peak_memory_mb'] - x['start_memory_mb'])
        
        # Color by retrieval type
        colors = {'pure_rag': 'red', 'contextual_rag': 'blue'}
        for ret_type in valid_df['retrieval_type'].unique():
            mask = valid_df['retrieval_type'] == ret_type
            subset = valid_df[mask]
            subset_memory = memory_values[mask]
            
            plt.scatter(subset_memory, subset['pass_at_5'], 
                       c=colors.get(ret_type, 'gray'), label=ret_type, alpha=0.7, s=80)
        
        plt.xlabel('Additional Memory Used (MB)')
        plt.ylabel('Pass@5 Accuracy')
        plt.title('Additional Memory Usage vs Retrieval Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        accuracy_plot = plots_dir / "memory_vs_accuracy.png"
        plt.savefig(accuracy_plot, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Memory vs Accuracy plot saved to: {accuracy_plot}")

def main():
    """Main function"""
    
    # Default results file path
    default_file = "/Users/choemanseung/789/hft/final_retrieval_testing/results/optimized_comprehensive/optimized_summary_20250923_011144.json"
    
    # Check if file exists
    if Path(default_file).exists():
        analyze_results(default_file)
    else:
        print(f"❌ Results file not found: {default_file}")
        print("\nSearching for other results files...")
        
        # Search for other results files
        results_dir = Path("/Users/choemanseung/789/hft/final_retrieval_testing/results")
        if results_dir.exists():
            json_files = list(results_dir.rglob("*.json"))
            if json_files:
                print("📁 Found results files:")
                for i, file in enumerate(json_files, 1):
                    print(f"  {i}. {file}")
                
                # Use the most recent one
                latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
                print(f"\n🔄 Using most recent file: {latest_file}")
                analyze_results(latest_file)
            else:
                print("❌ No JSON results files found in results directory")
        else:
            print("❌ Results directory not found")

if __name__ == "__main__":
    main()