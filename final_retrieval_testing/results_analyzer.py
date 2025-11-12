#!/usr/bin/env python3
"""
Results Analyzer for Hybrid Retrieval Evaluation

This script analyzes the results from optimized_hybrid_evaluator.py and generates:
1. Top performers comparison tables for Pass@5, Pass@10, Pass@20
2. Memory usage vs retrieval accuracy scatter plots
3. Performance breakdown by retrieval type and bias configuration
4. Chunking method analysis
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ResultsAnalyzer:
    """Analyze and visualize hybrid retrieval evaluation results"""
    
    def __init__(self, results_file: str, output_dir: str = "analysis_output"):
        self.results_file = Path(results_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load and process data
        self.data = self.load_results()
        self.df = self.create_dataframe()
        
        print(f"📊 Loaded {len(self.df)} configurations from {results_file}")
        print(f"📁 Output directory: {self.output_dir}")
    
    def load_results(self):
        """Load results from JSON file"""
        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")
        
        with open(self.results_file, 'r') as f:
            data = json.load(f)
        
        return data
    
    def create_dataframe(self):
        """Convert JSON data to pandas DataFrame for easier analysis"""
        rows = []

        for item in self.data:
            row = {
                'config_name': item['config_name'],
                'chunking_method': item['chunking_method'],
                'retrieval_type': item['retrieval_type'],
                'bias_config': item['bias_config'],
                'pass_at_1': item.get('pass_at_1', 0.0),
                'pass_at_2': item.get('pass_at_2', 0.0),
                'pass_at_3': item.get('pass_at_3', 0.0),
                'pass_at_4': item.get('pass_at_4', 0.0),
                'pass_at_5': item['pass_at_5'],
                'pass_at_10': item['pass_at_10'],
                'pass_at_20': item['pass_at_20'],
                'avg_retrieval_time': item['avg_retrieval_time'],
                'total_test_cases': item['total_test_cases'],
                'peak_memory_mb': item['memory_stats']['peak_memory_mb'] - item['memory_stats']['start_memory_mb'],
                'absolute_peak_mb': item['memory_stats']['peak_memory_mb'],
                'total_retrieved': sum(item['source_distribution'].values()),
                'healthify_pct': item['source_distribution']['healthify'] / sum(item['source_distribution'].values()) * 100,
                'mayo_pct': item['source_distribution']['mayo'] / sum(item['source_distribution'].values()) * 100,
                'nhs_pct': item['source_distribution']['nhs'] / sum(item['source_distribution'].values()) * 100
            }
            rows.append(row)

        return pd.DataFrame(rows)
    
    def generate_top_performers_table(self, top_n: int = 10):
        """Generate tables for top performers across different metrics"""
        
        print(f"\n🏆 GENERATING TOP {top_n} PERFORMERS TABLES")
        print("="*60)
        
        # Filter out configurations with 0.0 performance
        valid_df = self.df[self.df['pass_at_5'] > 0.0].copy()
        
        if len(valid_df) == 0:
            print("⚠️  Warning: No configurations with pass_at_5 > 0.0 found!")
            valid_df = self.df.copy()
        
        # Top performers for each metric
        metrics = {
            'Pass@1': ('pass_at_1', 'Top Pass@1 Performers'),
            'Pass@2': ('pass_at_2', 'Top Pass@2 Performers'),
            'Pass@3': ('pass_at_3', 'Top Pass@3 Performers'),
            'Pass@4': ('pass_at_4', 'Top Pass@4 Performers'),
            'Pass@5': ('pass_at_5', 'Top Pass@5 Performers'),
            'Pass@10': ('pass_at_10', 'Top Pass@10 Performers'),
            'Pass@20': ('pass_at_20', 'Top Pass@20 Performers')
        }
        
        tables = {}
        
        for metric_name, (column, title) in metrics.items():
            # Sort by metric and get top performers
            top_performers = valid_df.nlargest(top_n, column)
            
            # Create formatted table
            table_data = top_performers[[
                'chunking_method', 'retrieval_type', 'bias_config',
                'pass_at_1', 'pass_at_2', 'pass_at_3', 'pass_at_4',
                'pass_at_5', 'pass_at_10', 'pass_at_20',
                'avg_retrieval_time', 'peak_memory_mb'
            ]].copy()

            # Format for better display
            table_data['pass_at_1'] = table_data['pass_at_1'].round(3)
            table_data['pass_at_2'] = table_data['pass_at_2'].round(3)
            table_data['pass_at_3'] = table_data['pass_at_3'].round(3)
            table_data['pass_at_4'] = table_data['pass_at_4'].round(3)
            table_data['pass_at_5'] = table_data['pass_at_5'].round(3)
            table_data['pass_at_10'] = table_data['pass_at_10'].round(3)
            table_data['pass_at_20'] = table_data['pass_at_20'].round(3)
            table_data['avg_retrieval_time'] = table_data['avg_retrieval_time'].round(4)
            table_data['peak_memory_mb'] = table_data['peak_memory_mb'].round(1)
            
            tables[metric_name] = table_data
            
            # Print to console
            print(f"\n📋 {title}")
            print("-" * len(title))
            print(table_data.to_string(index=False))
            
            # Save to CSV
            csv_file = self.output_dir / f"top_{metric_name.lower().replace('@', '_')}_performers.csv"
            table_data.to_csv(csv_file, index=False)
            print(f"💾 Saved to: {csv_file}")
        
        # Combined top performers summary
        self.generate_combined_summary(valid_df, top_n)
        
        return tables
    
    def generate_combined_summary(self, df, top_n):
        """Generate a combined summary of top performers"""
        
        # Get top 5 for each metric
        top_5_configs = set()
        for metric in ['pass_at_1', 'pass_at_2', 'pass_at_3', 'pass_at_4', 'pass_at_5', 'pass_at_10', 'pass_at_20']:
            top_configs = df.nlargest(5, metric)['config_name'].tolist()
            top_5_configs.update(top_configs)
        
        # Create summary for these top configs
        summary_df = df[df['config_name'].isin(top_5_configs)].copy()
        summary_df = summary_df.sort_values('pass_at_5', ascending=False)
        
        print(f"\n🌟 COMBINED TOP PERFORMERS SUMMARY")
        print("="*50)
        print("Configurations that appear in top 5 for any metric:")
        
        summary_table = summary_df[[
            'chunking_method', 'retrieval_type', 'bias_config',
            'pass_at_1', 'pass_at_2', 'pass_at_3', 'pass_at_4',
            'pass_at_5', 'pass_at_10', 'pass_at_20',
            'peak_memory_mb'
        ]].copy()

        summary_table['pass_at_1'] = summary_table['pass_at_1'].round(3)
        summary_table['pass_at_2'] = summary_table['pass_at_2'].round(3)
        summary_table['pass_at_3'] = summary_table['pass_at_3'].round(3)
        summary_table['pass_at_4'] = summary_table['pass_at_4'].round(3)
        summary_table['pass_at_5'] = summary_table['pass_at_5'].round(3)
        summary_table['pass_at_10'] = summary_table['pass_at_10'].round(3)
        summary_table['pass_at_20'] = summary_table['pass_at_20'].round(3)
        summary_table['peak_memory_mb'] = summary_table['peak_memory_mb'].round(1)
        
        print(summary_table.to_string(index=False))
        
        # Save combined summary
        csv_file = self.output_dir / "combined_top_performers.csv"
        summary_table.to_csv(csv_file, index=False)
        print(f"💾 Combined summary saved to: {csv_file}")
    
    def create_memory_vs_accuracy_plots(self):
        """Create scatter plots showing memory usage vs retrieval accuracy"""
        
        print(f"\n📈 GENERATING MEMORY VS ACCURACY PLOTS")
        print("="*50)
        
        # Filter valid data
        valid_df = self.df[self.df['pass_at_5'] > 0.0].copy()
        
        if len(valid_df) == 0:
            print("⚠️  Warning: No valid data for plotting!")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Memory Usage vs Retrieval Accuracy Analysis', fontsize=16, fontweight='bold')
        
        # Color mapping for retrieval types
        colors = {'pure_rag': '#FF6B6B', 'contextual_rag': '#4ECDC4'}
        
        # Plot 1: Memory vs Pass@5
        ax1 = axes[0, 0]
        for ret_type in valid_df['retrieval_type'].unique():
            data = valid_df[valid_df['retrieval_type'] == ret_type]
            ax1.scatter(data['peak_memory_mb'], data['pass_at_5'], 
                       c=colors[ret_type], label=ret_type, alpha=0.7, s=60)
        
        ax1.set_xlabel('Peak Memory (MB)')
        ax1.set_ylabel('Pass@5')
        ax1.set_title('Memory Usage vs Pass@5')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Memory vs Pass@10
        ax2 = axes[0, 1]
        for ret_type in valid_df['retrieval_type'].unique():
            data = valid_df[valid_df['retrieval_type'] == ret_type]
            ax2.scatter(data['peak_memory_mb'], data['pass_at_10'], 
                       c=colors[ret_type], label=ret_type, alpha=0.7, s=60)
        
        ax2.set_xlabel('Peak Memory (MB)')
        ax2.set_ylabel('Pass@10')
        ax2.set_title('Memory Usage vs Pass@10')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Memory vs Pass@20
        ax3 = axes[1, 0]
        for ret_type in valid_df['retrieval_type'].unique():
            data = valid_df[valid_df['retrieval_type'] == ret_type]
            ax3.scatter(data['peak_memory_mb'], data['pass_at_20'], 
                       c=colors[ret_type], label=ret_type, alpha=0.7, s=60)
        
        ax3.set_xlabel('Peak Memory (MB)')
        ax3.set_ylabel('Pass@20')
        ax3.set_title('Memory Usage vs Pass@20')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Efficiency (Pass@5 per MB)
        ax4 = axes[1, 1]
        valid_df['efficiency'] = valid_df['pass_at_5'] / valid_df['peak_memory_mb'] * 1000  # per GB
        
        for ret_type in valid_df['retrieval_type'].unique():
            data = valid_df[valid_df['retrieval_type'] == ret_type]
            ax4.scatter(data['peak_memory_mb'], data['efficiency'], 
                       c=colors[ret_type], label=ret_type, alpha=0.7, s=60)
        
        ax4.set_xlabel('Peak Memory (MB)')
        ax4.set_ylabel('Efficiency (Pass@5 per GB)')
        ax4.set_title('Memory Efficiency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "memory_vs_accuracy_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Memory vs Accuracy plots saved to: {plot_file}")
    
    def create_performance_breakdown_plots(self):
        """Create breakdown plots by retrieval type and bias configuration"""
        
        print(f"\n📊 GENERATING PERFORMANCE BREAKDOWN PLOTS")
        print("="*50)
        
        valid_df = self.df[self.df['pass_at_5'] > 0.0].copy()
        
        if len(valid_df) == 0:
            print("⚠️  Warning: No valid data for plotting!")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Breakdown Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Performance by Retrieval Type
        ax1 = axes[0, 0]
        retrieval_performance = valid_df.groupby('retrieval_type')[['pass_at_5', 'pass_at_10', 'pass_at_20']].mean()
        retrieval_performance.plot(kind='bar', ax=ax1, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('Average Performance by Retrieval Type')
        ax1.set_ylabel('Pass Rate')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Performance by Bias Configuration
        ax2 = axes[0, 1]
        bias_performance = valid_df.groupby('bias_config')[['pass_at_5', 'pass_at_10', 'pass_at_20']].mean()
        bias_performance.plot(kind='bar', ax=ax2, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax2.set_title('Average Performance by Bias Configuration')
        ax2.set_ylabel('Pass Rate')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Top Chunking Methods
        ax3 = axes[1, 0]
        chunking_performance = valid_df.groupby('chunking_method')['pass_at_5'].mean().sort_values(ascending=False).head(10)
        chunking_performance.plot(kind='barh', ax=ax3, color='#96CEB4')
        ax3.set_title('Top 10 Chunking Methods (Pass@5)')
        ax3.set_xlabel('Pass@5')
        
        # Plot 4: Memory Usage Distribution
        ax4 = axes[1, 1]
        ax4.hist(valid_df['peak_memory_mb'], bins=20, alpha=0.7, color='#FFEAA7', edgecolor='black')
        ax4.axvline(valid_df['peak_memory_mb'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {valid_df["peak_memory_mb"].mean():.1f} MB')
        ax4.set_title('Memory Usage Distribution')
        ax4.set_xlabel('Peak Memory (MB)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "performance_breakdown_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Performance breakdown plots saved to: {plot_file}")
    
    def generate_statistical_summary(self):
        """Generate comprehensive statistical summary"""
        
        print(f"\n📈 STATISTICAL SUMMARY")
        print("="*40)
        
        valid_df = self.df[self.df['pass_at_5'] > 0.0]
        
        # Basic statistics
        print(f"Total configurations: {len(self.df)}")
        print(f"Valid configurations (Pass@5 > 0): {len(valid_df)}")
        print(f"Success rate: {len(valid_df)/len(self.df)*100:.1f}%")
        
        if len(valid_df) == 0:
            print("⚠️  No valid configurations found!")
            return
        
        # Performance statistics
        print(f"\n📊 Performance Statistics:")
        for metric in ['pass_at_1', 'pass_at_2', 'pass_at_3', 'pass_at_4', 'pass_at_5', 'pass_at_10', 'pass_at_20']:
            print(f"{metric}: Mean={valid_df[metric].mean():.3f}, Std={valid_df[metric].std():.3f}, Max={valid_df[metric].max():.3f}")
        
        # Memory statistics
        print(f"\n💾 Memory Statistics:")
        print(f"Peak Memory: Mean={valid_df['peak_memory_mb'].mean():.1f} MB, Std={valid_df['peak_memory_mb'].std():.1f} MB")
        print(f"Memory Range: {valid_df['peak_memory_mb'].min():.1f} - {valid_df['peak_memory_mb'].max():.1f} MB")
        
        # Best overall configuration
        best_config = valid_df.loc[valid_df['pass_at_5'].idxmax()]
        print(f"\n🏆 Best Overall Configuration (Pass@5):")
        print(f"Name: {best_config['config_name']}")
        print(f"Pass@5: {best_config['pass_at_5']:.3f}")
        print(f"Pass@10: {best_config['pass_at_10']:.3f}")
        print(f"Pass@20: {best_config['pass_at_20']:.3f}")
        print(f"Memory: {best_config['peak_memory_mb']:.1f} MB")
    
    def run_full_analysis(self, top_n: int = 10):
        """Run complete analysis pipeline"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"🚀 HYBRID RETRIEVAL RESULTS ANALYSIS")
        print(f"📅 Analysis started: {timestamp}")
        print(f"📄 Input file: {self.results_file}")
        print("="*60)
        
        # Generate all analyses
        self.generate_statistical_summary()
        self.generate_top_performers_table(top_n)
        self.create_memory_vs_accuracy_plots()
        self.create_performance_breakdown_plots()
        
        print(f"\n✅ Analysis complete! Check {self.output_dir} for all outputs.")
        print(f"📁 Generated files:")
        for file in self.output_dir.glob("*"):
            print(f"  - {file.name}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Analyze hybrid retrieval evaluation results")
    parser.add_argument("--results-file", 
                       default="/Users/choemanseung/789/hft/final_retrieval_testing/results/optimized_comprehensive/optimized_summary_20251010_095218.json",
                       help="Path to results JSON file")
    parser.add_argument("--output-dir", default="analysis_output", 
                       help="Output directory for analysis results")
    parser.add_argument("--top-n", type=int, default=10, 
                       help="Number of top performers to show")
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    try:
        analyzer = ResultsAnalyzer(args.results_file, args.output_dir)
        analyzer.run_full_analysis(args.top_n)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Available results files:")
        results_dir = Path("results")
        if results_dir.exists():
            for file in results_dir.rglob("*.json"):
                print(f"  - {file}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()