#!/usr/bin/env python3
"""
Visualize retrieval performance test results with charts and detailed analysis.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Any
import re

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ResultsVisualizer:
    """Creates visualizations from retrieval test results"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_df = None
    
    def load_results(self, summary_file: str = None) -> pd.DataFrame:
        """Load results from summary JSON file"""
        if summary_file:
            file_path = Path(summary_file)
        else:
            # Find the most recent summary file
            summary_files = list(self.results_dir.glob("summary_*.json"))
            if not summary_files:
                raise FileNotFoundError(f"No summary files found in {self.results_dir}")
            file_path = max(summary_files, key=lambda x: x.stat().st_mtime)
        
        print(f"Loading results from: {file_path}")
        
        with open(file_path, 'r') as f:
            results_data = json.load(f)
        
        # Convert to DataFrame
        self.results_df = pd.DataFrame(results_data)
        
        # Extract additional features for analysis
        self.results_df['chunk_size'] = self.results_df['chunking_method'].apply(self._extract_chunk_size)
        self.results_df['overlap'] = self.results_df['chunking_method'].apply(self._extract_overlap)
        self.results_df['chunk_type'] = self.results_df['chunking_method'].apply(self._extract_chunk_type)
        
        print(f"Loaded {len(self.results_df)} configurations")
        return self.results_df
    
    def _extract_chunk_size(self, method: str) -> int:
        """Extract chunk size from method name"""
        # Fixed chunking: fixed_c512_o0 -> 512
        # Sentence chunking: sentence_t384_o2 -> 384
        # Paragraph chunking: paragraph_m25 -> 25
        if 'fixed_c' in method:
            match = re.search(r'c(\d+)', method)
            return int(match.group(1)) if match else 0
        elif 'sentence_t' in method:
            match = re.search(r't(\d+)', method)
            return int(match.group(1)) if match else 0
        elif 'paragraph_m' in method:
            match = re.search(r'm(\d+)', method)
            return int(match.group(1)) if match else 0
        elif 'contextual' in method:
            match = re.search(r'c(\d+)', method)
            return int(match.group(1)) if match else 1024
        return 0
    
    def _extract_overlap(self, method: str) -> int:
        """Extract overlap from method name"""
        if '_o' in method:
            match = re.search(r'o(\d+)', method)
            return int(match.group(1)) if match else 0
        return 0
    
    def _extract_chunk_type(self, method: str) -> str:
        """Extract chunk type from method name"""
        if 'fixed' in method:
            return 'fixed'
        elif 'sentence' in method:
            return 'sentence'
        elif 'paragraph' in method:
            return 'paragraph'
        elif 'contextual' in method:
            return 'contextual'
        return 'unknown'
    
    def create_overview_dashboard(self, save_path: str = None):
        """Create comprehensive overview dashboard"""
        if self.results_df is None:
            raise ValueError("No results loaded. Call load_results() first.")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Overall performance comparison (top plot)
        ax1 = plt.subplot(3, 3, (1, 3))
        top_configs = self.results_df.nlargest(10, 'pass_at_10')
        
        x_pos = np.arange(len(top_configs))
        bars = ax1.bar(x_pos, top_configs['pass_at_10'], alpha=0.8, color='skyblue', edgecolor='navy')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Pass@10 (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Top 10 Performing Configurations (Pass@10)', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(top_configs['config_name'], rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Performance by chunking method
        ax2 = plt.subplot(3, 3, 4)
        chunk_performance = self.results_df.groupby('chunk_type')['pass_at_10'].agg(['mean', 'std', 'count'])
        
        bars = ax2.bar(chunk_performance.index, chunk_performance['mean'], 
                      yerr=chunk_performance['std'], capsize=5, alpha=0.8, color='lightcoral')
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_xlabel('Chunking Method', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Average Pass@10 (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Performance by Chunking Method', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Performance by retrieval type
        ax3 = plt.subplot(3, 3, 5)
        retrieval_performance = self.results_df.groupby('retrieval_type')['pass_at_10'].agg(['mean', 'std'])
        
        bars = ax3.bar(retrieval_performance.index, retrieval_performance['mean'], 
                      yerr=retrieval_performance['std'], capsize=5, alpha=0.8, color='lightgreen')
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax3.set_xlabel('Retrieval Type', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Average Pass@10 (%)', fontsize=11, fontweight='bold')
        ax3.set_title('Performance by Retrieval Type', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Performance vs Speed tradeoff
        ax4 = plt.subplot(3, 3, 6)
        scatter = ax4.scatter(self.results_df['avg_retrieval_time_ms'], self.results_df['pass_at_10'], 
                            c=self.results_df['chunk_size'], s=80, alpha=0.7, cmap='viridis')
        
        ax4.set_xlabel('Average Retrieval Time (ms)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Pass@10 (%)', fontsize=11, fontweight='bold')
        ax4.set_title('Performance vs Speed Tradeoff', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Chunk Size', fontsize=10)
        
        # 5. Chunk size impact
        ax5 = plt.subplot(3, 3, 7)
        
        # Group by chunk size and calculate mean performance
        size_bins = [0, 256, 512, 768, 1024, 2000]
        size_labels = ['<256', '256-512', '512-768', '768-1024', '>1024']
        self.results_df['size_bin'] = pd.cut(self.results_df['chunk_size'], bins=size_bins, labels=size_labels)
        size_performance = self.results_df.groupby('size_bin')['pass_at_10'].mean()
        
        bars = ax5.bar(range(len(size_performance)), size_performance.values, alpha=0.8, color='orange')
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if not np.isnan(height):
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax5.set_xlabel('Chunk Size Range', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Average Pass@10 (%)', fontsize=11, fontweight='bold')
        ax5.set_title('Impact of Chunk Size', fontsize=12, fontweight='bold')
        ax5.set_xticks(range(len(size_performance)))
        ax5.set_xticklabels(size_performance.index)
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. Pass@k comparison for top methods
        ax6 = plt.subplot(3, 3, 8)
        top_5_configs = self.results_df.nlargest(5, 'pass_at_10')
        
        k_values = ['pass_at_5', 'pass_at_10', 'pass_at_20']
        x = np.arange(len(k_values))
        width = 0.15
        
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
        
        for i, (_, config) in enumerate(top_5_configs.iterrows()):
            values = [config['pass_at_5'], config['pass_at_10'], config['pass_at_20']]
            ax6.bar(x + i * width, values, width, label=config['config_name'][:15], 
                   alpha=0.8, color=colors[i])
        
        ax6.set_xlabel('Metric', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax6.set_title('Pass@K Comparison (Top 5)', fontsize=12, fontweight='bold')
        ax6.set_xticks(x + width * 2)
        ax6.set_xticklabels(['Pass@5', 'Pass@10', 'Pass@20'])
        ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax6.grid(axis='y', alpha=0.3)
        
        # 7. Performance distribution
        ax7 = plt.subplot(3, 3, 9)
        
        # Box plot of performance by retrieval type
        retrieval_types = self.results_df['retrieval_type'].unique()
        performance_data = [self.results_df[self.results_df['retrieval_type'] == rt]['pass_at_10'].values 
                          for rt in retrieval_types]
        
        box_plot = ax7.boxplot(performance_data, labels=retrieval_types, patch_artist=True)
        
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax7.set_xlabel('Retrieval Type', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Pass@10 (%)', fontsize=11, fontweight='bold')
        ax7.set_title('Performance Distribution', fontsize=12, fontweight='bold')
        ax7.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to: {save_path}")
        else:
            plt.show()
    
    def create_detailed_analysis(self, save_path: str = None):
        """Create detailed analysis plots"""
        if self.results_df is None:
            raise ValueError("No results loaded. Call load_results() first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Correlation heatmap
        ax1 = axes[0, 0]
        numeric_cols = ['pass_at_5', 'pass_at_10', 'pass_at_20', 'avg_retrieval_time_ms', 'chunk_size', 'overlap']
        correlation_matrix = self.results_df[numeric_cols].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=ax1, cbar_kws={'label': 'Correlation'})
        ax1.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
        
        # 2. Chunk size vs performance scatter with trend line
        ax2 = axes[0, 1]
        
        for ret_type in self.results_df['retrieval_type'].unique():
            subset = self.results_df[self.results_df['retrieval_type'] == ret_type]
            ax2.scatter(subset['chunk_size'], subset['pass_at_10'], 
                       label=ret_type, alpha=0.7, s=60)
            
            # Add trend line
            z = np.polyfit(subset['chunk_size'], subset['pass_at_10'], 1)
            p = np.poly1d(z)
            ax2.plot(subset['chunk_size'], p(subset['chunk_size']), '--', alpha=0.8)
        
        ax2.set_xlabel('Chunk Size', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Pass@10 (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Chunk Size vs Performance by Retrieval Type', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Performance improvement from overlap
        ax3 = axes[1, 0]
        
        # Group by base chunking method and compare with/without overlap
        chunk_base_methods = self.results_df['chunking_method'].apply(
            lambda x: re.sub(r'_o\d+', '', x)
        ).unique()
        
        improvements = []
        base_methods = []
        
        for base_method in chunk_base_methods:
            no_overlap = self.results_df[
                (self.results_df['chunking_method'] == f"{base_method}_o0") |
                (self.results_df['chunking_method'] == base_method)
            ]
            with_overlap = self.results_df[
                self.results_df['chunking_method'].str.contains(f"{base_method}_o") &
                ~self.results_df['chunking_method'].str.contains("_o0")
            ]
            
            if len(no_overlap) > 0 and len(with_overlap) > 0:
                no_overlap_perf = no_overlap['pass_at_10'].mean()
                with_overlap_perf = with_overlap['pass_at_10'].mean()
                improvement = with_overlap_perf - no_overlap_perf
                improvements.append(improvement)
                base_methods.append(base_method)
        
        if improvements:
            bars = ax3.bar(range(len(improvements)), improvements, 
                          color=['green' if x > 0 else 'red' for x in improvements], alpha=0.7)
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax3.set_xlabel('Base Chunking Method', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Performance Improvement (%)', fontsize=11, fontweight='bold')
            ax3.set_title('Impact of Chunk Overlap', fontsize=12, fontweight='bold')
            ax3.set_xticks(range(len(base_methods)))
            ax3.set_xticklabels(base_methods, rotation=45)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.grid(axis='y', alpha=0.3)
        
        # 4. Speed vs accuracy quadrant analysis
        ax4 = axes[1, 1]
        
        median_time = self.results_df['avg_retrieval_time_ms'].median()
        median_accuracy = self.results_df['pass_at_10'].median()
        
        # Create quadrant colors
        colors = []
        for _, row in self.results_df.iterrows():
            if row['avg_retrieval_time_ms'] < median_time and row['pass_at_10'] > median_accuracy:
                colors.append('green')  # Fast & Accurate
            elif row['avg_retrieval_time_ms'] < median_time and row['pass_at_10'] < median_accuracy:
                colors.append('orange')  # Fast but Inaccurate
            elif row['avg_retrieval_time_ms'] > median_time and row['pass_at_10'] > median_accuracy:
                colors.append('blue')  # Slow but Accurate
            else:
                colors.append('red')  # Slow & Inaccurate
        
        scatter = ax4.scatter(self.results_df['avg_retrieval_time_ms'], self.results_df['pass_at_10'], 
                            c=colors, alpha=0.7, s=80)
        
        # Add quadrant lines
        ax4.axvline(x=median_time, color='black', linestyle='--', alpha=0.5)
        ax4.axhline(y=median_accuracy, color='black', linestyle='--', alpha=0.5)
        
        # Add quadrant labels
        ax4.text(median_time*0.7, median_accuracy*1.02, 'Fast & Accurate', 
                ha='center', va='bottom', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        ax4.text(median_time*1.3, median_accuracy*1.02, 'Slow but Accurate', 
                ha='center', va='bottom', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        ax4.text(median_time*0.7, median_accuracy*0.98, 'Fast but Inaccurate', 
                ha='center', va='top', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
        ax4.text(median_time*1.3, median_accuracy*0.98, 'Slow & Inaccurate', 
                ha='center', va='top', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        
        ax4.set_xlabel('Average Retrieval Time (ms)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Pass@10 (%)', fontsize=11, fontweight='bold')
        ax4.set_title('Speed vs Accuracy Quadrant Analysis', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Detailed analysis saved to: {save_path}")
        else:
            plt.show()
    
    def generate_recommendations(self) -> Dict[str, Any]:
        """Generate performance recommendations based on results"""
        if self.results_df is None:
            raise ValueError("No results loaded. Call load_results() first.")
        
        recommendations = {}
        
        # Best overall configuration
        best_overall = self.results_df.loc[self.results_df['pass_at_10'].idxmax()]
        recommendations['best_overall'] = {
            'config': best_overall['config_name'],
            'pass_at_10': best_overall['pass_at_10'],
            'retrieval_time': best_overall['avg_retrieval_time_ms']
        }
        
        # Best for production (balance of speed and accuracy)
        # Define production criteria: Pass@10 > 85% and time < 50ms
        production_candidates = self.results_df[
            (self.results_df['pass_at_10'] > 85) & 
            (self.results_df['avg_retrieval_time_ms'] < 50)
        ]
        
        if len(production_candidates) > 0:
            best_production = production_candidates.loc[production_candidates['pass_at_10'].idxmax()]
            recommendations['best_for_production'] = {
                'config': best_production['config_name'],
                'pass_at_10': best_production['pass_at_10'],
                'retrieval_time': best_production['avg_retrieval_time_ms']
            }
        
        # Best chunking method
        chunk_performance = self.results_df.groupby('chunk_type')['pass_at_10'].mean()
        best_chunk_method = chunk_performance.idxmax()
        recommendations['best_chunking_method'] = {
            'method': best_chunk_method,
            'avg_performance': chunk_performance[best_chunk_method]
        }
        
        # Best retrieval type
        retrieval_performance = self.results_df.groupby('retrieval_type')['pass_at_10'].mean()
        best_retrieval_type = retrieval_performance.idxmax()
        recommendations['best_retrieval_type'] = {
            'type': best_retrieval_type,
            'avg_performance': retrieval_performance[best_retrieval_type]
        }
        
        # Optimal chunk size
        size_performance = self.results_df.groupby('chunk_size')['pass_at_10'].mean()
        optimal_size = size_performance.idxmax()
        recommendations['optimal_chunk_size'] = {
            'size': optimal_size,
            'performance': size_performance[optimal_size]
        }
        
        return recommendations
    
    def print_recommendations(self):
        """Print performance recommendations"""
        recommendations = self.generate_recommendations()
        
        print("\n" + "="*60)
        print("PERFORMANCE RECOMMENDATIONS")
        print("="*60)
        
        print(f"\n🏆 BEST OVERALL CONFIGURATION:")
        print(f"   Config: {recommendations['best_overall']['config']}")
        print(f"   Pass@10: {recommendations['best_overall']['pass_at_10']:.1f}%")
        print(f"   Avg Time: {recommendations['best_overall']['retrieval_time']:.1f}ms")
        
        if 'best_for_production' in recommendations:
            print(f"\n🚀 BEST FOR PRODUCTION:")
            print(f"   Config: {recommendations['best_for_production']['config']}")
            print(f"   Pass@10: {recommendations['best_for_production']['pass_at_10']:.1f}%")
            print(f"   Avg Time: {recommendations['best_for_production']['retrieval_time']:.1f}ms")
            print(f"   (Balanced speed & accuracy)")
        
        print(f"\n📊 BEST CHUNKING METHOD:")
        print(f"   Method: {recommendations['best_chunking_method']['method']}")
        print(f"   Avg Performance: {recommendations['best_chunking_method']['avg_performance']:.1f}%")
        
        print(f"\n🔍 BEST RETRIEVAL TYPE:")
        print(f"   Type: {recommendations['best_retrieval_type']['type']}")
        print(f"   Avg Performance: {recommendations['best_retrieval_type']['avg_performance']:.1f}%")
        
        print(f"\n📏 OPTIMAL CHUNK SIZE:")
        print(f"   Size: {recommendations['optimal_chunk_size']['size']}")
        print(f"   Performance: {recommendations['optimal_chunk_size']['performance']:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Visualize retrieval performance test results')
    parser.add_argument('--results-dir', default='results', help='Directory containing results')
    parser.add_argument('--summary-file', help='Specific summary file to visualize')
    parser.add_argument('--output-dir', default='visualizations', help='Directory to save plots')
    parser.add_argument('--show-plots', action='store_true', help='Display plots instead of saving')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize visualizer
    visualizer = ResultsVisualizer(args.results_dir)
    
    try:
        # Load results
        visualizer.load_results(args.summary_file)
        
        # Generate visualizations
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        if args.show_plots:
            print("Creating overview dashboard...")
            visualizer.create_overview_dashboard()
            
            print("Creating detailed analysis...")
            visualizer.create_detailed_analysis()
        else:
            print("Creating overview dashboard...")
            dashboard_path = output_dir / f"overview_dashboard_{timestamp}.png"
            visualizer.create_overview_dashboard(str(dashboard_path))
            
            print("Creating detailed analysis...")
            analysis_path = output_dir / f"detailed_analysis_{timestamp}.png"
            visualizer.create_detailed_analysis(str(analysis_path))
        
        # Print recommendations
        visualizer.print_recommendations()
        
        # Save recommendations to file
        recommendations = visualizer.generate_recommendations()
        rec_path = output_dir / f"recommendations_{timestamp}.json"
        with open(rec_path, 'w') as f:
            json.dump(recommendations, f, indent=2)
        print(f"\nRecommendations saved to: {rec_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())