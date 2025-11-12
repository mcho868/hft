#!/usr/bin/env python3
"""
LLM-as-Judge Results Visualization Generator for Medical Triage Testing

Creates comprehensive visualizations for LLM-as-judge quality metrics including:
- Reasoning quality scores (0-100)
- Next step quality scores (0-100)
- Overall quality scores (0-100)
- Quality vs accuracy correlations
- Configuration comparisons

Based on top5_final_test_results_20251002_022750.json
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Optional
import argparse
from datetime import datetime

class LLMJudgePlotsGenerator: # LLMJudgePlotsGenerator was used in the original code
    """Generate high-quality plots for LLM-as-judge evaluation analysis"""
    
    def __init__(self, results_file: str):
        self.results_file = Path(results_file)
        self.results_data = self._load_results()
        self.df = self._create_dataframe()
        
        # Set style for publication-quality plots
        plt.style.use('default')
        sns.set_palette("Set2")
        plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
        
        # Create output directory
        self.output_dir = Path("llm_judge_plots")
        self.output_dir.mkdir(exist_ok=True)
        
    def _load_results(self) -> List[Dict[str, Any]]:
        """Load results from LLM-as-judge evaluation file"""
        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")
        
        print(f"📊 Loading LLM-as-judge results from: {self.results_file}")
        
        with open(self.results_file, 'r') as f:
            results = json.load(f)
        
        # Handle both direct list format and wrapped format
        if isinstance(results, dict) and 'results' in results:
            results = results['results']
        
        print(f"✅ Loaded {len(results)} configurations with LLM judge metrics")
        return results
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis"""
        data = []
        
        for result in self.results_data:
            config = result['config']
            
            # Parse configuration details
            model_name = config['model_name']
            has_rag = config['rag_config'] is not None
            rag_name = config['rag_config']['name'] if has_rag else 'NoRAG'
            
            # Extract adapter type from path
            adapter_path = config.get('adapter_path', '')
            if adapter_path:
                adapter_parts = adapter_path.split('_')
                adapter_type = adapter_parts[-2] if len(adapter_parts) > 2 else 'unknown'
            else:
                adapter_type = 'none'
            
            # Extract LLM judge metrics
            llm_metrics = result.get('llm_judge_metrics', {})
            
            # Calculate extraction success rate correctly (fix the bug)
            cases_evaluated = result.get('cases_evaluated', 1975)
            unknown_count = result.get('unknown_triage_count', 0)
            extraction_success_rate = 1 - (unknown_count / cases_evaluated)
            
            # Extract class-specific recalls from classification report
            class_report = result.get('classification_report', {})
            ed_recall = class_report.get('ED', {}).get('recall', 0)
            gp_recall = class_report.get('GP', {}).get('recall', 0)
            home_recall = class_report.get('HOME', {}).get('recall', 0)
            
            row = {
                # Configuration info
                'model_name': model_name,
                'adapter_type': adapter_type,
                'has_rag': has_rag,
                'rag_config': rag_name,
                'test_name': config['test_name'],
                'short_name': self._create_short_name(config['test_name']),
                
                # Traditional metrics
                'triage_accuracy': result['triage_accuracy'],
                'f1_score': result['f1_score'],
                'f2_score': result['f2_score'],
                'unknown_triage_count': unknown_count,
                'cases_evaluated': cases_evaluated,
                'extraction_success_rate': extraction_success_rate,
                
                # Class-specific recalls
                'ed_recall': ed_recall,
                'gp_recall': gp_recall,
                'home_recall': home_recall,
                
                # LLM-as-judge metrics
                'avg_next_step_quality': llm_metrics.get('avg_next_step_quality', 0),
                'avg_reasoning_quality': llm_metrics.get('avg_reasoning_quality', 0),
                'avg_overall_quality': llm_metrics.get('avg_overall_quality', 0),
                'std_overall_quality': llm_metrics.get('std_overall_quality', 0),
                'min_overall_quality': llm_metrics.get('min_overall_quality', 0),
                'max_overall_quality': llm_metrics.get('max_overall_quality', 0),
                'cases_with_llm_evaluation': llm_metrics.get('cases_with_llm_evaluation', 0),
                
                # Timing
                'avg_inference_time': result['avg_inference_time_per_case'],
                'rag_time': result.get('rag_retrieval_time', 0.0) or 0.0,
            }
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Calculate composite scores
        df['quality_accuracy_ratio'] = df['avg_overall_quality'] / (df['triage_accuracy'] * 100)
        df['quality_gap'] = (df['triage_accuracy'] * 100) - df['avg_overall_quality']
        df['reliability_score'] = df['triage_accuracy'] * 0.7 + df['extraction_success_rate'] * 0.3
        
        print(f"📈 Created DataFrame with {len(df)} configurations")
        print(f"🧑‍⚕️ Average LLM Overall Quality: {df['avg_overall_quality'].mean():.1f}/100")
        print(f"🎯 Accuracy range: {df['triage_accuracy'].min():.3f} - {df['triage_accuracy'].max():.3f}")
        
        return df
    
    def _create_short_name(self, test_name: str) -> str:
        """Create shortened version of test name for plots"""
        short = test_name.replace('SmolLM2-135M_4bit_FineTuned_adapter_safe_triage_SmolLM2-135M_4bit_', '')
        short = short.replace('_NoRAG', ' (No RAG)')
        short = short.replace('_RAG_top1_structured_contextual_diverse', ' (RAG Top1)')
        short = short.replace('_RAG_top2_structured_pure_diverse', ' (RAG Top2)')
        short = short.replace('high_capacity_safe', 'High Capacity')
        short = short.replace('balanced_safe', 'Balanced')
        short = short.replace('performance_safe', 'Performance')
        return short
    
    def generate_all_plots(self):
        """Generate all LLM-as-judge visualization plots"""
        
        print("🎨 Creating LLM-as-judge visualization plots...")
        
        plot_functions = [
            ('01_quality_scores_overview', self.plot_quality_scores_overview),
            ('02_quality_vs_accuracy_correlation', self.plot_quality_vs_accuracy_correlation),
            ('03_quality_components_breakdown', self.plot_quality_components_breakdown),
            ('04_rag_quality_impact', self.plot_rag_quality_impact),
            ('05_adapter_quality_comparison', self.plot_adapter_quality_comparison),
            ('06_quality_distribution_violin', self.plot_quality_distribution_violin),
            ('07_quality_accuracy_gap_analysis', self.plot_quality_accuracy_gap_analysis),
            ('08_configuration_quality_ranking', self.plot_configuration_quality_ranking),
            ('09_quality_variance_analysis', self.plot_quality_variance_analysis),
            ('10_quality_score_heatmap', self.plot_quality_score_heatmap),
            ('11_accuracy_vs_recall_tradeoffs', self.plot_accuracy_vs_recall_tradeoffs),
            ('12_latency_vs_accuracy_scatter', self.plot_latency_vs_accuracy_scatter),
        ]
        
        for plot_name, plot_function in plot_functions:
            try:
                print(f"  📊 Creating {plot_name}...")
                plot_function(plot_name)
            except Exception as e:
                print(f"  ❌ Error creating {plot_name}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"✅ All LLM-as-judge plots saved to: {self.output_dir}")
    
    def plot_quality_scores_overview(self, filename: str):
        """Overview of all quality scores by configuration"""
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(self.df))
        width = 0.25
        
        bars1 = ax.bar(x - width, self.df['avg_next_step_quality'], width, 
                      label='Next Step Quality', alpha=0.8, color='lightcoral')
        bars2 = ax.bar(x, self.df['avg_reasoning_quality'], width, 
                      label='Reasoning Quality', alpha=0.8, color='lightblue')
        bars3 = ax.bar(x + width, self.df['avg_overall_quality'], width, 
                      label='Overall Quality', alpha=0.8, color='lightgreen')
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel('LLM Quality Score (0-100)')
        ax.set_title('LLM-as-Judge Quality Scores by Configuration\n(llama3-3-70b Evaluation)')
        ax.set_xticks(x)
        ax.set_xticklabels(self.df['short_name'], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_quality_vs_accuracy_correlation(self, filename: str):
        """Correlation between LLM quality scores and traditional accuracy"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Overall Quality vs Accuracy
        correlation = self.df['avg_overall_quality'].corr(self.df['triage_accuracy'])
        ax1.scatter(self.df['triage_accuracy'] * 100, self.df['avg_overall_quality'], 
                   s=80, alpha=0.7, c=self.df['has_rag'].map({True: 'red', False: 'blue'}),
                   edgecolors='black', linewidth=0.5)
        
        # Add trend line
        z = np.polyfit(self.df['triage_accuracy'] * 100, self.df['avg_overall_quality'], 1)
        p = np.poly1d(z)
        ax1.plot(self.df['triage_accuracy'] * 100, p(self.df['triage_accuracy'] * 100), 
                "r--", alpha=0.8, linewidth=2)
        
        ax1.set_xlabel('Traditional Accuracy (%)')
        ax1.set_ylabel('LLM Overall Quality Score')
        ax1.set_title(f'Overall Quality vs Accuracy\n(Correlation: {correlation:.3f})')
        ax1.grid(True, alpha=0.3)
        
        # Add legend for RAG vs No RAG
        blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                               markersize=8, label='No RAG')
        red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                              markersize=8, label='With RAG')
        ax1.legend(handles=[blue_patch, red_patch])
        
        # Next Step Quality vs Accuracy
        correlation2 = self.df['avg_next_step_quality'].corr(self.df['triage_accuracy'])
        ax2.scatter(self.df['triage_accuracy'] * 100, self.df['avg_next_step_quality'], 
                   s=80, alpha=0.7, c=self.df['has_rag'].map({True: 'red', False: 'blue'}),
                   edgecolors='black', linewidth=0.5)
        
        z2 = np.polyfit(self.df['triage_accuracy'] * 100, self.df['avg_next_step_quality'], 1)
        p2 = np.poly1d(z2)
        ax2.plot(self.df['triage_accuracy'] * 100, p2(self.df['triage_accuracy'] * 100), 
                "r--", alpha=0.8, linewidth=2)
        
        ax2.set_xlabel('Traditional Accuracy (%)')
        ax2.set_ylabel('Next Step Quality Score')
        ax2.set_title(f'Next Step Quality vs Accuracy\n(Correlation: {correlation2:.3f})')
        ax2.grid(True, alpha=0.3)
        
        # Reasoning Quality vs Accuracy
        correlation3 = self.df['avg_reasoning_quality'].corr(self.df['triage_accuracy'])
        ax3.scatter(self.df['triage_accuracy'] * 100, self.df['avg_reasoning_quality'], 
                   s=80, alpha=0.7, c=self.df['has_rag'].map({True: 'red', False: 'blue'}),
                   edgecolors='black', linewidth=0.5)
        
        z3 = np.polyfit(self.df['triage_accuracy'] * 100, self.df['avg_reasoning_quality'], 1)
        p3 = np.poly1d(z3)
        ax3.plot(self.df['triage_accuracy'] * 100, p3(self.df['triage_accuracy'] * 100), 
                "r--", alpha=0.8, linewidth=2)
        
        ax3.set_xlabel('Traditional Accuracy (%)')
        ax3.set_ylabel('Reasoning Quality Score')
        ax3.set_title(f'Reasoning Quality vs Accuracy\n(Correlation: {correlation3:.3f})')
        ax3.grid(True, alpha=0.3)
        
        # Quality Gap Analysis
        ax4.scatter(self.df['triage_accuracy'] * 100, self.df['quality_gap'], 
                   s=80, alpha=0.7, c=self.df['has_rag'].map({True: 'red', False: 'blue'}),
                   edgecolors='black', linewidth=0.5)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.set_xlabel('Traditional Accuracy (%)')
        ax4.set_ylabel('Quality Gap (Accuracy% - LLM Quality)')
        ax4.set_title('Quality-Accuracy Gap Analysis\n(Positive = Accuracy Higher than LLM Assessment)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_quality_components_breakdown(self, filename: str):
        """Detailed breakdown of quality components"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Radar chart for quality components
        categories = ['Next Step\nQuality', 'Reasoning\nQuality', 'Overall\nQuality', 'Traditional\nAccuracy']
        N = len(categories)
        
        # Prepare data for radar chart
        for i, (_, row) in enumerate(self.df.iterrows()):
            values = [
                row['avg_next_step_quality'],
                row['avg_reasoning_quality'], 
                row['avg_overall_quality'],
                row['triage_accuracy'] * 100  # Convert to 0-100 scale
            ]
            values += values[:1]  # Complete the circle
            
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            
            color = plt.cm.Set3(i / len(self.df))
            ax1.plot(angles, values, 'o-', linewidth=1, label=row['short_name'], 
                    color=color, alpha=0.7)
            ax1.fill(angles, values, alpha=0.1, color=color)
        
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories)
        ax1.set_ylim(0, 100)
        ax1.set_title('Quality Components Radar Chart\n(0-100 Scale)')
        ax1.legend(bbox_to_anchor=(1.3, 1), loc='upper left')
        ax1.grid(True)
        
        # Quality component correlation matrix
        quality_cols = ['avg_next_step_quality', 'avg_reasoning_quality', 'avg_overall_quality', 'triage_accuracy']
        corr_matrix = self.df[quality_cols].corr()
        
        # Scale accuracy to 0-100 for fair comparison
        corr_data = self.df[quality_cols].copy()
        corr_data['triage_accuracy'] = corr_data['triage_accuracy'] * 100
        corr_matrix = corr_data.corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                   ax=ax2, square=True, cbar_kws={'label': 'Correlation Coefficient'})
        ax2.set_title('Quality Metrics Correlation Matrix')
        ax2.set_xticklabels(['Next Step', 'Reasoning', 'Overall', 'Accuracy'], rotation=45)
        ax2.set_yticklabels(['Next Step', 'Reasoning', 'Overall', 'Accuracy'], rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_rag_quality_impact(self, filename: str):
        """Impact of RAG on LLM quality assessment"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Quality scores by RAG status
        rag_comparison = self.df.groupby('has_rag').agg({
            'avg_next_step_quality': ['mean', 'std'],
            'avg_reasoning_quality': ['mean', 'std'],
            'avg_overall_quality': ['mean', 'std'],
            'triage_accuracy': 'mean'
        }).round(2)
        
        # Flatten column names
        rag_comparison.columns = ['next_mean', 'next_std', 'reasoning_mean', 'reasoning_std', 
                                 'overall_mean', 'overall_std', 'accuracy_mean']
        rag_comparison.reset_index(inplace=True)
        
        x = [0, 1]
        width = 0.2
        
        bars1 = ax1.bar([x[0] - width, x[1] - width], 
                       [rag_comparison.iloc[0]['next_mean'], rag_comparison.iloc[1]['next_mean']], 
                       width, yerr=[rag_comparison.iloc[0]['next_std'], rag_comparison.iloc[1]['next_std']],
                       label='Next Step Quality', alpha=0.8, capsize=5, color='lightcoral')
        bars2 = ax1.bar([x[0], x[1]], 
                       [rag_comparison.iloc[0]['reasoning_mean'], rag_comparison.iloc[1]['reasoning_mean']], 
                       width, yerr=[rag_comparison.iloc[0]['reasoning_std'], rag_comparison.iloc[1]['reasoning_std']],
                       label='Reasoning Quality', alpha=0.8, capsize=5, color='lightblue')
        bars3 = ax1.bar([x[0] + width, x[1] + width], 
                       [rag_comparison.iloc[0]['overall_mean'], rag_comparison.iloc[1]['overall_mean']], 
                       width, yerr=[rag_comparison.iloc[0]['overall_std'], rag_comparison.iloc[1]['overall_std']],
                       label='Overall Quality', alpha=0.8, capsize=5, color='lightgreen')
        
        ax1.set_xlabel('RAG Configuration')
        ax1.set_ylabel('LLM Quality Score')
        ax1.set_title('RAG Impact on LLM Quality Assessment')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['No RAG', 'With RAG'])
        ax1.legend()
        ax1.set_ylim(0, 80)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Individual configuration comparison
        no_rag_data = self.df[self.df['has_rag'] == False]
        rag_data = self.df[self.df['has_rag'] == True]
        
        ax2.scatter(no_rag_data['triage_accuracy'] * 100, no_rag_data['avg_overall_quality'], 
                   s=100, alpha=0.7, color='blue', label='No RAG', edgecolors='black')
        ax2.scatter(rag_data['triage_accuracy'] * 100, rag_data['avg_overall_quality'], 
                   s=100, alpha=0.7, color='red', label='With RAG', edgecolors='black')
        
        ax2.set_xlabel('Traditional Accuracy (%)')
        ax2.set_ylabel('LLM Overall Quality Score')
        ax2.set_title('Quality vs Accuracy: RAG Impact')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Quality difference analysis
        if len(rag_data) > 0 and len(no_rag_data) > 0:
            # Calculate average quality differences
            no_rag_avg = no_rag_data['avg_overall_quality'].mean()
            rag_avg = rag_data['avg_overall_quality'].mean()
            quality_diff = rag_avg - no_rag_avg
            
            no_rag_acc = no_rag_data['triage_accuracy'].mean() * 100
            rag_acc = rag_data['triage_accuracy'].mean() * 100
            acc_diff = rag_acc - no_rag_acc
            
            categories = ['Overall Quality', 'Traditional Accuracy']
            differences = [quality_diff, acc_diff]
            colors = ['lightcoral' if d < 0 else 'lightgreen' for d in differences]
            
            bars = ax3.bar(categories, differences, color=colors, alpha=0.8, edgecolor='black')
            
            # Add value labels
            for bar, diff in zip(bars, differences):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.5 if diff >= 0 else -1.5),
                        f'{diff:.1f}', ha='center', va='bottom' if diff >= 0 else 'top', 
                        fontweight='bold', fontsize=12)
            
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax3.set_ylabel('RAG Impact (Δ points)')
            ax3.set_title('RAG Effect on Quality vs Accuracy\n(Positive = RAG Helps)')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Adapter type quality comparison
        adapter_quality = self.df.groupby(['adapter_type', 'has_rag'])['avg_overall_quality'].mean().unstack()
        
        if not adapter_quality.empty:
            adapter_quality.plot(kind='bar', ax=ax4, alpha=0.8, width=0.7)
            ax4.set_xlabel('Adapter Type')
            ax4.set_ylabel('Average Overall Quality Score')
            ax4.set_title('Quality Scores by Adapter Type and RAG Status')
            ax4.legend(['No RAG', 'With RAG'])
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_adapter_quality_comparison(self, filename: str):
        """Compare quality scores across different adapter types"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Quality by adapter type
        adapter_stats = self.df.groupby('adapter_type').agg({
            'avg_overall_quality': ['mean', 'std'],
            'avg_next_step_quality': 'mean',
            'avg_reasoning_quality': 'mean',
            'triage_accuracy': 'mean'
        }).round(2)
        
        adapter_types = adapter_stats.index
        x = np.arange(len(adapter_types))
        width = 0.25
        
        # Plot quality components
        bars1 = ax1.bar(x - width, adapter_stats[('avg_next_step_quality', 'mean')], width,
                       label='Next Step Quality', alpha=0.8, color='lightcoral')
        bars2 = ax1.bar(x, adapter_stats[('avg_reasoning_quality', 'mean')], width,
                       label='Reasoning Quality', alpha=0.8, color='lightblue')  
        bars3 = ax1.bar(x + width, adapter_stats[('avg_overall_quality', 'mean')], width,
                       label='Overall Quality', alpha=0.8, color='lightgreen')
        
        ax1.set_xlabel('Adapter Type')
        ax1.set_ylabel('LLM Quality Score')
        ax1.set_title('Quality Scores by Safety Adapter Type')
        ax1.set_xticks(x)
        ax1.set_xticklabels(adapter_types, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Quality vs Accuracy by adapter
        for adapter in adapter_types:
            adapter_data = self.df[self.df['adapter_type'] == adapter]
            ax2.scatter(adapter_data['triage_accuracy'] * 100, adapter_data['avg_overall_quality'],
                       s=80, alpha=0.7, label=adapter, edgecolors='black', linewidth=0.5)
        
        ax2.set_xlabel('Traditional Accuracy (%)')
        ax2.set_ylabel('LLM Overall Quality Score')
        ax2.set_title('Quality vs Accuracy by Adapter Type')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Quality variance by adapter
        quality_variance = self.df.groupby('adapter_type')['avg_overall_quality'].std()
        bars = ax3.bar(quality_variance.index, quality_variance.values, alpha=0.8, 
                      color='orange', edgecolor='black')
        
        # Add value labels
        for bar, var in zip(bars, quality_variance.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{var:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax3.set_xlabel('Adapter Type')
        ax3.set_ylabel('Quality Score Standard Deviation')
        ax3.set_title('Quality Score Consistency by Adapter Type\n(Lower = More Consistent)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Adapter effectiveness radar
        for i, adapter in enumerate(adapter_types):
            adapter_data = self.df[self.df['adapter_type'] == adapter]
            if not adapter_data.empty:
                values = [
                    adapter_data['avg_next_step_quality'].mean(),
                    adapter_data['avg_reasoning_quality'].mean(),
                    adapter_data['avg_overall_quality'].mean(),
                    adapter_data['triage_accuracy'].mean() * 100
                ]
                
                categories = ['Next Step', 'Reasoning', 'Overall', 'Accuracy']
                angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
                values += values[:1]
                angles += angles[:1]
                
                color = plt.cm.Set2(i)
                ax4.plot(angles, values, 'o-', linewidth=2, label=adapter, color=color, alpha=0.8)
                ax4.fill(angles, values, alpha=0.1, color=color)
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories)
        ax4.set_ylim(0, 80)
        ax4.set_title('Adapter Performance Radar Chart')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_quality_distribution_violin(self, filename: str):
        """Violin plots showing quality score distributions"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Overall quality distribution by configuration
        quality_data = []
        labels = []
        for _, row in self.df.iterrows():
            # Simulate distribution based on mean and std
            if row['std_overall_quality'] > 0:
                simulated_scores = np.random.normal(row['avg_overall_quality'], 
                                                  row['std_overall_quality'], 100)
                simulated_scores = np.clip(simulated_scores, 0, 100)  # Clip to valid range
                quality_data.append(simulated_scores)
                labels.append(row['short_name'])
        
        if quality_data:
            parts = ax1.violinplot(quality_data, showmeans=True, showmedians=True)
            ax1.set_xticks(range(1, len(labels) + 1))
            ax1.set_xticklabels(labels, rotation=45, ha='right')
            ax1.set_ylabel('Overall Quality Score Distribution')
            ax1.set_title('Quality Score Distributions by Configuration\n(Violin Plot)')
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.set_ylim(0, 100)
        
        # Quality components distribution
        quality_components = ['avg_next_step_quality', 'avg_reasoning_quality', 'avg_overall_quality']
        component_names = ['Next Step\nQuality', 'Reasoning\nQuality', 'Overall\nQuality']
        
        component_data = [self.df[col].values for col in quality_components]
        parts = ax2.violinplot(component_data, showmeans=True, showmedians=True)
        ax2.set_xticks(range(1, len(component_names) + 1))
        ax2.set_xticklabels(component_names)
        ax2.set_ylabel('Quality Score')
        ax2.set_title('Quality Component Score Distributions')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 100)
        
        # Quality by RAG status
        no_rag_quality = self.df[self.df['has_rag'] == False]['avg_overall_quality'].values
        rag_quality = self.df[self.df['has_rag'] == True]['avg_overall_quality'].values
        
        parts = ax3.violinplot([no_rag_quality, rag_quality], showmeans=True, showmedians=True)
        ax3.set_xticks([1, 2])
        ax3.set_xticklabels(['No RAG', 'With RAG'])
        ax3.set_ylabel('Overall Quality Score')
        ax3.set_title('Quality Distribution: RAG vs No RAG')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim(0, 100)
        
        # Quality vs Accuracy scatter with density
        ax4.scatter(self.df['triage_accuracy'] * 100, self.df['avg_overall_quality'], 
                   s=100, alpha=0.6, c=self.df['has_rag'].map({True: 'red', False: 'blue'}),
                   edgecolors='black', linewidth=0.5)
        
        # Add marginal histograms as insets
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        
        # Right histogram (quality distribution)
        axins1 = inset_axes(ax4, width="20%", height="50%", loc='upper right')
        axins1.hist(self.df['avg_overall_quality'], bins=10, alpha=0.7, orientation='horizontal', color='gray')
        axins1.set_ylim(ax4.get_ylim())
        axins1.set_xticks([])
        axins1.set_yticks([])
        
        # Top histogram (accuracy distribution)
        axins2 = inset_axes(ax4, width="50%", height="20%", loc='upper left')
        axins2.hist(self.df['triage_accuracy'] * 100, bins=10, alpha=0.7, color='gray')
        axins2.set_xlim(ax4.get_xlim())
        axins2.set_xticks([])
        axins2.set_yticks([])
        
        ax4.set_xlabel('Traditional Accuracy (%)')
        ax4.set_ylabel('LLM Overall Quality Score')
        ax4.set_title('Quality vs Accuracy with Marginal Distributions')
        ax4.grid(True, alpha=0.3)
        
        # Add legend
        blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                               markersize=8, label='No RAG')
        red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                              markersize=8, label='With RAG')
        ax4.legend(handles=[blue_patch, red_patch])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_quality_accuracy_gap_analysis(self, filename: str):
        """Analyze gaps between quality assessment and traditional accuracy"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Quality gap by configuration
        bars = ax1.bar(range(len(self.df)), self.df['quality_gap'], 
                      color=['red' if gap > 0 else 'blue' for gap in self.df['quality_gap']],
                      alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Quality Gap (Accuracy% - LLM Quality)')
        ax1.set_title('Quality-Accuracy Gap by Configuration\n(Positive = Traditional Accuracy Overestimates Quality)')
        ax1.set_xticks(range(len(self.df)))
        ax1.set_xticklabels(self.df['short_name'], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, gap in zip(bars, self.df['quality_gap']):
            ax1.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (1 if gap >= 0 else -2),
                    f'{gap:.1f}', ha='center', 
                    va='bottom' if gap >= 0 else 'top', 
                    fontweight='bold', fontsize=8)
        
        # Gap vs accuracy relationship
        ax2.scatter(self.df['triage_accuracy'] * 100, self.df['quality_gap'], 
                   s=80, alpha=0.7, c=self.df['has_rag'].map({True: 'red', False: 'blue'}),
                   edgecolors='black', linewidth=0.5)
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Traditional Accuracy (%)')
        ax2.set_ylabel('Quality Gap')
        ax2.set_title('Quality Gap vs Traditional Accuracy')
        ax2.grid(True, alpha=0.3)
        
        # Add legend
        blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                               markersize=8, label='No RAG')
        red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                              markersize=8, label='With RAG')
        ax2.legend(handles=[blue_patch, red_patch])
        
        # Gap distribution by RAG status
        no_rag_gaps = self.df[self.df['has_rag'] == False]['quality_gap']
        rag_gaps = self.df[self.df['has_rag'] == True]['quality_gap']
        
        gap_data = [no_rag_gaps.values, rag_gaps.values]
        box_plot = ax3.boxplot(gap_data, labels=['No RAG', 'With RAG'], patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_ylabel('Quality Gap')
        ax3.set_title('Quality Gap Distribution by RAG Status')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Gap statistics summary
        gap_stats = self.df.groupby('has_rag')['quality_gap'].agg(['mean', 'std', 'min', 'max']).round(2)
        
        # Create text summary
        summary_text = "Quality Gap Statistics:\n\n"
        for rag_status, stats in gap_stats.iterrows():
            rag_label = "With RAG" if rag_status else "No RAG"
            summary_text += f"{rag_label}:\n"
            summary_text += f"  Mean: {stats['mean']:.1f}\n"
            summary_text += f"  Std:  {stats['std']:.1f}\n"
            summary_text += f"  Range: [{stats['min']:.1f}, {stats['max']:.1f}]\n\n"
        
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Gap Analysis Summary')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_configuration_quality_ranking(self, filename: str):
        """Comprehensive ranking with multiple metrics including F2, extraction rate, and reliability"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Sort by overall quality
        df_sorted = self.df.sort_values('avg_overall_quality', ascending=True)
        
        y_pos = np.arange(len(df_sorted))
        bars = ax1.barh(y_pos, df_sorted['avg_overall_quality'], alpha=0.8, 
                       color='steelblue', edgecolor='black', linewidth=1)
        
        # Color bars by RAG status
        colors = ['red' if has_rag else 'blue' for has_rag in df_sorted['has_rag']]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            bar.set_alpha(0.7)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(df_sorted['short_name'], fontsize=9)
        ax1.set_xlabel('LLM Overall Quality Score')
        ax1.set_title('Configuration Ranking by LLM Quality\n(Blue = No RAG, Red = With RAG)')
        
        # Enhanced value labels with multiple metrics
        for i, (bar, quality, accuracy, f2, extraction, reliability) in enumerate(zip(
                bars, df_sorted['avg_overall_quality'], df_sorted['triage_accuracy'],
                df_sorted['f2_score'], df_sorted['extraction_success_rate'], df_sorted['reliability_score'])):
            
            label_text = f'Q:{quality:.1f}\nA:{accuracy:.1%}\nF2:{f2:.3f}\nE:{extraction:.3f}\nR:{reliability:.3f}'
            ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    label_text, va='center', fontsize=7, fontweight='bold')
        
        ax1.set_xlim(0, max(df_sorted['avg_overall_quality']) * 1.8)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Multi-metric comparison radar chart
        categories = ['Quality', 'Accuracy', 'F2 Score', 'Extraction\nSuccess', 'Reliability']
        N = len(categories)
        
        # Prepare data for radar chart (normalize to 0-1 scale)
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            values = [
                row['avg_overall_quality'] / 100,  # Quality (0-100 -> 0-1)
                row['triage_accuracy'],  # Already 0-1
                row['f2_score'],  # Already 0-1
                row['extraction_success_rate'],  # Already 0-1
                row['reliability_score']  # Already 0-1
            ]
            values += values[:1]  # Complete the circle
            
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            
            color = plt.cm.Set3(i / len(df_sorted))
            if i < 3:  # Only show top 3 to avoid clutter
                ax2.plot(angles, values, 'o-', linewidth=2, label=f'#{i+1}: {row["short_name"]}', 
                        color=color, alpha=0.8)
                ax2.fill(angles, values, alpha=0.1, color=color)
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories)
        ax2.set_ylim(0, 1)
        ax2.set_title('Top 3 Configurations - Multi-Metric Radar\n(All metrics normalized to 0-1 scale)')
        ax2.legend(bbox_to_anchor=(1.3, 1), loc='upper left', fontsize=8)
        ax2.grid(True)
        
        # Comprehensive metrics heatmap
        metrics_data = df_sorted[[
            'avg_overall_quality', 'triage_accuracy', 'f2_score', 
            'extraction_success_rate', 'reliability_score'
        ]].copy()
        
        # Convert to 0-100 scale for better comparison
        metrics_data['triage_accuracy'] = metrics_data['triage_accuracy'] * 100
        metrics_data['f2_score'] = metrics_data['f2_score'] * 100
        metrics_data['extraction_success_rate'] = metrics_data['extraction_success_rate'] * 100
        metrics_data['reliability_score'] = metrics_data['reliability_score'] * 100
        
        metrics_data.columns = ['Quality', 'Accuracy', 'F2 Score', 'Extraction', 'Reliability']
        metrics_data.index = df_sorted['short_name']
        
        sns.heatmap(metrics_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                   ax=ax3, cbar_kws={'label': 'Score (0-100)'}, 
                   linewidths=0.5, linecolor='white')
        ax3.set_title('Comprehensive Metrics Heatmap\n(All scores on 0-100 scale)')
        ax3.set_xlabel('Metric Type')
        ax3.set_ylabel('Configuration (Sorted by Quality)')
        
        # Ranking differences comparison
        quality_rank = list(range(1, len(df_sorted) + 1))
        accuracy_rank = [list(self.df.sort_values('triage_accuracy', ascending=False).index).index(idx) + 1 
                        for idx in df_sorted.index]
        f2_rank = [list(self.df.sort_values('f2_score', ascending=False).index).index(idx) + 1 
                  for idx in df_sorted.index]
        reliability_rank = [list(self.df.sort_values('reliability_score', ascending=False).index).index(idx) + 1 
                           for idx in df_sorted.index]
        
        x = np.arange(len(df_sorted))
        width = 0.2
        
        acc_diff = np.array(accuracy_rank) - np.array(quality_rank)
        f2_diff = np.array(f2_rank) - np.array(quality_rank)
        rel_diff = np.array(reliability_rank) - np.array(quality_rank)
        
        bars1 = ax4.bar(x - width, acc_diff, width, label='Accuracy vs Quality', 
                       alpha=0.8, color='lightcoral')
        bars2 = ax4.bar(x, f2_diff, width, label='F2 vs Quality', 
                       alpha=0.8, color='lightblue')
        bars3 = ax4.bar(x + width, rel_diff, width, label='Reliability vs Quality', 
                       alpha=0.8, color='lightgreen')
        
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.set_xlabel('Configuration (Sorted by LLM Quality)')
        ax4.set_ylabel('Rank Difference')
        ax4.set_title('Ranking Discrepancies vs LLM Quality\n(Positive = Metric Ranks Higher than Quality)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(df_sorted['short_name'], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_quality_variance_analysis(self, filename: str):
        """Analyze variance and consistency in quality scores"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Quality consistency (std dev) by configuration
        bars = ax1.bar(range(len(self.df)), self.df['std_overall_quality'], 
                      alpha=0.8, color='orange', edgecolor='black', linewidth=0.5)
        
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Quality Score Standard Deviation')
        ax1.set_title('Quality Score Consistency by Configuration\n(Lower = More Consistent)')
        ax1.set_xticks(range(len(self.df)))
        ax1.set_xticklabels(self.df['short_name'], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, std in zip(bars, self.df['std_overall_quality']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{std:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # Mean vs Variance scatter
        ax2.scatter(self.df['avg_overall_quality'], self.df['std_overall_quality'], 
                   s=100, alpha=0.7, c=self.df['has_rag'].map({True: 'red', False: 'blue'}),
                   edgecolors='black', linewidth=0.5)
        
        ax2.set_xlabel('Average Overall Quality Score')
        ax2.set_ylabel('Quality Score Standard Deviation')
        ax2.set_title('Quality Mean vs Variance Relationship')
        ax2.grid(True, alpha=0.3)
        
        # Add configuration labels
        for i, row in self.df.iterrows():
            ax2.annotate(row['short_name'], 
                        (row['avg_overall_quality'], row['std_overall_quality']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
        
        # Add legend
        blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                               markersize=8, label='No RAG')
        red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                              markersize=8, label='With RAG')
        ax2.legend(handles=[blue_patch, red_patch])
        
        # Quality range (min-max) analysis
        quality_ranges = self.df['max_overall_quality'] - self.df['min_overall_quality']
        bars = ax3.bar(range(len(self.df)), quality_ranges, 
                      alpha=0.8, color='purple', edgecolor='black', linewidth=0.5)
        
        ax3.set_xlabel('Configuration')
        ax3.set_ylabel('Quality Score Range (Max - Min)')
        ax3.set_title('Quality Score Range by Configuration')
        ax3.set_xticks(range(len(self.df)))
        ax3.set_xticklabels(self.df['short_name'], rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Coefficient of variation (CV) analysis
        cv_scores = (self.df['std_overall_quality'] / self.df['avg_overall_quality']) * 100
        
        bars = ax4.bar(range(len(self.df)), cv_scores, 
                      alpha=0.8, color='teal', edgecolor='black', linewidth=0.5)
        
        ax4.set_xlabel('Configuration')
        ax4.set_ylabel('Coefficient of Variation (%)')
        ax4.set_title('Quality Score Relative Variability\n(CV = σ/μ × 100%)')
        ax4.set_xticks(range(len(self.df)))
        ax4.set_xticklabels(self.df['short_name'], rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, cv in zip(bars, cv_scores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{cv:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_quality_score_heatmap(self, filename: str):
        """Create comprehensive heatmap of all quality metrics"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Prepare data for heatmap
        heatmap_data = self.df[[
            'avg_next_step_quality', 'avg_reasoning_quality', 'avg_overall_quality',
            'triage_accuracy', 'f1_score', 'f2_score'
        ]].copy()
        
        # Convert accuracy metrics to 0-100 scale for comparison
        heatmap_data['triage_accuracy'] = heatmap_data['triage_accuracy'] * 100
        heatmap_data['f1_score'] = heatmap_data['f1_score'] * 100
        heatmap_data['f2_score'] = heatmap_data['f2_score'] * 100
        
        # Set row labels as short names
        heatmap_data.index = self.df['short_name']
        
        # Rename columns for better display
        heatmap_data.columns = ['Next Step\nQuality', 'Reasoning\nQuality', 'Overall\nQuality', 
                               'Accuracy', 'F1 Score', 'F2 Score']
        
        # Create main heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                   ax=ax1, cbar_kws={'label': 'Score (0-100)'}, 
                   linewidths=0.5, linecolor='white')
        ax1.set_title('Comprehensive Quality and Performance Heatmap\n(All Scores on 0-100 Scale)')
        ax1.set_xlabel('Metric Type')
        ax1.set_ylabel('Configuration')
        
        # Create correlation heatmap
        corr_matrix = heatmap_data.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Hide upper triangle
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r', 
                   center=0, ax=ax2, square=True, cbar_kws={'label': 'Correlation Coefficient'},
                   linewidths=0.5, linecolor='white')
        ax2.set_title('Quality and Performance Correlation Matrix')
        ax2.set_xlabel('Metric')
        ax2.set_ylabel('Metric')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_accuracy_vs_recall_tradeoffs(self, filename: str):
        """Bar chart showing accuracy vs ED recall vs HOME recall tradeoffs"""
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Sort by accuracy for consistent ordering
        df_sorted = self.df.sort_values('triage_accuracy', ascending=False)
        
        x = np.arange(len(df_sorted))
        width = 0.25
        
        # Create grouped bar chart
        bars1 = ax.bar(x - width, df_sorted['triage_accuracy'] * 100, width, 
                      label='Overall Accuracy (%)', alpha=0.8, color='steelblue')
        bars2 = ax.bar(x, df_sorted['ed_recall'] * 100, width, 
                      label='ED Recall (%)', alpha=0.8, color='red')
        bars3 = ax.bar(x + width, df_sorted['home_recall'] * 100, width, 
                      label='HOME Recall (%)', alpha=0.8, color='green')
        
        # Add value labels on bars
        def add_labels(bars, values):
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{value*100:.1f}%', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
        
        add_labels(bars1, df_sorted['triage_accuracy'])
        add_labels(bars2, df_sorted['ed_recall'])
        add_labels(bars3, df_sorted['home_recall'])
        
        # Customize plot
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Performance (%)')
        ax.set_title('Accuracy vs Safety Tradeoffs: Overall vs Emergency vs Home Detection\n(Shows critical trade-off between accuracy and emergency detection)')
        ax.set_xticks(x)
        ax.set_xticklabels(df_sorted['short_name'], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add annotation highlighting the tradeoff
        ax.annotate('Safety-Critical Trade-off:\nHigh accuracy ≠ High ED detection', 
                   xy=(0, 70), xytext=(0.5, 85),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   fontsize=11, fontweight='bold', ha='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_latency_vs_accuracy_scatter(self, filename: str):
        """Scatter plot showing latency vs accuracy with RAG cost visualization"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create scatter plot with different colors for RAG vs No-RAG
        no_rag_data = self.df[self.df['has_rag'] == False]
        rag_data = self.df[self.df['has_rag'] == True]
        
        # Plot No-RAG configurations
        scatter1 = ax.scatter(no_rag_data['avg_inference_time'], 
                             no_rag_data['triage_accuracy'] * 100,
                             s=150, alpha=0.8, c='blue', marker='o', 
                             edgecolors='black', linewidth=2, label='No RAG')
        
        # Plot RAG configurations  
        scatter2 = ax.scatter(rag_data['avg_inference_time'], 
                             rag_data['triage_accuracy'] * 100,
                             s=150, alpha=0.8, c='red', marker='s', 
                             edgecolors='black', linewidth=2, label='With RAG')
        
        # Add configuration labels
        for _, row in self.df.iterrows():
            ax.annotate(row['short_name'], 
                       (row['avg_inference_time'], row['triage_accuracy'] * 100),
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=9, alpha=0.8, fontweight='bold')
        
        # Add trend lines
        if len(no_rag_data) > 1:
            z1 = np.polyfit(no_rag_data['avg_inference_time'], no_rag_data['triage_accuracy'] * 100, 1)
            p1 = np.poly1d(z1)
            ax.plot(no_rag_data['avg_inference_time'], 
                   p1(no_rag_data['avg_inference_time']), 
                   "b--", alpha=0.8, linewidth=2, label='No RAG trend')
        
        if len(rag_data) > 1:
            z2 = np.polyfit(rag_data['avg_inference_time'], rag_data['triage_accuracy'] * 100, 1)
            p2 = np.poly1d(z2)
            ax.plot(rag_data['avg_inference_time'], 
                   p2(rag_data['avg_inference_time']), 
                   "r--", alpha=0.8, linewidth=2, label='RAG trend')
        
        # Calculate and display RAG cost
        if len(no_rag_data) > 0 and len(rag_data) > 0:
            avg_no_rag_latency = no_rag_data['avg_inference_time'].mean()
            avg_rag_latency = rag_data['avg_inference_time'].mean()
            avg_no_rag_acc = no_rag_data['triage_accuracy'].mean() * 100
            avg_rag_acc = rag_data['triage_accuracy'].mean() * 100
            
            latency_increase = (avg_rag_latency / avg_no_rag_latency - 1) * 100
            accuracy_change = avg_rag_acc - avg_no_rag_acc
            
            # Add cost annotation
            cost_text = f'RAG Cost:\\n+{latency_increase:.0f}% latency\\n{accuracy_change:+.1f}% accuracy' ax.text(0.02, 0.98, cost_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        # Customize plot
        ax.set_xlabel('Average Inference Time (seconds per case)')
        ax.set_ylabel('Triage Accuracy (%)')
        ax.set_title('Latency vs Accuracy Trade-off\\n(Illustrates the computational cost of RAG)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set reasonable axis limits
        ax.set_xlim(0, max(self.df['avg_inference_time']) * 1.1)
        ax.set_ylim(min(self.df['triage_accuracy']) * 95, 
                   max(self.df['triage_accuracy']) * 105)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate LLM-as-Judge Visualization Plots")
    parser.add_argument("results_file", 
                       help="Path to LLM-as-judge results JSON file")
    
    args = parser.parse_args()
    
    try:
        generator = LLMJudgePlotsGenerator(args.results_file)
        generator.generate_all_plots()
        
        print(f"\n✅ LLM-as-judge plots complete! Check the '{generator.output_dir}' directory for 12 high-quality plots:")
        print("  📊 01_quality_scores_overview.png - Quality scores by configuration")
        print("  📊 02_quality_vs_accuracy_correlation.png - Quality-accuracy correlations")
        print("  📊 03_quality_components_breakdown.png - Radar chart and correlation matrix")
        print("  📊 04_rag_quality_impact.png - RAG impact on quality assessment")
        print("  📊 05_adapter_quality_comparison.png - Quality by adapter type")
        print("  📊 06_quality_distribution_violin.png - Quality score distributions")
        print("  📊 07_quality_accuracy_gap_analysis.png - Quality vs accuracy gaps")
        print("  📊 08_configuration_quality_ranking.png - Quality-based ranking")
        print("  📊 09_quality_variance_analysis.png - Quality consistency analysis")
        print("  📊 10_quality_score_heatmap.png - Comprehensive quality heatmap")
        print("  📊 11_accuracy_vs_recall_tradeoffs.png - Accuracy vs ED/HOME recall tradeoffs")
        print("  📊 12_latency_vs_accuracy_scatter.png - Latency vs accuracy with RAG cost")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()