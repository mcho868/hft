#!/usr/bin/env python3
"""
Advanced Medical Triage Evaluation Analysis Dashboard - UNKNOWN Tracking Focus
Analyzes results with proper UNKNOWN extraction failure tracking.

Key Analysis Areas:
1. Honest accuracy metrics (no GP inflation)
2. Extraction failure rates (UNKNOWN cases)
3. Model reliability across different conditions
4. RAG impact on both accuracy and extraction success
5. Class-specific performance breakdowns
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

class UnknownTrackingEvaluationDashboard:
    """Advanced dashboard for analyzing medical triage evaluation with UNKNOWN tracking"""
    
    def __init__(self, progress_file: str):
        self.progress_file = Path(progress_file)
        self.results_data = self._load_progress_results()
        self.df = self._create_dataframe()
        
        # Set style for publication-quality plots
        plt.style.use('default')
        sns.set_palette("Set2")
        
        # Create output directory
        self.output_dir = Path("plots_unknown_tracking")
        self.output_dir.mkdir(exist_ok=True)
        
    def _load_progress_results(self) -> List[Dict[str, Any]]:
        """Load results from progress file"""
        if not self.progress_file.exists():
            raise FileNotFoundError(f"Progress file not found: {self.progress_file}")
        
        print(f"📊 Loading UNKNOWN tracking results from: {self.progress_file}")
        
        with open(self.progress_file, 'r') as f:
            progress_data = json.load(f)
        
        results = progress_data.get('results', [])
        completed = progress_data.get('completed_configs', 0)
        total = progress_data.get('total_configs', 96)
        
        print(f"✅ Loaded {len(results)} results ({completed}/{total} completed)")
        return results
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis"""
        data = []
        
        for result in self.results_data:
            config = result['config']
            
            # Parse configuration details
            model_name = config['model_name']
            adapter_name = Path(config['adapter_path']).name.split('_')[-2] if config['adapter_path'] else 'none'
            has_rag = config['rag_config'] is not None
            rag_name = config['rag_config']['name'] if has_rag else 'NoRAG'
            
            # Extract key metrics
            row = {
                # Configuration info
                'model_name': model_name,
                'model_family': model_name.split('-')[0],
                'model_size': model_name.split('-')[1],
                'quantization': '4bit' if '4bit' in model_name else '8bit',
                'adapter_type': adapter_name,
                'has_rag': has_rag,
                'rag_config': rag_name,
                'test_name': config['test_name'],
                
                # Performance metrics
                'triage_accuracy': result['triage_accuracy'],
                'f1_score': result['f1_score'],
                'f2_score': result['f2_score'],
                
                # UNKNOWN tracking metrics (KEY ADDITION)
                'unknown_triage_count': result.get('unknown_triage_count', 0),
                'total_failures': result.get('total_failures', 0),
                'extraction_success_rate': 1 - (result.get('unknown_triage_count', 0) / 200),
                'effective_accuracy': result['triage_accuracy'],  # Already honest with UNKNOWN
                
                # Timing
                'avg_inference_time': result['avg_inference_time_per_case'],
                'rag_time': result.get('rag_retrieval_time', 0.0) or 0.0,
                
                # Class-specific performance from confusion matrix
                'confusion_matrix': result['confusion_matrix'],
            }
            
            # Extract class-specific metrics from classification report
            class_report = result.get('classification_report', {})
            for class_name in ['ED', 'GP', 'HOME']:
                if class_name in class_report:
                    row[f'{class_name.lower()}_precision'] = class_report[class_name]['precision']
                    row[f'{class_name.lower()}_recall'] = class_report[class_name]['recall']
                    row[f'{class_name.lower()}_f1'] = class_report[class_name]['f1-score']
            
            data.append(row)
        
        df = pd.DataFrame(data)
        print(f"📈 Created DataFrame with {len(df)} configurations")
        return df
    
    def create_comprehensive_analysis(self):
        """Generate comprehensive analysis focused on UNKNOWN tracking insights"""
        
        print("🎨 Creating comprehensive UNKNOWN tracking analysis...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Overall Performance Distribution (with UNKNOWN impact)
        ax1 = plt.subplot(4, 3, 1)
        self._plot_accuracy_distribution_with_unknown(ax1)
        
        # 2. Extraction Success Rate Analysis
        ax2 = plt.subplot(4, 3, 2)
        self._plot_extraction_success_rates(ax2)
        
        # 3. Model Reliability Comparison
        ax3 = plt.subplot(4, 3, 3)
        self._plot_model_reliability_comparison(ax3)
        
        # 4. RAG Impact on Accuracy vs Extraction
        ax4 = plt.subplot(4, 3, 4)
        self._plot_rag_impact_dual_metrics(ax4)
        
        # 5. Class-Specific Performance Breakdown
        ax5 = plt.subplot(4, 3, 5)
        self._plot_class_performance_breakdown(ax5)
        
        # 6. Adapter Type Effectiveness
        ax6 = plt.subplot(4, 3, 6)
        self._plot_adapter_effectiveness(ax6)
        
        # 7. Accuracy vs Unknown Rate Scatter
        ax7 = plt.subplot(4, 3, 7)
        self._plot_accuracy_vs_unknown_scatter(ax7)
        
        # 8. Model Size Impact Analysis
        ax8 = plt.subplot(4, 3, 8)
        self._plot_model_size_impact(ax8)
        
        # 9. Performance vs Speed Trade-off
        ax9 = plt.subplot(4, 3, 9)
        self._plot_performance_speed_tradeoff(ax9)
        
        # 10. Top Performing Configurations
        ax10 = plt.subplot(4, 3, 10)
        self._plot_top_configurations(ax10)
        
        # 11. RAG Configuration Effectiveness
        ax11 = plt.subplot(4, 3, 11)
        self._plot_rag_config_effectiveness(ax11)
        
        # 12. Failure Analysis Heatmap
        ax12 = plt.subplot(4, 3, 12)
        self._plot_failure_analysis_heatmap(ax12)
        
        plt.tight_layout(pad=3.0)
        plt.savefig(self.output_dir / 'comprehensive_unknown_tracking_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Generate summary statistics
        self._generate_summary_report()
    
    def _plot_accuracy_distribution_with_unknown(self, ax):
        """Plot accuracy distribution highlighting UNKNOWN impact"""
        
        # Create bins for accuracy ranges
        bins = np.arange(0, 1.01, 0.05)
        
        # Plot histogram with UNKNOWN rate coloring
        scatter = ax.scatter(self.df['triage_accuracy'], 
                           self.df['unknown_triage_count'], 
                           c=self.df['extraction_success_rate'],
                           s=60, alpha=0.7, cmap='RdYlGn')
        
        ax.set_xlabel('Triage Accuracy')
        ax.set_ylabel('Unknown Cases Count')
        ax.set_title('Accuracy vs Unknown Cases\n(Color = Extraction Success Rate)')
        plt.colorbar(scatter, ax=ax, label='Extraction Success Rate')
        
        # Add trend line
        z = np.polyfit(self.df['triage_accuracy'], self.df['unknown_triage_count'], 1)
        p = np.poly1d(z)
        ax.plot(self.df['triage_accuracy'], p(self.df['triage_accuracy']), "r--", alpha=0.8)
    
    def _plot_extraction_success_rates(self, ax):
        """Plot extraction success rates by configuration type"""
        
        # Group by model family and RAG status
        grouped = self.df.groupby(['model_family', 'has_rag'])['extraction_success_rate'].mean().reset_index()
        
        # Create grouped bar plot
        x = np.arange(len(grouped['model_family'].unique()))
        width = 0.35
        
        no_rag = grouped[grouped['has_rag'] == False]['extraction_success_rate']
        with_rag = grouped[grouped['has_rag'] == True]['extraction_success_rate']
        
        ax.bar(x - width/2, no_rag, width, label='No RAG', alpha=0.8)
        ax.bar(x + width/2, with_rag, width, label='With RAG', alpha=0.8)
        
        ax.set_xlabel('Model Family')
        ax.set_ylabel('Extraction Success Rate')
        ax.set_title('Extraction Success Rate by Model & RAG')
        ax.set_xticks(x)
        ax.set_xticklabels(grouped['model_family'].unique())
        ax.legend()
        ax.set_ylim(0, 1)
    
    def _plot_model_reliability_comparison(self, ax):
        """Compare model reliability (accuracy + extraction success)"""
        
        # Calculate reliability score (weighted combination)
        self.df['reliability_score'] = (self.df['triage_accuracy'] * 0.7 + 
                                       self.df['extraction_success_rate'] * 0.3)
        
        # Group by model and adapter
        reliability_data = self.df.groupby(['model_family', 'adapter_type'])['reliability_score'].mean().reset_index()
        
        # Create heatmap
        pivot_data = reliability_data.pivot(index='adapter_type', 
                                          columns='model_family', 
                                          values='reliability_score')
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax)
        ax.set_title('Model Reliability Score\n(70% Accuracy + 30% Extraction Success)')
    
    def _plot_rag_impact_dual_metrics(self, ax):
        """Show RAG impact on both accuracy and extraction success"""
        
        rag_comparison = self.df.groupby(['model_name', 'has_rag']).agg({
            'triage_accuracy': 'mean',
            'extraction_success_rate': 'mean'
        }).reset_index()
        
        # Calculate RAG benefit
        no_rag = rag_comparison[rag_comparison['has_rag'] == False]
        with_rag = rag_comparison[rag_comparison['has_rag'] == True]
        
        models = no_rag['model_name'].unique()
        
        accuracy_benefit = []
        extraction_benefit = []
        
        for model in models:
            no_rag_model = no_rag[no_rag['model_name'] == model]
            with_rag_model = with_rag[with_rag['model_name'] == model]
            
            if not no_rag_model.empty and not with_rag_model.empty:
                acc_benefit = with_rag_model['triage_accuracy'].mean() - no_rag_model['triage_accuracy'].mean()
                ext_benefit = with_rag_model['extraction_success_rate'].mean() - no_rag_model['extraction_success_rate'].mean()
                
                accuracy_benefit.append(acc_benefit)
                extraction_benefit.append(ext_benefit)
        
        x = np.arange(len(models))
        width = 0.35
        
        ax.bar(x - width/2, accuracy_benefit, width, label='Accuracy Benefit', alpha=0.8)
        ax.bar(x + width/2, extraction_benefit, width, label='Extraction Benefit', alpha=0.8)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('RAG Benefit (Δ)')
        ax.set_title('RAG Benefits: Accuracy vs Extraction')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('SmolLM2-', '').replace('Gemma-', '') for m in models], rotation=45)
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    def _plot_class_performance_breakdown(self, ax):
        """Show performance breakdown by class (ED/GP/HOME)"""
        
        # Calculate average F1 scores by class
        class_f1_data = []
        for class_name in ['ed', 'gp', 'home']:
            f1_col = f'{class_name}_f1'
            if f1_col in self.df.columns:
                avg_f1 = self.df[f1_col].mean()
                class_f1_data.append({'class': class_name.upper(), 'avg_f1': avg_f1})
        
        class_df = pd.DataFrame(class_f1_data)
        
        colors = ['#ff7f7f', '#7f7fff', '#7fff7f']  # Red, Blue, Green for ED, GP, HOME
        bars = ax.bar(class_df['class'], class_df['avg_f1'], color=colors, alpha=0.8)
        
        ax.set_xlabel('Triage Class')
        ax.set_ylabel('Average F1 Score')
        ax.set_title('Class-Specific Performance\n(Average F1 Scores)')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, class_df['avg_f1']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_adapter_effectiveness(self, ax):
        """Compare effectiveness of different adapter types"""
        
        adapter_performance = self.df.groupby('adapter_type').agg({
            'triage_accuracy': 'mean',
            'extraction_success_rate': 'mean',
            'f2_score': 'mean'
        }).reset_index()
        
        # Create grouped bar plot
        x = np.arange(len(adapter_performance))
        width = 0.25
        
        ax.bar(x - width, adapter_performance['triage_accuracy'], width, 
               label='Accuracy', alpha=0.8)
        ax.bar(x, adapter_performance['extraction_success_rate'], width, 
               label='Extraction Success', alpha=0.8)
        ax.bar(x + width, adapter_performance['f2_score'], width, 
               label='F2 Score', alpha=0.8)
        
        ax.set_xlabel('Adapter Type')
        ax.set_ylabel('Performance Score')
        ax.set_title('Adapter Type Effectiveness')
        ax.set_xticks(x)
        ax.set_xticklabels(adapter_performance['adapter_type'], rotation=45)
        ax.legend()
        ax.set_ylim(0, 1)
    
    def _plot_accuracy_vs_unknown_scatter(self, ax):
        """Scatter plot showing relationship between accuracy and unknown rate"""
        
        # Color by model family
        model_families = self.df['model_family'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(model_families)))
        
        for i, family in enumerate(model_families):
            family_data = self.df[self.df['model_family'] == family]
            ax.scatter(family_data['unknown_triage_count'], 
                      family_data['triage_accuracy'],
                      c=[colors[i]], label=family, s=60, alpha=0.7)
        
        ax.set_xlabel('Unknown Cases Count')
        ax.set_ylabel('Triage Accuracy')
        ax.set_title('Accuracy vs Unknown Cases by Model Family')
        ax.legend()
        
        # Add correlation coefficient
        corr = self.df['unknown_triage_count'].corr(self.df['triage_accuracy'])
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_model_size_impact(self, ax):
        """Analyze impact of model size on performance"""
        
        size_performance = self.df.groupby('model_size').agg({
            'triage_accuracy': ['mean', 'std'],
            'extraction_success_rate': ['mean', 'std'],
            'avg_inference_time': 'mean'
        }).reset_index()
        
        # Flatten column names
        size_performance.columns = ['model_size', 'acc_mean', 'acc_std', 
                                   'ext_mean', 'ext_std', 'time_mean']
        
        x = np.arange(len(size_performance))
        width = 0.35
        
        ax.bar(x - width/2, size_performance['acc_mean'], width,
               yerr=size_performance['acc_std'], label='Accuracy', 
               alpha=0.8, capsize=5)
        ax.bar(x + width/2, size_performance['ext_mean'], width,
               yerr=size_performance['ext_std'], label='Extraction Success', 
               alpha=0.8, capsize=5)
        
        ax.set_xlabel('Model Size')
        ax.set_ylabel('Performance')
        ax.set_title('Model Size Impact on Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(size_performance['model_size'])
        ax.legend()
        ax.set_ylim(0, 1)
    
    def _plot_performance_speed_tradeoff(self, ax):
        """Show performance vs speed trade-off"""
        
        # Create bubble plot where bubble size represents reliability score
        bubble_sizes = self.df['reliability_score'] * 200  # Scale for visibility
        
        scatter = ax.scatter(self.df['avg_inference_time'], 
                           self.df['triage_accuracy'],
                           s=bubble_sizes, 
                           c=self.df['unknown_triage_count'],
                           cmap='RdYlBu_r', alpha=0.6)
        
        ax.set_xlabel('Average Inference Time (s/case)')
        ax.set_ylabel('Triage Accuracy')
        ax.set_title('Performance vs Speed Trade-off\n(Bubble size = Reliability, Color = Unknown count)')
        plt.colorbar(scatter, ax=ax, label='Unknown Cases')
    
    def _plot_top_configurations(self, ax):
        """Show top performing configurations"""
        
        # Rank by reliability score
        top_configs = self.df.nlargest(10, 'reliability_score')[
            ['test_name', 'reliability_score', 'triage_accuracy', 'extraction_success_rate']
        ]
        
        # Shorten test names for display
        top_configs['short_name'] = top_configs['test_name'].apply(
            lambda x: x.replace('_FineTuned_adapter_safe_triage_', '_').replace('SmolLM2-', 'S')
        )
        
        y_pos = np.arange(len(top_configs))
        bars = ax.barh(y_pos, top_configs['reliability_score'], alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_configs['short_name'], fontsize=8)
        ax.set_xlabel('Reliability Score')
        ax.set_title('Top 10 Configurations\n(by Reliability Score)')
        
        # Add values on bars
        for i, (bar, score) in enumerate(zip(bars, top_configs['reliability_score'])):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', va='center', fontsize=8)
    
    def _plot_rag_config_effectiveness(self, ax):
        """Compare different RAG configurations"""
        
        rag_data = self.df[self.df['has_rag'] == True]
        
        if not rag_data.empty:
            rag_performance = rag_data.groupby('rag_config').agg({
                'triage_accuracy': 'mean',
                'extraction_success_rate': 'mean',
                'rag_time': 'mean'
            }).reset_index()
            
            # Create grouped bar plot
            x = np.arange(len(rag_performance))
            width = 0.35
            
            ax.bar(x - width/2, rag_performance['triage_accuracy'], width,
                   label='Accuracy', alpha=0.8)
            ax.bar(x + width/2, rag_performance['extraction_success_rate'], width,
                   label='Extraction Success', alpha=0.8)
            
            ax.set_xlabel('RAG Configuration')
            ax.set_ylabel('Performance')
            ax.set_title('RAG Configuration Effectiveness')
            ax.set_xticks(x)
            ax.set_xticklabels([name.replace('top', 'T').replace('_', '_\n') 
                               for name in rag_performance['rag_config']], fontsize=8)
            ax.legend()
            ax.set_ylim(0, 1)
        else:
            ax.text(0.5, 0.5, 'No RAG data available', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('RAG Configuration Effectiveness')
    
    def _plot_failure_analysis_heatmap(self, ax):
        """Create heatmap of failure patterns"""
        
        # Create failure analysis matrix
        failure_data = self.df.pivot_table(
            values='unknown_triage_count',
            index='adapter_type',
            columns='model_family',
            aggfunc='mean'
        )
        
        sns.heatmap(failure_data, annot=True, fmt='.1f', cmap='Reds', ax=ax)
        ax.set_title('Average Unknown Cases\n(by Model & Adapter)')
        ax.set_xlabel('Model Family')
        ax.set_ylabel('Adapter Type')
    
    def _generate_summary_report(self):
        """Generate comprehensive summary report"""
        
        report_path = self.output_dir / 'evaluation_summary_unknown_tracking.txt'
        
        with open(report_path, 'w') as f:
            f.write("MEDICAL TRIAGE EVALUATION SUMMARY - UNKNOWN TRACKING ANALYSIS\n")
            f.write("=" * 70 + "\n\n")
            
            # Overall Statistics
            f.write("OVERALL PERFORMANCE STATISTICS:\n")
            f.write("-" * 35 + "\n")
            f.write(f"Total Configurations Evaluated: {len(self.df)}\n")
            f.write(f"Average Triage Accuracy: {self.df['triage_accuracy'].mean():.3f} ± {self.df['triage_accuracy'].std():.3f}\n")
            f.write(f"Average Extraction Success Rate: {self.df['extraction_success_rate'].mean():.3f} ± {self.df['extraction_success_rate'].std():.3f}\n")
            f.write(f"Average Unknown Cases per Config: {self.df['unknown_triage_count'].mean():.1f} ± {self.df['unknown_triage_count'].std():.1f}\n")
            f.write(f"Average F2 Score: {self.df['f2_score'].mean():.3f} ± {self.df['f2_score'].std():.3f}\n\n")
            
            # Best Performing Configurations
            f.write("TOP 5 MOST RELIABLE CONFIGURATIONS:\n")
            f.write("-" * 38 + "\n")
            top_5 = self.df.nlargest(5, 'reliability_score')
            for i, (_, config) in enumerate(top_5.iterrows(), 1):
                f.write(f"{i}. {config['test_name']}\n")
                f.write(f"   Reliability Score: {config['reliability_score']:.3f}\n")
                f.write(f"   Accuracy: {config['triage_accuracy']:.3f}\n")
                f.write(f"   Extraction Success: {config['extraction_success_rate']:.3f}\n")
                f.write(f"   Unknown Cases: {config['unknown_triage_count']}\n\n")
            
            # Model Family Comparison
            f.write("PERFORMANCE BY MODEL FAMILY:\n")
            f.write("-" * 32 + "\n")
            family_stats = self.df.groupby('model_family').agg({
                'triage_accuracy': ['mean', 'std'],
                'extraction_success_rate': ['mean', 'std'],
                'unknown_triage_count': 'mean'
            })
            
            for family in family_stats.index:
                f.write(f"{family}:\n")
                f.write(f"  Accuracy: {family_stats.loc[family, ('triage_accuracy', 'mean')]:.3f} ± {family_stats.loc[family, ('triage_accuracy', 'std')]:.3f}\n")
                f.write(f"  Extraction Success: {family_stats.loc[family, ('extraction_success_rate', 'mean')]:.3f} ± {family_stats.loc[family, ('extraction_success_rate', 'std')]:.3f}\n")
                f.write(f"  Avg Unknown Cases: {family_stats.loc[family, ('unknown_triage_count', 'mean')]:.1f}\n\n")
            
            # RAG Impact Analysis
            f.write("RAG IMPACT ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            rag_comparison = self.df.groupby('has_rag').agg({
                'triage_accuracy': 'mean',
                'extraction_success_rate': 'mean',
                'unknown_triage_count': 'mean'
            })
            
            if True in rag_comparison.index and False in rag_comparison.index:
                no_rag = rag_comparison.loc[False]
                with_rag = rag_comparison.loc[True]
                
                f.write(f"Without RAG:\n")
                f.write(f"  Accuracy: {no_rag['triage_accuracy']:.3f}\n")
                f.write(f"  Extraction Success: {no_rag['extraction_success_rate']:.3f}\n")
                f.write(f"  Avg Unknown Cases: {no_rag['unknown_triage_count']:.1f}\n\n")
                
                f.write(f"With RAG:\n")
                f.write(f"  Accuracy: {with_rag['triage_accuracy']:.3f}\n")
                f.write(f"  Extraction Success: {with_rag['extraction_success_rate']:.3f}\n")
                f.write(f"  Avg Unknown Cases: {with_rag['unknown_triage_count']:.1f}\n\n")
                
                f.write(f"RAG Benefits:\n")
                f.write(f"  Accuracy Improvement: {with_rag['triage_accuracy'] - no_rag['triage_accuracy']:.3f}\n")
                f.write(f"  Extraction Improvement: {with_rag['extraction_success_rate'] - no_rag['extraction_success_rate']:.3f}\n")
                f.write(f"  Unknown Reduction: {no_rag['unknown_triage_count'] - with_rag['unknown_triage_count']:.1f}\n\n")
            
            # Key Insights
            f.write("KEY INSIGHTS FROM UNKNOWN TRACKING:\n")
            f.write("-" * 40 + "\n")
            f.write("1. Extraction Success Rate provides crucial insight into model reliability\n")
            f.write("2. UNKNOWN cases reveal parsing/format issues vs true performance gaps\n")
            f.write("3. Reliability Score (accuracy + extraction) gives holistic view\n")
            f.write("4. RAG impact should be measured on both accuracy AND extraction success\n")
            f.write("5. Model comparison is now honest (no artificial GP inflation)\n")
        
        print(f"📝 Summary report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Advanced Medical Triage Evaluation Analysis with UNKNOWN Tracking")
    parser.add_argument("progress_file", 
                       help="Path to evaluation progress JSON file")
    parser.add_argument("--output-dir", 
                       default="plots_unknown_tracking",
                       help="Output directory for plots and analysis")
    
    args = parser.parse_args()
    
    try:
        dashboard = UnknownTrackingEvaluationDashboard(args.progress_file)
        dashboard.create_comprehensive_analysis()
        
        print(f"\n✅ Analysis complete! Check the '{dashboard.output_dir}' directory for:")
        print("  📊 comprehensive_unknown_tracking_analysis.png - Main analysis dashboard")
        print("  📝 evaluation_summary_unknown_tracking.txt - Detailed summary report")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()