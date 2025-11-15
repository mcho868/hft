#!/usr/bin/env python3
"""
Individual Plot Generator for Medical Triage Evaluation - UNKNOWN Tracking Focus
Creates separate high-quality figures for each analysis component.
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

class IndividualPlotsGenerator:
    """Generate individual high-quality plots for medical triage evaluation analysis"""
    
    def __init__(self, progress_file: str):
        self.progress_file = Path(progress_file)
        self.results_data = self._load_progress_results()
        self.df = self._create_dataframe()
        
        # Set style for publication-quality plots
        plt.style.use('default')
        sns.set_palette("Set2")
        plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
        
        # Create output directory
        self.output_dir = Path("individual_plots_unknown_tracking")
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
                
                # UNKNOWN tracking metrics
                'unknown_triage_count': result.get('unknown_triage_count', 0),
                'total_failures': result.get('total_failures', 0),
                'extraction_success_rate': 1 - (result.get('unknown_triage_count', 0) / 200),
                'effective_accuracy': result['triage_accuracy'],
                
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
        
        # Calculate reliability score
        df['reliability_score'] = (df['triage_accuracy'] * 0.7 + df['extraction_success_rate'] * 0.3)
        
        print(f"📈 Created DataFrame with {len(df)} configurations")
        return df
    
    def generate_all_plots(self):
        """Generate all individual plots"""
        
        print("🎨 Creating individual plots...")
        
        plot_functions = [
            ('01_accuracy_vs_unknown_scatter', self.plot_accuracy_vs_unknown_scatter),
            ('02_extraction_success_rates', self.plot_extraction_success_rates),
            ('03_model_reliability_heatmap', self.plot_model_reliability_heatmap),
            ('04_rag_impact_comparison', self.plot_rag_impact_comparison),
            ('05_class_performance_breakdown', self.plot_class_performance_breakdown),
            ('06_adapter_effectiveness', self.plot_adapter_effectiveness),
            ('07_model_size_impact', self.plot_model_size_impact),
            ('08_performance_speed_tradeoff', self.plot_performance_speed_tradeoff),
            ('09_top_configurations', self.plot_top_configurations),
            ('10_rag_config_effectiveness', self.plot_rag_config_effectiveness),
            ('11_failure_analysis_heatmap', self.plot_failure_analysis_heatmap),
            ('12_accuracy_distribution', self.plot_accuracy_distribution),
        ]
        
        for plot_name, plot_function in plot_functions:
            try:
                print(f"  📊 Creating {plot_name}...")
                plot_function(plot_name)
            except Exception as e:
                print(f"  ❌ Error creating {plot_name}: {e}")
        
        print(f"✅ All plots saved to: {self.output_dir}")
    
    def plot_accuracy_vs_unknown_scatter(self, filename: str):
        """Scatter plot showing relationship between accuracy and unknown cases"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Color by model family
        model_families = self.df['model_family'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(model_families)))
        
        for i, family in enumerate(model_families):
            family_data = self.df[self.df['model_family'] == family]
            ax.scatter(family_data['unknown_triage_count'], 
                      family_data['triage_accuracy'],
                      c=[colors[i]], label=family, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Unknown Cases Count')
        ax.set_ylabel('Triage Accuracy')
        ax.set_title('Accuracy vs Unknown Cases by Model Family\n(Honest Metrics - No GP Inflation)')
        ax.legend(title='Model Family')
        ax.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = self.df['unknown_triage_count'].corr(self.df['triage_accuracy'])
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add trend line
        z = np.polyfit(self.df['unknown_triage_count'], self.df['triage_accuracy'], 1)
        p = np.poly1d(z)
        ax.plot(self.df['unknown_triage_count'], p(self.df['unknown_triage_count']), 
               "r--", alpha=0.8, linewidth=2, label=f'Trend (r={corr:.3f})')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_extraction_success_rates(self, filename: str):
        """Plot extraction success rates by model family and RAG status"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Group by model family and RAG status
        grouped = self.df.groupby(['model_family', 'has_rag'])['extraction_success_rate'].mean().reset_index()
        
        # Create grouped bar plot
        model_families = grouped['model_family'].unique()
        x = np.arange(len(model_families))
        width = 0.35
        
        no_rag_data = []
        with_rag_data = []
        
        for family in model_families:
            no_rag = grouped[(grouped['model_family'] == family) & (grouped['has_rag'] == False)]
            with_rag = grouped[(grouped['model_family'] == family) & (grouped['has_rag'] == True)]
            
            no_rag_data.append(no_rag['extraction_success_rate'].iloc[0] if not no_rag.empty else 0)
            with_rag_data.append(with_rag['extraction_success_rate'].iloc[0] if not with_rag.empty else 0)
        
        bars1 = ax.bar(x - width/2, no_rag_data, width, label='No RAG', alpha=0.8, color='lightcoral')
        bars2 = ax.bar(x + width/2, with_rag_data, width, label='With RAG', alpha=0.8, color='lightblue')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Model Family')
        ax.set_ylabel('Extraction Success Rate')
        ax.set_title('Extraction Success Rate by Model Family and RAG Status\n(Higher = Better Format Compliance)')
        ax.set_xticks(x)
        ax.set_xticklabels(model_families)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_reliability_heatmap(self, filename: str):
        """Heatmap showing model reliability scores"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create reliability matrix
        reliability_data = self.df.groupby(['model_family', 'adapter_type'])['reliability_score'].mean().reset_index()
        pivot_data = reliability_data.pivot(index='adapter_type', 
                                          columns='model_family', 
                                          values='reliability_score')
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   ax=ax, cbar_kws={'label': 'Reliability Score'})
        ax.set_title('Model Reliability Score Heatmap\n(70% Accuracy + 30% Extraction Success)')
        ax.set_xlabel('Model Family')
        ax.set_ylabel('Adapter Type')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_rag_impact_comparison(self, filename: str):
        """Compare RAG impact on both accuracy and extraction success"""
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        rag_comparison = self.df.groupby(['model_name', 'has_rag']).agg({
            'triage_accuracy': 'mean',
            'extraction_success_rate': 'mean'
        }).reset_index()
        
        # Calculate RAG benefit for each model
        models = rag_comparison['model_name'].unique()
        accuracy_benefits = []
        extraction_benefits = []
        model_labels = []
        
        for model in models:
            no_rag = rag_comparison[(rag_comparison['model_name'] == model) & (rag_comparison['has_rag'] == False)]
            with_rag = rag_comparison[(rag_comparison['model_name'] == model) & (rag_comparison['has_rag'] == True)]
            
            if not no_rag.empty and not with_rag.empty:
                acc_benefit = with_rag['triage_accuracy'].iloc[0] - no_rag['triage_accuracy'].iloc[0]
                ext_benefit = with_rag['extraction_success_rate'].iloc[0] - no_rag['extraction_success_rate'].iloc[0]
                
                accuracy_benefits.append(acc_benefit)
                extraction_benefits.append(ext_benefit)
                model_labels.append(model.replace('SmolLM2-', 'S').replace('Gemma-', 'G'))
        
        x = np.arange(len(model_labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, accuracy_benefits, width, label='Accuracy Benefit', 
                      alpha=0.8, color='lightgreen')
        bars2 = ax.bar(x + width/2, extraction_benefits, width, label='Extraction Success Benefit', 
                      alpha=0.8, color='lightblue')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (0.005 if height >= 0 else -0.015),
                       f'{height:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                       fontweight='bold', fontsize=10)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('RAG Benefit (Δ)')
        ax.set_title('RAG Benefits: Accuracy vs Extraction Success\n(Positive = RAG Helps, Negative = RAG Hurts)')
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=45, ha='right')
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_class_performance_breakdown(self, filename: str):
        """Show performance breakdown by medical triage class"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate average F1 scores by class
        class_data = []
        class_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']  # Red, Teal, Blue for ED, GP, HOME
        
        for i, class_name in enumerate(['ed', 'gp', 'home']):
            f1_col = f'{class_name}_f1'
            if f1_col in self.df.columns:
                f1_scores = self.df[f1_col].dropna()
                if not f1_scores.empty:
                    avg_f1 = f1_scores.mean()
                    std_f1 = f1_scores.std()
                    class_data.append({
                        'class': class_name.upper(),
                        'avg_f1': avg_f1,
                        'std_f1': std_f1,
                        'color': class_colors[i]
                    })
        
        if class_data:
            class_df = pd.DataFrame(class_data)
            
            bars = ax.bar(class_df['class'], class_df['avg_f1'], 
                         color=class_df['color'], alpha=0.8, 
                         yerr=class_df['std_f1'], capsize=10, edgecolor='black', linewidth=1)
            
            # Add value labels on bars
            for bar, avg, std in zip(bars, class_df['avg_f1'], class_df['std_f1']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                       f'{avg:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
            
            ax.set_xlabel('Medical Triage Class')
            ax.set_ylabel('Average F1 Score')
            ax.set_title('Class-Specific Performance Breakdown\n(Average F1 Scores with Standard Deviation)')
            ax.set_ylim(0, max(class_df['avg_f1'] + class_df['std_f1']) * 1.2)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add medical context annotations
            annotations = {
                'ED': 'Emergency\nDepartment',
                'GP': 'General\nPractitioner',
                'HOME': 'Home\nCare'
            }
            
            for i, (bar, class_name) in enumerate(zip(bars, class_df['class'])):
                ax.text(bar.get_x() + bar.get_width()/2, -0.1,
                       annotations[class_name], ha='center', va='top', 
                       style='italic', fontsize=10, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_adapter_effectiveness(self, filename: str):
        """Compare effectiveness of different safety adapter types"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        adapter_performance = self.df.groupby('adapter_type').agg({
            'triage_accuracy': ['mean', 'std'],
            'extraction_success_rate': ['mean', 'std'],
            'f2_score': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        adapter_performance.columns = ['adapter_type', 'acc_mean', 'acc_std', 
                                     'ext_mean', 'ext_std', 'f2_mean', 'f2_std']
        
        # Create grouped bar plot
        x = np.arange(len(adapter_performance))
        width = 0.25
        
        bars1 = ax.bar(x - width, adapter_performance['acc_mean'], width,
                      yerr=adapter_performance['acc_std'], label='Accuracy', 
                      alpha=0.8, capsize=5, color='lightcoral')
        bars2 = ax.bar(x, adapter_performance['ext_mean'], width,
                      yerr=adapter_performance['ext_std'], label='Extraction Success', 
                      alpha=0.8, capsize=5, color='lightblue')
        bars3 = ax.bar(x + width, adapter_performance['f2_mean'], width,
                      yerr=adapter_performance['f2_std'], label='F2 Score (Medical Priority)', 
                      alpha=0.8, capsize=5, color='lightgreen')
        
        ax.set_xlabel('Safety Adapter Type')
        ax.set_ylabel('Performance Score')
        ax.set_title('Safety Adapter Effectiveness Comparison\n(Higher = Better Performance)')
        ax.set_xticks(x)
        ax.set_xticklabels(adapter_performance['adapter_type'], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_size_impact(self, filename: str):
        """Analyze impact of model size on performance - Line plot with average and top config"""

        fig, ax = plt.subplots(figsize=(14, 8))

        # Calculate average performance per model size
        size_performance = self.df.groupby('model_size').agg({
            'triage_accuracy': ['mean', 'std', 'max'],
            'extraction_success_rate': ['mean', 'std'],
            'avg_inference_time': 'mean'
        }).reset_index()

        # Flatten column names
        size_performance.columns = ['model_size', 'acc_mean', 'acc_std', 'acc_max',
                                   'ext_mean', 'ext_std', 'time_mean']

        # Sort by model size for proper line plotting
        # Extract numeric part for sorting (135M, 270M, 360M)
        size_performance['size_num'] = size_performance['model_size'].str.extract(r'(\d+)M')[0].astype(int)
        size_performance['quant'] = size_performance['model_size'].str.extract(r'(\d+)bit')[0].astype(int)
        size_performance = size_performance.sort_values(['size_num', 'quant']).reset_index(drop=True)

        x = np.arange(len(size_performance))

        # Plot lines
        line1 = ax.plot(x, size_performance['acc_mean'],
                       marker='o', linewidth=3, markersize=12,
                       label='Average Accuracy', color='#e74c3c',
                       markerfacecolor='white', markeredgewidth=2, markeredgecolor='#e74c3c')

        line2 = ax.plot(x, size_performance['acc_max'],
                       marker='s', linewidth=3, markersize=12,
                       label='Top Config Accuracy', color='#2ecc71',
                       markerfacecolor='white', markeredgewidth=2, markeredgecolor='#2ecc71')

        # Add error bars for average accuracy
        ax.errorbar(x, size_performance['acc_mean'], yerr=size_performance['acc_std'],
                   fmt='none', ecolor='#e74c3c', alpha=0.3, capsize=5, capthick=2)

        # Add value labels
        for i, row in size_performance.iterrows():
            # Average accuracy label
            ax.text(i, row['acc_mean'] + 0.04, f"{row['acc_mean']:.1%}",
                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                   color='#c0392b', bbox=dict(boxstyle='round,pad=0.3',
                   facecolor='white', edgecolor='#e74c3c', alpha=0.7))

            # Top config accuracy label
            ax.text(i, row['acc_max'] + 0.04, f"{row['acc_max']:.1%}",
                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                   color='#27ae60', bbox=dict(boxstyle='round,pad=0.3',
                   facecolor='white', edgecolor='#2ecc71', alpha=0.7))

        # Styling
        ax.set_xlabel('Model Size', fontsize=13, fontweight='bold')
        ax.set_ylabel('Triage Accuracy', fontsize=13, fontweight='bold')
        ax.set_title('Model Size Impact: Average vs Best Configuration Performance\n(Comparing all configurations per model size)',
                    fontsize=15, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(size_performance['model_size'], rotation=45, ha='right', fontsize=11)
        ax.legend(fontsize=12, loc='best', framealpha=0.9)
        ax.set_ylim(0, max(size_performance['acc_max']) + 0.15)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add gap indicator between average and top
        for i, row in size_performance.iterrows():
            gap = row['acc_max'] - row['acc_mean']
            ax.annotate('', xy=(i, row['acc_max']), xytext=(i, row['acc_mean']),
                       arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5, alpha=0.5))
            ax.text(i + 0.15, (row['acc_mean'] + row['acc_max']) / 2,
                   f"Δ{gap:.1%}", fontsize=9, color='gray', style='italic')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_speed_tradeoff(self, filename: str):
        """Show performance vs speed trade-off with bubble sizes"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create bubble plot where bubble size represents reliability score
        bubble_sizes = self.df['reliability_score'] * 300  # Scale for visibility
        
        scatter = ax.scatter(self.df['avg_inference_time'], 
                           self.df['triage_accuracy'],
                           s=bubble_sizes, 
                           c=self.df['unknown_triage_count'],
                           cmap='RdYlBu_r', alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Average Inference Time (seconds/case)')
        ax.set_ylabel('Triage Accuracy')
        ax.set_title('Performance vs Speed Trade-off\n(Bubble size = Reliability Score, Color = Unknown Cases Count)')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, label='Unknown Cases Count')
        
        # Add reliability score legend
        legend_sizes = [0.3, 0.6, 0.9]
        legend_labels = ['Low Reliability', 'Medium Reliability', 'High Reliability']
        legend_handles = []
        
        for size, label in zip(legend_sizes, legend_labels):
            legend_handles.append(plt.scatter([], [], s=size*300, c='gray', alpha=0.6, edgecolors='black'))
        
        ax.legend(legend_handles, legend_labels, title='Reliability Score', 
                 loc='upper right', bbox_to_anchor=(1, 1))
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_top_configurations(self, filename: str):
        """Show top 10 performing configurations"""
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Get top 10 by reliability score
        top_configs = self.df.nlargest(10, 'reliability_score')[
            ['test_name', 'reliability_score', 'triage_accuracy', 'extraction_success_rate', 'unknown_triage_count']
        ].copy()
        
        # Shorten test names for display
        top_configs['short_name'] = top_configs['test_name'].apply(
            lambda x: x.replace('_FineTuned_adapter_safe_triage_', '_')
                      .replace('SmolLM2-', 'S')
                      .replace('Gemma-', 'G')
                      .replace('_4bit_', '_4b_')
                      .replace('_8bit_', '_8b_')
                      .replace('_NoRAG', '_No')
                      .replace('_RAG_top', '_T')
        )
        
        # Create horizontal bar plot
        y_pos = np.arange(len(top_configs))
        bars = ax.barh(y_pos, top_configs['reliability_score'], alpha=0.8, 
                      color='steelblue', edgecolor='black', linewidth=1)
        
        # Color bars by unknown count (red = more unknowns)
        colors = plt.cm.RdYlGn_r(top_configs['unknown_triage_count'] / top_configs['unknown_triage_count'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_configs['short_name'], fontsize=10)
        ax.set_xlabel('Reliability Score (0.7×Accuracy + 0.3×Extraction Success)')
        ax.set_title('Top 10 Most Reliable Configurations\n(Color: Red = More Unknown Cases, Green = Fewer Unknown Cases)')
        
        # Add value labels on bars
        for i, (bar, score, acc, ext, unk) in enumerate(zip(bars, top_configs['reliability_score'],
                                                           top_configs['triage_accuracy'],
                                                           top_configs['extraction_success_rate'],
                                                           top_configs['unknown_triage_count'])):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}\n(A:{acc:.3f}, E:{ext:.3f}, U:{unk})',
                   va='center', fontsize=8, fontweight='bold')
        
        ax.set_xlim(0, max(top_configs['reliability_score']) * 1.3)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_rag_config_effectiveness(self, filename: str):
        """Compare different RAG configurations"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        rag_data = self.df[self.df['has_rag'] == True]
        
        if not rag_data.empty:
            rag_performance = rag_data.groupby('rag_config').agg({
                'triage_accuracy': ['mean', 'std'],
                'extraction_success_rate': ['mean', 'std'],
                'rag_time': 'mean'
            }).reset_index()
            
            # Flatten columns
            rag_performance.columns = ['rag_config', 'acc_mean', 'acc_std', 
                                     'ext_mean', 'ext_std', 'rag_time_mean']
            
            # Create grouped bar plot
            x = np.arange(len(rag_performance))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, rag_performance['acc_mean'], width,
                          yerr=rag_performance['acc_std'], label='Accuracy', 
                          alpha=0.8, capsize=5, color='lightcoral')
            bars2 = ax.bar(x + width/2, rag_performance['ext_mean'], width,
                          yerr=rag_performance['ext_std'], label='Extraction Success', 
                          alpha=0.8, capsize=5, color='lightblue')
            
            ax.set_xlabel('RAG Configuration')
            ax.set_ylabel('Performance')
            ax.set_title('RAG Configuration Effectiveness\n(Structured vs Contextual vs Pure RAG)')
            ax.set_xticks(x)
            ax.set_xticklabels([name.replace('top', 'T').replace('_', '\n') 
                               for name in rag_performance['rag_config']], fontsize=10)
            ax.legend()
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No RAG data available', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=16)
            ax.set_title('RAG Configuration Effectiveness')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_failure_analysis_heatmap(self, filename: str):
        """Create heatmap of unknown cases by model and adapter"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create failure analysis matrix
        failure_data = self.df.pivot_table(
            values='unknown_triage_count',
            index='adapter_type',
            columns='model_family',
            aggfunc='mean'
        )
        
        # Create heatmap
        sns.heatmap(failure_data, annot=True, fmt='.1f', cmap='Reds', 
                   ax=ax, cbar_kws={'label': 'Average Unknown Cases'})
        ax.set_title('Extraction Failure Analysis\n(Average Unknown Cases by Model Family & Adapter Type)')
        ax.set_xlabel('Model Family')
        ax.set_ylabel('Safety Adapter Type')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_accuracy_distribution(self, filename: str):
        """Plot distribution of accuracy scores across all configurations"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram of accuracy scores
        ax1.hist(self.df['triage_accuracy'], bins=20, alpha=0.8, color='steelblue', 
                edgecolor='black', linewidth=1)
        ax1.axvline(self.df['triage_accuracy'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {self.df["triage_accuracy"].mean():.3f}')
        ax1.axvline(self.df['triage_accuracy'].median(), color='orange', linestyle='--', 
                   linewidth=2, label=f'Median: {self.df["triage_accuracy"].median():.3f}')
        ax1.set_xlabel('Triage Accuracy')
        ax1.set_ylabel('Number of Configurations')
        ax1.set_title('Distribution of Triage Accuracy Scores\n(Honest Metrics - No GP Inflation)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot by model family
        model_families = self.df['model_family'].unique()
        accuracy_by_family = [self.df[self.df['model_family'] == family]['triage_accuracy'] 
                             for family in model_families]
        
        box_plot = ax2.boxplot(accuracy_by_family, labels=model_families, patch_artist=True)
        
        # Color the boxes
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        ax2.set_xlabel('Model Family')
        ax2.set_ylabel('Triage Accuracy')
        ax2.set_title('Accuracy Distribution by Model Family\n(Box Plot with Quartiles)')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate Individual Medical Triage Evaluation Plots")
    parser.add_argument("progress_file", 
                       help="Path to evaluation progress JSON file")
    
    args = parser.parse_args()
    
    try:
        generator = IndividualPlotsGenerator(args.progress_file)
        generator.generate_all_plots()
        
        print(f"\n✅ Individual plots complete! Check the '{generator.output_dir}' directory for 12 separate high-quality plots:")
        print("  📊 01_accuracy_vs_unknown_scatter.png - Accuracy vs Unknown cases")
        print("  📊 02_extraction_success_rates.png - Success rates by model/RAG")
        print("  📊 03_model_reliability_heatmap.png - Reliability comparison")
        print("  📊 04_rag_impact_comparison.png - RAG benefits analysis")
        print("  📊 05_class_performance_breakdown.png - ED/GP/HOME performance")
        print("  📊 06_adapter_effectiveness.png - Safety adapter comparison")
        print("  📊 07_model_size_impact.png - Size vs performance (FIXED x-axis)")
        print("  📊 08_performance_speed_tradeoff.png - Speed vs accuracy")
        print("  📊 09_top_configurations.png - Best 10 configurations")
        print("  📊 10_rag_config_effectiveness.png - RAG type comparison")
        print("  📊 11_failure_analysis_heatmap.png - Failure pattern analysis")
        print("  📊 12_accuracy_distribution.png - Overall accuracy distribution")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()