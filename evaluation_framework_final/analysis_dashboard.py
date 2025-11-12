#!/usr/bin/env python3
"""
Comprehensive Medical Triage Evaluation Results Analysis Dashboard
Analyzes and visualizes results from the comprehensive evaluation framework.
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

class ComprehensiveEvaluationDashboard:
    """Dashboard for analyzing comprehensive medical triage evaluation results"""
    
    def __init__(self, progress_file: str):
        self.progress_file = Path(progress_file)
        self.results_data = self._load_progress_results()
        self.df = self._create_dataframe()
        
        # Set style for better plots
        plt.style.use('default')
        sns.set_palette("husl")
        
    def _load_progress_results(self) -> List[Dict[str, Any]]:
        """Load results from progress file"""
        if not self.progress_file.exists():
            raise FileNotFoundError(f"Progress file not found: {self.progress_file}")
        
        print(f"📊 Loading results from: {self.progress_file}")
        
        with open(self.progress_file, 'r') as f:
            progress_data = json.load(f)
        
        results = progress_data.get('results', [])
        completed = progress_data.get('completed_configs', 0)
        total = progress_data.get('total_configs', 120)
        
        print(f"✅ Loaded {len(results)} results ({completed}/{total} completed)")
        return results
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis"""
        data = []
        
        for result in self.results_data:
            config = result['config']
            
            # Parse configuration details
            model_name = config['model_name']
            test_name = config['test_name']
            
            # Extract model components
            model_family, model_size, quantization = self._parse_model_name(model_name)
            
            # Extract adapter info
            adapter_path = config.get('adapter_path')
            if adapter_path:
                adapter_type = self._extract_adapter_type(adapter_path)
                is_finetuned = True
            else:
                adapter_type = "None"
                is_finetuned = False
            
            # Extract RAG info
            rag_config = config.get('rag_config')
            if rag_config:
                rag_name = rag_config['name']
                rag_chunking = rag_config['chunking_method']
                rag_retrieval = rag_config['retrieval_type']
                rag_bias = rag_config['bias_config']
                rag_pass_at_5 = rag_config.get('pass_at_5', 0)
                has_rag = True
            else:
                rag_name = "None"
                rag_chunking = "None"
                rag_retrieval = "None"
                rag_bias = "None"
                rag_pass_at_5 = 0
                has_rag = False
            
            # Metrics
            row = {
                # Configuration
                'config_name': test_name,
                'model_family': model_family,
                'model_size': model_size,
                'quantization': quantization,
                'adapter_type': adapter_type,
                'is_finetuned': is_finetuned,
                'rag_name': rag_name,
                'rag_chunking': rag_chunking,
                'rag_retrieval': rag_retrieval,
                'rag_bias': rag_bias,
                'rag_pass_at_5': rag_pass_at_5,
                'has_rag': has_rag,
                
                # Performance metrics
                'triage_accuracy': result['triage_accuracy'],
                'f1_score': result['f1_score'],
                'f2_score': result['f2_score'],
                'cases_evaluated': result['cases_evaluated'],
                'success_count': result['success_count'],
                'error_count': result['error_count'],
                
                # Timing metrics
                'total_inference_time': result['total_inference_time'],
                'avg_inference_time': result['avg_inference_time_per_case'],
                'rag_retrieval_time': result.get('rag_retrieval_time', 0),
                'rag_context_length': result.get('rag_context_length_avg', 0),
                
                # Configuration type for grouping
                'config_type': self._categorize_config(is_finetuned, has_rag)
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        print(f"📊 Created DataFrame with {len(df)} configurations")
        return df
    
    def _parse_model_name(self, model_name: str) -> tuple:
        """Parse model name into components"""
        parts = model_name.split('_')
        if len(parts) >= 2:
            model_family = parts[0]  # SmolLM2, Gemma
            model_size = parts[1]    # 135M, 360M, 270M
            quantization = parts[2] if len(parts) > 2 else "unknown"
        else:
            model_family = model_name
            model_size = "unknown"
            quantization = "unknown"
        
        return model_family, model_size, quantization
    
    def _extract_adapter_type(self, adapter_path: str) -> str:
        """Extract adapter type from path"""
        if not adapter_path:
            return "None"
        
        path = Path(adapter_path).name
        if "balanced_safe" in path:
            return "balanced_safe"
        elif "high_capacity_safe" in path:
            return "high_capacity_safe"
        elif "performance_safe" in path:
            return "performance_safe"
        elif "ultra_safe" in path:
            return "ultra_safe"
        else:
            return "unknown_adapter"
    
    def _categorize_config(self, is_finetuned: bool, has_rag: bool) -> str:
        """Categorize configuration type"""
        if not is_finetuned and not has_rag:
            return "Base Model"
        elif not is_finetuned and has_rag:
            return "Base + RAG"
        elif is_finetuned and not has_rag:
            return "Fine-tuned"
        else:
            return "Fine-tuned + RAG"
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE EVALUATION RESULTS SUMMARY")
        print("="*80)
        
        # Overall statistics
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"   Total Configurations: {len(self.df)}")
        print(f"   Average Accuracy: {self.df['triage_accuracy'].mean():.3f}")
        print(f"   Average F1 Score: {self.df['f1_score'].mean():.3f}")
        print(f"   Average F2 Score: {self.df['f2_score'].mean():.3f}")
        print(f"   Average Inference Time: {self.df['avg_inference_time'].mean():.3f}s/case")
        
        # Top performers
        print(f"\n🏆 TOP 5 PERFORMERS BY ACCURACY:")
        top_accuracy = self.df.nlargest(5, 'triage_accuracy')[['config_name', 'triage_accuracy', 'f2_score', 'avg_inference_time']]
        for i, (_, row) in enumerate(top_accuracy.iterrows(), 1):
            print(f"   {i}. {row['config_name']}")
            print(f"      Accuracy: {row['triage_accuracy']:.3f}, F2: {row['f2_score']:.3f}, Time: {row['avg_inference_time']:.3f}s")
        
        print(f"\n🎯 TOP 5 PERFORMERS BY F2 SCORE:")
        top_f2 = self.df.nlargest(5, 'f2_score')[['config_name', 'triage_accuracy', 'f2_score', 'avg_inference_time']]
        for i, (_, row) in enumerate(top_f2.iterrows(), 1):
            print(f"   {i}. {row['config_name']}")
            print(f"      F2: {row['f2_score']:.3f}, Accuracy: {row['triage_accuracy']:.3f}, Time: {row['avg_inference_time']:.3f}s")
        
        # Configuration type comparison
        print(f"\n🔍 PERFORMANCE BY CONFIGURATION TYPE:")
        config_stats = self.df.groupby('config_type').agg({
            'triage_accuracy': ['mean', 'std', 'count'],
            'f2_score': ['mean', 'std'],
            'avg_inference_time': ['mean', 'std']
        }).round(3)
        print(config_stats)
        
        # Model family comparison
        print(f"\n🤖 PERFORMANCE BY MODEL FAMILY:")
        model_stats = self.df.groupby('model_family').agg({
            'triage_accuracy': ['mean', 'std', 'count'],
            'f2_score': ['mean', 'std'],
            'avg_inference_time': ['mean', 'std']
        }).round(3)
        print(model_stats)
        
        # RAG impact analysis
        if self.df['has_rag'].any():
            print(f"\n🔍 RAG IMPACT ANALYSIS:")
            rag_comparison = self.df.groupby('has_rag')[['triage_accuracy', 'f2_score', 'avg_inference_time']].mean().round(3)
            print(rag_comparison)
            
            print(f"\n📊 RAG CONFIGURATION PERFORMANCE:")
            rag_stats = self.df[self.df['has_rag']].groupby('rag_name').agg({
                'triage_accuracy': ['mean', 'std', 'count'],
                'f2_score': ['mean', 'std'],
                'rag_pass_at_5': 'first'
            }).round(3)
            print(rag_stats)
    
    def create_visualizations(self, output_dir: str = "analysis_plots"):
        """Create comprehensive visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n📊 Generating visualizations in: {output_path}")
        
        # 1. Overall performance distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy distribution
        axes[0,0].hist(self.df['triage_accuracy'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_title('Distribution of Triage Accuracy')
        axes[0,0].set_xlabel('Accuracy')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].axvline(self.df['triage_accuracy'].mean(), color='red', linestyle='--', label=f'Mean: {self.df["triage_accuracy"].mean():.3f}')
        axes[0,0].legend()
        
        # F2 Score distribution
        axes[0,1].hist(self.df['f2_score'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0,1].set_title('Distribution of F2 Score')
        axes[0,1].set_xlabel('F2 Score')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].axvline(self.df['f2_score'].mean(), color='red', linestyle='--', label=f'Mean: {self.df["f2_score"].mean():.3f}')
        axes[0,1].legend()
        
        # Inference time distribution
        axes[1,0].hist(self.df['avg_inference_time'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1,0].set_title('Distribution of Inference Time')
        axes[1,0].set_xlabel('Avg Inference Time (s)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].axvline(self.df['avg_inference_time'].mean(), color='red', linestyle='--', label=f'Mean: {self.df["avg_inference_time"].mean():.3f}s')
        axes[1,0].legend()
        
        # Accuracy vs F2 Score scatter
        axes[1,1].scatter(self.df['triage_accuracy'], self.df['f2_score'], alpha=0.6, color='purple')
        axes[1,1].set_title('Accuracy vs F2 Score')
        axes[1,1].set_xlabel('Triage Accuracy')
        axes[1,1].set_ylabel('F2 Score')
        
        # Add correlation
        correlation = self.df['triage_accuracy'].corr(self.df['f2_score'])
        axes[1,1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=axes[1,1].transAxes, 
                      bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path / 'overall_performance_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Configuration type comparison
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        sns.boxplot(data=self.df, x='config_type', y='triage_accuracy')
        plt.title('Accuracy by Configuration Type')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 3, 2)
        sns.boxplot(data=self.df, x='config_type', y='f2_score')
        plt.title('F2 Score by Configuration Type')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 3, 3)
        sns.boxplot(data=self.df, x='config_type', y='avg_inference_time')
        plt.title('Inference Time by Configuration Type')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 3, 4)
        sns.boxplot(data=self.df, x='model_family', y='triage_accuracy')
        plt.title('Accuracy by Model Family')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 3, 5)
        sns.boxplot(data=self.df, x='model_family', y='f2_score')
        plt.title('F2 Score by Model Family')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 3, 6)
        sns.boxplot(data=self.df, x='quantization', y='triage_accuracy')
        plt.title('Accuracy by Quantization')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'configuration_comparisons.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. RAG Analysis (if RAG configurations exist)
        if self.df['has_rag'].any():
            plt.figure(figsize=(15, 10))
            
            # RAG vs No RAG comparison
            plt.subplot(2, 3, 1)
            rag_comparison = self.df.groupby('has_rag')[['triage_accuracy', 'f2_score']].mean()
            rag_comparison.plot(kind='bar', ax=plt.gca())
            plt.title('RAG vs No RAG Performance')
            plt.xticks([0, 1], ['No RAG', 'With RAG'], rotation=0)
            plt.legend()
            
            # RAG configuration performance
            rag_df = self.df[self.df['has_rag']]
            if len(rag_df) > 0:
                plt.subplot(2, 3, 2)
                sns.boxplot(data=rag_df, x='rag_name', y='triage_accuracy')
                plt.title('Accuracy by RAG Configuration')
                plt.xticks(rotation=45)
                
                plt.subplot(2, 3, 3)
                sns.boxplot(data=rag_df, x='rag_name', y='f2_score')
                plt.title('F2 Score by RAG Configuration')
                plt.xticks(rotation=45)
                
                plt.subplot(2, 3, 4)
                sns.scatterplot(data=rag_df, x='rag_pass_at_5', y='triage_accuracy', hue='rag_name')
                plt.title('RAG Pass@5 vs Accuracy')
                
                plt.subplot(2, 3, 5)
                sns.boxplot(data=rag_df, x='rag_retrieval', y='triage_accuracy')
                plt.title('Accuracy by RAG Retrieval Type')
                plt.xticks(rotation=45)
                
                plt.subplot(2, 3, 6)
                if 'rag_context_length' in rag_df.columns and rag_df['rag_context_length'].sum() > 0:
                    sns.scatterplot(data=rag_df, x='rag_context_length', y='triage_accuracy')
                    plt.title('Context Length vs Accuracy')
                else:
                    plt.text(0.5, 0.5, 'RAG Context Length\nData Not Available', 
                           ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title('Context Length vs Accuracy')
            
            plt.tight_layout()
            plt.savefig(output_path / 'rag_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 4. Adapter Analysis (if fine-tuned models exist)
        finetuned_df = self.df[self.df['is_finetuned']]
        if len(finetuned_df) > 0:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            sns.boxplot(data=finetuned_df, x='adapter_type', y='triage_accuracy')
            plt.title('Accuracy by Adapter Type')
            plt.xticks(rotation=45)
            
            plt.subplot(2, 2, 2)
            sns.boxplot(data=finetuned_df, x='adapter_type', y='f2_score')
            plt.title('F2 Score by Adapter Type')
            plt.xticks(rotation=45)
            
            plt.subplot(2, 2, 3)
            sns.boxplot(data=finetuned_df, x='adapter_type', y='avg_inference_time')
            plt.title('Inference Time by Adapter Type')
            plt.xticks(rotation=45)
            
            plt.subplot(2, 2, 4)
            # Adapter performance vs base models
            base_vs_adapter = []
            for model in finetuned_df['model_family'].unique():
                base_acc = self.df[(self.df['model_family'] == model) & (~self.df['is_finetuned'])]['triage_accuracy'].mean()
                adapter_acc = finetuned_df[finetuned_df['model_family'] == model]['triage_accuracy'].mean()
                base_vs_adapter.append({'Model': model, 'Base': base_acc, 'Fine-tuned': adapter_acc})
            
            if base_vs_adapter:
                comparison_df = pd.DataFrame(base_vs_adapter)
                comparison_df.set_index('Model')[['Base', 'Fine-tuned']].plot(kind='bar', ax=plt.gca())
                plt.title('Base vs Fine-tuned Model Performance')
                plt.xticks(rotation=45)
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(output_path / 'adapter_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 5. Performance vs Efficiency Trade-off
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(self.df['avg_inference_time'], self.df['triage_accuracy'], 
                             c=self.df['f2_score'], cmap='viridis', alpha=0.7, s=60)
        plt.xlabel('Avg Inference Time (s)')
        plt.ylabel('Triage Accuracy')
        plt.title('Accuracy vs Inference Time (colored by F2 Score)')
        plt.colorbar(scatter, label='F2 Score')
        
        plt.subplot(2, 2, 2)
        sns.scatterplot(data=self.df, x='avg_inference_time', y='f2_score', hue='config_type', alpha=0.7)
        plt.title('F2 Score vs Inference Time by Config Type')
        
        plt.subplot(2, 2, 3)
        # Model size analysis (if we have different sizes)
        if len(self.df['model_size'].unique()) > 1:
            sns.boxplot(data=self.df, x='model_size', y='triage_accuracy')
            plt.title('Accuracy by Model Size')
        else:
            plt.text(0.5, 0.5, 'Single Model Size\nAnalysis Not Applicable', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Accuracy by Model Size')
        
        plt.subplot(2, 2, 4)
        # Efficiency frontier
        pareto_frontier = []
        sorted_df = self.df.sort_values('avg_inference_time')
        max_acc = 0
        for _, row in sorted_df.iterrows():
            if row['triage_accuracy'] > max_acc:
                max_acc = row['triage_accuracy']
                pareto_frontier.append(row)
        
        if pareto_frontier:
            frontier_df = pd.DataFrame(pareto_frontier)
            plt.scatter(self.df['avg_inference_time'], self.df['triage_accuracy'], alpha=0.3, color='gray', label='All configs')
            plt.plot(frontier_df['avg_inference_time'], frontier_df['triage_accuracy'], 'ro-', label='Pareto frontier')
            plt.xlabel('Avg Inference Time (s)')
            plt.ylabel('Triage Accuracy')
            plt.title('Efficiency Frontier')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ All visualizations saved to: {output_path}")
    
    def export_results(self, output_file: str = "comprehensive_results_analysis.csv"):
        """Export processed results to CSV"""
        self.df.to_csv(output_file, index=False)
        print(f"📊 Results exported to: {output_file}")
        
        # Also create a summary statistics file
        summary_file = output_file.replace('.csv', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("COMPREHENSIVE EVALUATION RESULTS SUMMARY\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Total Configurations: {len(self.df)}\n")
            f.write(f"Average Accuracy: {self.df['triage_accuracy'].mean():.3f} ± {self.df['triage_accuracy'].std():.3f}\n")
            f.write(f"Average F1 Score: {self.df['f1_score'].mean():.3f} ± {self.df['f1_score'].std():.3f}\n")
            f.write(f"Average F2 Score: {self.df['f2_score'].mean():.3f} ± {self.df['f2_score'].std():.3f}\n")
            f.write(f"Average Inference Time: {self.df['avg_inference_time'].mean():.3f} ± {self.df['avg_inference_time'].std():.3f}s\n\n")
            
            f.write("TOP 10 CONFIGURATIONS BY ACCURACY:\n")
            f.write("-" * 40 + "\n")
            top_configs = self.df.nlargest(10, 'triage_accuracy')
            for i, (_, row) in enumerate(top_configs.iterrows(), 1):
                f.write(f"{i:2d}. {row['config_name']}\n")
                f.write(f"    Accuracy: {row['triage_accuracy']:.3f}, F2: {row['f2_score']:.3f}, Time: {row['avg_inference_time']:.3f}s\n")
        
        print(f"📊 Summary exported to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze comprehensive medical triage evaluation results')
    parser.add_argument('progress_file', help='Path to evaluation progress JSON file')
    parser.add_argument('--output-dir', default='analysis_plots', help='Directory for output plots')
    parser.add_argument('--export-csv', default='comprehensive_results_analysis.csv', help='CSV export filename')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    
    args = parser.parse_args()
    
    try:
        # Initialize dashboard
        dashboard = ComprehensiveEvaluationDashboard(args.progress_file)
        
        # Generate summary report
        dashboard.generate_summary_report()
        
        # Export results
        dashboard.export_results(args.export_csv)
        
        # Create visualizations (unless disabled)
        if not args.no_plots:
            dashboard.create_visualizations(args.output_dir)
        
        print(f"\n🎉 Analysis complete!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()