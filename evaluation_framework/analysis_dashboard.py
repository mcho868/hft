#!/usr/bin/env python3
"""
Evaluation Results Analysis Dashboard
Displays best combinations and generates visualizations from optimized evaluation results.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Any

class EvaluationAnalysisDashboard:
    """Dashboard for analyzing optimized evaluation results"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results_data = self._load_all_results()
        
        # Model parameter sizes (in millions)
        self.model_params = {
            "SmolLM2-135M": 135,
            "SmolLM2-360M": 360,
            "Gemma-270M": 270
        }
        
    def _load_all_results(self) -> List[Dict[str, Any]]:
        """Load all evaluation results from the results directory"""
        results = []
        
        # Try different possible result file locations
        possible_files = [
            self.results_dir / "results" / "enhanced_evaluation_results.json",
            self.results_dir / "detailed_results.json",
            self.results_dir / "enhanced_evaluation_results.jsonl"
        ]
        
        for file_path in possible_files:
            if file_path.exists():
                print(f"📊 Loading results from: {file_path}")
                try:
                    if file_path.suffix == '.jsonl':
                        # Load JSONL format
                        with open(file_path, 'r') as f:
                            for line in f:
                                if line.strip():
                                    results.append(json.loads(line))
                    else:
                        # Load JSON format
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                results.extend(data)
                            else:
                                results.append(data)
                    break
                except Exception as e:
                    print(f"⚠️  Error loading {file_path}: {e}")
                    continue
        
        if not results:
            # Try loading from all JSON files in results directory
            results_files = list(self.results_dir.glob("**/*.json"))
            for file_path in results_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list) and data and 'combo_id' in str(data[0]):
                            results.extend(data)
                            print(f"📊 Loaded results from: {file_path}")
                            break
                except:
                    continue
        
        print(f"✅ Loaded {len(results)} evaluation results")
        return results
    
    def _extract_model_info(self, result: Dict[str, Any]) -> Dict[str, str]:
        """Extract model information from result data including quantization"""
        combo_id = result.get('combo_id', '')
        model_name = "Unknown"
        adapter_type = "standard"
        quantization = "unknown"
        
        # Try to get adapter path from the configuration matrix
        # Load the configuration matrix to get adapter_path
        try:
            config_file = self.results_dir / "configurations" / "optimized_evaluation_matrix.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    matrix_data = json.load(f)
                
                # Find matching combo_id
                for combo in matrix_data:
                    if combo.get('combo_id') == combo_id:
                        adapter_path = combo.get('adapter_config', {}).get('adapter_path', '')
                        
                        # Extract quantization from adapter path
                        if "_4bit_" in adapter_path:
                            quantization = "4bit"
                        elif "_8bit_" in adapter_path:
                            quantization = "8bit"
                        
                        # Extract model name from adapter path
                        if "SmolLM2-360M" in adapter_path:
                            model_name = f"SmolLM2-360M-{quantization}"
                        elif "SmolLM2-135M" in adapter_path:
                            model_name = f"SmolLM2-135M-{quantization}"
                        elif "Gemma-270M" in adapter_path:
                            model_name = f"Gemma-270M-{quantization}"
                        
                        # Check for safety adapters
                        if "safety_triage_adapters" in adapter_path:
                            adapter_type = "safety_enhanced"
                        
                        break
        except Exception as e:
            print(f"⚠️  Could not load configuration matrix: {e}")
        
        # Fallback: Extract base model from combo_id if adapter path method failed
        if model_name == "Unknown":
            if "S360" in combo_id:
                model_name = "SmolLM2-360M"
            elif "S135" in combo_id:
                model_name = "SmolLM2-135M"
            elif "G270" in combo_id:
                model_name = "Gemma-270M"
        
        # Get base parameter size
        base_model_name = model_name.split('-')[0] + "-" + model_name.split('-')[1] if '-' in model_name else model_name
        param_size = self.model_params.get(base_model_name, 0)
        
        return {
            "model_name": model_name,
            "base_model": base_model_name,
            "quantization": quantization,
            "adapter_type": adapter_type,
            "param_size": param_size
        }
    
    def create_performance_tables(self):
        """Create and display performance tables"""
        if not self.results_data:
            print("❌ No results data available")
            return
        
        # Convert to DataFrame
        df_data = []
        for result in self.results_data:
            if result.get('error_message'):
                continue
                
            model_info = self._extract_model_info(result)
            
            row = {
                'combo_id': result.get('combo_id', ''),
                'model_name': model_info['model_name'],
                'base_model': model_info['base_model'],
                'quantization': model_info['quantization'],
                'adapter_type': model_info['adapter_type'],
                'param_size': model_info['param_size'],
                'triage_accuracy': result.get('triage_accuracy', 0),
                'f1_score': result.get('f1_score', 0),
                'f2_score': result.get('f2_score', 0),
                'next_step_quality': result.get('next_step_quality', 0),
                'memory_usage_mb': result.get('memory_usage_mb', 0),
                'inference_speed_tps': result.get('inference_speed_tps', 0),
                'latency_ms': result.get('latency_ms', 0),
                'llm_judge_used': result.get('llm_judge_used', False)
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        if df.empty:
            print("❌ No valid results to analyze")
            return
        
        print("=" * 100)
        print("🚀 EVALUATION RESULTS ANALYSIS DASHBOARD")
        print("=" * 100)
        
        # Table 1: Best Memory/Inference Speed Combinations
        print("\n📈 TABLE 1: BEST MEMORY & INFERENCE SPEED COMBINATIONS")
        print("-" * 80)
        
        # Sort by efficiency (lower memory, higher speed)
        df['efficiency_score'] = (df['inference_speed_tps'] / (df['memory_usage_mb'] + 1))
        speed_table = df.nlargest(10, 'efficiency_score')[
            ['combo_id', 'model_name', 'adapter_type', 'param_size', 
             'memory_usage_mb', 'inference_speed_tps', 'latency_ms', 'efficiency_score']
        ]
        
        print(speed_table.to_string(index=False, float_format='%.3f'))
        
        # Table 2: Best Triage Accuracy + Next Step Quality
        print("\n🎯 TABLE 2: BEST TRIAGE ACCURACY & NEXT STEP QUALITY COMBINATIONS")
        print("-" * 80)
        
        # Create combined quality score
        df['quality_score'] = (df['triage_accuracy'] * 0.6 + 
                              (df['next_step_quality'] / 10) * 0.4)
        
        quality_table = df.nlargest(10, 'quality_score')[
            ['combo_id', 'model_name', 'adapter_type', 'param_size',
             'triage_accuracy', 'f2_score', 'next_step_quality', 'quality_score', 'llm_judge_used']
        ]
        
        print(quality_table.to_string(index=False, float_format='%.3f'))
        
        # Summary Statistics
        print("\n📊 SUMMARY STATISTICS")
        print("-" * 50)
        print(f"Total Configurations: {len(df)}")
        print(f"Average Triage Accuracy: {df['triage_accuracy'].mean():.3f}")
        print(f"Average F2 Score: {df['f2_score'].mean():.3f}")
        print(f"Average Memory Usage: {df['memory_usage_mb'].mean():.1f} MB")
        print(f"Average Inference Speed: {df['inference_speed_tps'].mean():.1f} tokens/sec")
        
        # Best overall combination
        best_overall = df.loc[df['quality_score'].idxmax()]
        print(f"\n🏆 BEST OVERALL COMBINATION:")
        print(f"   {best_overall['combo_id']}")
        print(f"   Model: {best_overall['model_name']} ({best_overall['param_size']}M params)")
        print(f"   Accuracy: {best_overall['triage_accuracy']:.3f}")
        print(f"   F2 Score: {best_overall['f2_score']:.3f}")
        print(f"   Next Step Quality: {best_overall['next_step_quality']:.1f}/10")
        print(f"   Memory: {best_overall['memory_usage_mb']:.1f} MB")
        print(f"   Speed: {best_overall['inference_speed_tps']:.1f} tokens/sec")
        
        return df
    
    def create_visualizations(self, df):
        """Create separate visualization plots as individual PNG files"""
        if df is None or df.empty:
            print("❌ No data available for visualization")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create output directory
        output_dir = self.results_dir / "analysis_plots"
        output_dir.mkdir(exist_ok=True)
        
        plot_files = []
        
        # Plot 1: Model Parameter Size vs Triage Accuracy
        print("📊 Creating Plot 1: Model Size vs Triage Accuracy...")
        plt.figure(figsize=(12, 8))
        model_accuracy = df.groupby(['model_name', 'param_size'])['triage_accuracy'].agg(['mean', 'std']).reset_index()
        
        plt.errorbar(model_accuracy['param_size'], model_accuracy['mean'], 
                    yerr=model_accuracy['std'], marker='o', capsize=5, capthick=2, markersize=10)
        
        # Add model names and accuracy values
        for i, row in model_accuracy.iterrows():
            plt.annotate(f"{row['model_name']}\n{row['mean']:.3f} ± {row['std']:.3f}", 
                        (row['param_size'], row['mean']), 
                        xytext=(10, 15), textcoords='offset points', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.xlabel('Model Parameters (Millions)')
        plt.ylabel('Triage Accuracy')
        plt.title('Model Size vs Triage Accuracy (with Standard Deviation)')
        plt.grid(True, alpha=0.3)
        plot1_file = output_dir / "1_model_size_vs_accuracy.png"
        plt.savefig(plot1_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot1_file)
        
        # Plot 2: Inference Speed by Model and Quantization
        print("📊 Creating Plot 2: Inference Speed by Model...")
        plt.figure(figsize=(12, 6))
        model_perf = df.groupby('model_name').agg({
            'inference_speed_tps': 'mean',
            'memory_usage_mb': 'mean'
        }).reset_index()
        
        x_pos = np.arange(len(model_perf))
        bars1 = plt.bar(x_pos, model_perf['inference_speed_tps'], alpha=0.7, color='skyblue')
        
        plt.xlabel('Model (with Quantization)')
        plt.ylabel('Inference Speed (tokens/sec)')
        plt.title('Inference Speed by Model and Quantization')
        plt.xticks(x_pos, model_perf['model_name'], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            plt.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        
        plot2_file = output_dir / "2_inference_speed_by_model.png"
        plt.savefig(plot2_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot2_file)

        # Plot 3: Memory Usage by Model and Quantization
        print("📊 Creating Plot 3: Memory Usage by Model...")
        plt.figure(figsize=(12, 6))
        bars2 = plt.bar(x_pos, model_perf['memory_usage_mb'], alpha=0.7, color='lightcoral')
        
        plt.xlabel('Model (with Quantization)')
        plt.ylabel('Memory Change (MB)')
        plt.title('Memory Usage Change by Model and Quantization')
        plt.xticks(x_pos, model_perf['model_name'], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)  # Add zero line
        
        for bar in bars2:
            height = bar.get_height()
            plt.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3) if height >= 0 else (0, -15), textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
        
        plot3_file = output_dir / "3_memory_usage_by_model.png"
        plt.savefig(plot3_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot3_file)
        
        # Plot 4: Adapter Type Comparison
        print("📊 Creating Plot 4: Adapter Type Comparison...")
        plt.figure(figsize=(12, 8))
        adapter_comparison = df.groupby('adapter_type').agg({
            'triage_accuracy': 'mean',
            'f2_score': 'mean',
            'next_step_quality': 'mean'
        }).reset_index()
        
        x_pos = np.arange(len(adapter_comparison))
        width = 0.25
        
        bars1 = plt.bar(x_pos - width, adapter_comparison['triage_accuracy'], width, 
                       label='Triage Accuracy', alpha=0.8, color='skyblue')
        bars2 = plt.bar(x_pos, adapter_comparison['f2_score'], width,
                       label='F2 Score', alpha=0.8, color='lightgreen')
        bars3 = plt.bar(x_pos + width, adapter_comparison['next_step_quality']/10, width,
                       label='Next Step Quality (/10)', alpha=0.8, color='salmon')
        
        # Add value labels on bars
        for bars, values in [(bars1, adapter_comparison['triage_accuracy']), 
                            (bars2, adapter_comparison['f2_score']), 
                            (bars3, adapter_comparison['next_step_quality']/10)]:
            for bar, value in zip(bars, values):
                height = bar.get_height()
                plt.annotate(f'{value:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 5), textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xlabel('Adapter Type')
        plt.ylabel('Score')
        plt.title('Standard vs Safety-Enhanced Adapters Performance Comparison')
        plt.xticks(x_pos, adapter_comparison['adapter_type'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, max(adapter_comparison['triage_accuracy'].max(), 
                        adapter_comparison['f2_score'].max()) * 1.15)
        
        plot4_file = output_dir / "4_adapter_type_comparison.png"
        plt.savefig(plot4_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot4_file)
        
        # Plot 5: Accuracy vs Efficiency Scatter
        print("📊 Creating Plot 5: Accuracy vs Efficiency Scatter...")
        plt.figure(figsize=(12, 8))
        efficiency_score = df['inference_speed_tps'] / (df['memory_usage_mb'] + 1)
        scatter = plt.scatter(df['triage_accuracy'], efficiency_score, 
                             c=df['param_size'], s=df['f2_score']*100, 
                             alpha=0.6, cmap='viridis')
        
        plt.xlabel('Triage Accuracy')
        plt.ylabel('Efficiency (Speed/Memory)')
        plt.title('Accuracy vs Efficiency (Bubble Size = F2 Score, Color = Parameter Count)')
        
        # Add colorbar with better positioning
        cbar = plt.colorbar(scatter, shrink=0.8)
        cbar.set_label('Parameter Count (M)', rotation=270, labelpad=20)
        
        # Add best performer annotation with better positioning
        best_idx = df['quality_score'].idxmax()
        best_row = df.loc[best_idx]
        best_efficiency = efficiency_score.loc[best_idx]
        
        # Position annotation to avoid colorbar overlap
        plt.annotate(f'Best Overall:\n{best_row["model_name"]}\nAcc: {best_row["triage_accuracy"]:.3f}\nEff: {best_efficiency:.1f}', 
                    xy=(best_row['triage_accuracy'], best_efficiency),
                    xytext=(-80, 30), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'),
                    fontsize=9, ha='center')
        
        plt.tight_layout()
        plot5_file = output_dir / "5_accuracy_vs_efficiency_scatter.png"
        plt.savefig(plot5_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot5_file)
        
        # Plot 6: F2 Score Distribution
        print("📊 Creating Plot 6: F2 Score Distribution...")
        plt.figure(figsize=(10, 6))
        model_f2 = df.groupby('model_name')['f2_score'].mean().reset_index()
        model_f2 = model_f2.sort_values('f2_score', ascending=True)
        
        bars = plt.barh(model_f2['model_name'], model_f2['f2_score'], alpha=0.7, color='green')
        plt.xlabel('F2 Score (Recall-Weighted)')
        plt.ylabel('Model')
        plt.title('F2 Score by Model (Recall-Prioritized for Medical Safety)')
        plt.grid(True, alpha=0.3)
        
        for bar in bars:
            width = bar.get_width()
            plt.annotate(f'{width:.3f}',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0), textcoords="offset points",
                        ha='left', va='center', fontsize=8)
        
        plot6_file = output_dir / "6_f2_score_distribution.png"
        plt.savefig(plot6_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot6_file)
        
        # Plot 7: Top 5 Performers Comparison with Full Configuration Details
        print("📊 Creating Plot 7: Top 5 Performers Comparison...")
        
        # Load configuration matrix to get detailed config info
        config_data = {}
        try:
            config_file = self.results_dir / "configurations" / "optimized_evaluation_matrix.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    matrix_data = json.load(f)
                    for combo in matrix_data:
                        config_data[combo['combo_id']] = combo
        except Exception as e:
            print(f"⚠️  Could not load detailed config data: {e}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Top 5 by F2 Score
        top_f2 = df.nlargest(5, 'f2_score').reset_index(drop=True)
        
        # Create detailed configuration labels
        def create_config_label(combo_id, model_name, quantization, adapter_type):
            config_info = config_data.get(combo_id, {})
            rag_config = config_info.get('rag_config', {})
            
            # Extract chunk info from combo_id
            chunks = "5" if "_C5_" in combo_id else "10" if "_C10_" in combo_id else "?"
            
            # Extract method info and shorten for display
            chunking = rag_config.get('chunking_method', 'unknown')
            # Shorten long method names
            if len(chunking) > 20:
                if 'structured_agent' in chunking:
                    chunking = 'structured_agent...'
                elif 'contextual_sentence' in chunking:
                    chunking = 'contextual_sent...'
                else:
                    chunking = chunking[:15] + "..."
            
            retrieval = rag_config.get('retrieval_type', 'unknown')
            
            # Safety indicator (using text instead of problematic emojis)
            safety_indicator = "[SAFE]" if adapter_type == "safety_enhanced" else "[STD]"
            
            return f"{model_name}\n{safety_indicator} {chunking}\n{retrieval} | {chunks}ch"
        
        top_f2['config_label'] = [create_config_label(row['combo_id'], row['model_name'], 
                                                     row['quantization'], row['adapter_type']) 
                                 for _, row in top_f2.iterrows()]
        
        bars1 = ax1.barh(top_f2['config_label'], top_f2['f2_score'], 
                        color=['gold', 'silver', '#CD7F32', 'lightblue', 'lightgreen'])
        
        # Add value labels
        for i, (bar, value, accuracy) in enumerate(zip(bars1, top_f2['f2_score'], top_f2['triage_accuracy'])):
            width = bar.get_width()
            ax1.annotate(f'F2: {value:.3f}\nAcc: {accuracy:.3f}',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0), textcoords="offset points",
                        ha='left', va='center', fontsize=9, fontweight='bold')
        
        ax1.set_xlabel('F2 Score (Recall-Weighted)')
        ax1.set_ylabel('Configuration Details')
        ax1.set_title('Top 5 by F2 Score (Medical Safety Priority)\n[SAFE]=Safety-Enhanced [STD]=Standard | ch=chunks')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, max(top_f2['f2_score']) * 1.25)
        
        # Top 5 by Triage Accuracy
        top_acc = df.nlargest(5, 'triage_accuracy').reset_index(drop=True)
        
        top_acc['config_label'] = [create_config_label(row['combo_id'], row['model_name'], 
                                                      row['quantization'], row['adapter_type']) 
                                  for _, row in top_acc.iterrows()]
        
        bars2 = ax2.barh(top_acc['config_label'], top_acc['triage_accuracy'], 
                        color=['gold', 'silver', '#CD7F32', 'lightblue', 'lightgreen'])
        
        # Add value labels
        for i, (bar, accuracy, f2) in enumerate(zip(bars2, top_acc['triage_accuracy'], top_acc['f2_score'])):
            width = bar.get_width()
            ax2.annotate(f'Acc: {accuracy:.3f}\nF2: {f2:.3f}',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0), textcoords="offset points",
                        ha='left', va='center', fontsize=9, fontweight='bold')
        
        ax2.set_xlabel('Triage Accuracy')
        ax2.set_ylabel('Configuration Details')
        ax2.set_title('Top 5 by Triage Accuracy\n[SAFE]=Safety-Enhanced [STD]=Standard | ch=chunks')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, max(top_acc['triage_accuracy']) * 1.15)
        
        plt.tight_layout()
        plot7_file = output_dir / "7_top_5_performers_comparison.png"
        plt.savefig(plot7_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot7_file)
        
        print(f"\n📊 All plots saved to: {output_dir}")
        for i, plot_file in enumerate(plot_files, 1):
            print(f"   {i}. {plot_file.name}")
        
        return plot_files
    
    def generate_detailed_report(self, df):
        """Generate detailed analysis report"""
        if df is None or df.empty:
            return
            
        report_file = self.results_dir / "detailed_analysis_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("MEDICAL TRIAGE MODEL EVALUATION - DETAILED ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            # Model performance summary
            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write("-" * 30 + "\n")
            for model in df['model_name'].unique():
                model_data = df[df['model_name'] == model]
                f.write(f"\n{model}:\n")
                f.write(f"  Average Accuracy: {model_data['triage_accuracy'].mean():.3f} ± {model_data['triage_accuracy'].std():.3f}\n")
                f.write(f"  Average F2 Score: {model_data['f2_score'].mean():.3f} ± {model_data['f2_score'].std():.3f}\n")
                f.write(f"  Average Memory: {model_data['memory_usage_mb'].mean():.1f} MB\n")
                f.write(f"  Average Speed: {model_data['inference_speed_tps'].mean():.1f} tokens/sec\n")
            
            # Top performers
            f.write(f"\nTOP 5 OVERALL PERFORMERS\n")
            f.write("-" * 25 + "\n")
            top_5 = df.nlargest(5, 'quality_score')
            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                f.write(f"{i}. {row['combo_id']}\n")
                f.write(f"   Model: {row['model_name']} ({row['param_size']}M)\n")
                f.write(f"   Accuracy: {row['triage_accuracy']:.3f}, Quality: {row['quality_score']:.3f}\n\n")
        
        print(f"📄 Detailed report saved to: {report_file}")
    
    def create_top_performers_table(self, df):
        """Create and display top 5 performers table with detailed configuration info"""
        if df is None or df.empty:
            print("❌ No data available for top performers table")
            return
        
        # Load configuration matrix to get detailed config info
        config_data = {}
        try:
            config_file = self.results_dir / "configurations" / "optimized_evaluation_matrix.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    matrix_data = json.load(f)
                    for combo in matrix_data:
                        config_data[combo['combo_id']] = combo
        except Exception as e:
            print(f"⚠️  Could not load detailed config data: {e}")
        
        print("\n" + "🏆" * 80)
        print("TOP 5 MODEL CONFIGURATIONS - RANKED BY PERFORMANCE")
        print("🏆" * 80)
        
        # Top 5 by F2 Score (Medical Safety Priority)
        print("\n🎯 TOP 5 BY F2 SCORE (Medical Safety Priority)")
        print("-" * 110)
        
        top_f2 = df.nlargest(5, 'f2_score')[
            ['combo_id', 'model_name', 'quantization', 'adapter_type', 
             'f2_score', 'triage_accuracy', 'next_step_quality', 'inference_speed_tps', 'memory_usage_mb']
        ].copy()
        
        for i, (_, row) in enumerate(top_f2.iterrows(), 1):
            combo_id = row['combo_id']
            config_info = config_data.get(combo_id, {})
            rag_config = config_info.get('rag_config', {})
            
            # Extract configuration details
            chunks = "5" if "_C5_" in combo_id else "10" if "_C10_" in combo_id else "unknown"
            chunking_method = rag_config.get('chunking_method', 'unknown')
            retrieval_type = rag_config.get('retrieval_type', 'unknown')
            safety_indicator = "[SAFE] Safety-Enhanced" if row['adapter_type'] == "safety_enhanced" else "[STD] Standard"
            
            print(f"{i}. {combo_id}")
            print(f"   Model: {row['model_name']} ({row['quantization']})")
            print(f"   {safety_indicator}")
            print(f"   Chunking: {chunking_method} | Retrieval: {retrieval_type} | Chunks: {chunks}")
            print(f"   F2 Score: {row['f2_score']:.3f} | Accuracy: {row['triage_accuracy']:.3f}")
            print(f"   Speed: {row['inference_speed_tps']:.1f} tps | Memory: {row['memory_usage_mb']:.1f} MB")
            print(f"   Next Step Quality: {row['next_step_quality']:.1f}/10")
            print()
        
        # Top 5 by Triage Accuracy
        print("🎯 TOP 5 BY TRIAGE ACCURACY")
        print("-" * 110)
        
        top_accuracy = df.nlargest(5, 'triage_accuracy')[
            ['combo_id', 'model_name', 'quantization', 'adapter_type', 
             'triage_accuracy', 'f2_score', 'next_step_quality', 'inference_speed_tps', 'memory_usage_mb']
        ].copy()
        
        for i, (_, row) in enumerate(top_accuracy.iterrows(), 1):
            combo_id = row['combo_id']
            config_info = config_data.get(combo_id, {})
            rag_config = config_info.get('rag_config', {})
            
            # Extract configuration details
            chunks = "5" if "_C5_" in combo_id else "10" if "_C10_" in combo_id else "unknown"
            chunking_method = rag_config.get('chunking_method', 'unknown')
            retrieval_type = rag_config.get('retrieval_type', 'unknown')
            safety_indicator = "[SAFE] Safety-Enhanced" if row['adapter_type'] == "safety_enhanced" else "[STD] Standard"
            
            print(f"{i}. {combo_id}")
            print(f"   🤖 Model: {row['model_name']} ({row['quantization']})")
            print(f"   {safety_indicator}")
            print(f"   📊 Chunking: {chunking_method} | Retrieval: {retrieval_type} | Chunks: {chunks}")
            print(f"   Accuracy: {row['triage_accuracy']:.3f} | F2 Score: {row['f2_score']:.3f}")
            print(f"   ⚡ Speed: {row['inference_speed_tps']:.1f} tps | Memory: {row['memory_usage_mb']:.1f} MB")
            print(f"   🏥 Next Step Quality: {row['next_step_quality']:.1f}/10")
            print()
        
        return top_f2, top_accuracy

def main():
    """Main function to run the analysis dashboard"""
    results_dir = "/Users/choemanseung/789/hft/evaluation_framework/optimized_evaluation_session_20250928_002822"
    
    print("🚀 Starting Evaluation Analysis Dashboard")
    print("=" * 50)
    
    # Initialize dashboard
    dashboard = EvaluationAnalysisDashboard(results_dir)
    
    # Create performance tables
    df = dashboard.create_performance_tables()
    
    if df is not None and not df.empty:
        # Create top performers table
        dashboard.create_top_performers_table(df)
        
        # Create visualizations
        dashboard.create_visualizations(df)
        
        # Generate detailed report
        dashboard.generate_detailed_report(df)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()