#!/usr/bin/env python3
"""
Top 10 Configurations Full Test Set Evaluation
Tests the best performing configurations from optimized evaluation on the complete test set (1,975 cases)
with LLM-as-judge clinical appropriateness evaluation.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse

import numpy as np

# Import enhanced evaluation components
from enhanced_evaluation_pipeline import EnhancedEvaluationPipeline
from appropriateness_judge import ClinicalAppropriatenessJudge
from optimized_config_generator import EvaluationCombo, AdapterConfig

class Top10FullTestEvaluationRunner:
    """Evaluation runner for top 10 configurations on full test set"""
    
    def __init__(self, base_dir: str = "/Users/choemanseung/789/hft", llm_judge_workers: int = 30):
        self.base_dir = Path(base_dir)
        self.llm_judge_workers = llm_judge_workers
        
        # Pre-selected top 10 configuration IDs (from optimized evaluation)
        self.top_10_combo_ids = [
            "G270_1_C5_3a482fea",
            "G270_1_C10_c1b77866", 
            "G270_2_C5_112fedce",
            "G270_2_C10_ff0ad1e7",
            "G270_3_C5_69179247",
            "G270_3_C10_94a82773",
            "G270_4_C5_3b1cbe85",
            "G270_4_C10_8e1f8d80",
            "G270_2_C5_d4a5684e",
            "G270_2_C10_4e5485d2"
        ]
        
        # Load optimized configurations
        self.optimized_config_file = (
            self.base_dir / "evaluation_framework" / 
            "optimized_evaluation_session_20250928_002822" / 
            "configurations" / "optimized_evaluation_matrix.json"
        )
        
        # Initialize enhanced evaluation pipeline with LLM judge and parallel workers
        self.evaluation_pipeline = EnhancedEvaluationPipeline(
            str(self.base_dir),
            enable_clinical_judge=True,  # Enable LLM-as-judge
            skip_models=False,
            llm_judge_workers=self.llm_judge_workers  # Configurable parallel workers
        )
        
        # Set full test dataset size
        os.environ["EVAL_SUBSET_SIZE"] = "1975"  # Use full test set
        os.environ["CLINICAL_EVAL_LIMIT"] = "1975"  # Use full test set for clinical evaluation too
        
        # Session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def load_top_configurations(self) -> List[EvaluationCombo]:
        """Load the top 10 configurations from optimized evaluation"""
        if not self.optimized_config_file.exists():
            raise FileNotFoundError(f"Optimized config file not found: {self.optimized_config_file}")
        
        with open(self.optimized_config_file, 'r') as f:
            all_configs = json.load(f)
        
        # Filter to top 10 configurations and convert to EvaluationCombo objects
        top_configs = []
        for config_dict in all_configs:
            if config_dict["combo_id"] in self.top_10_combo_ids:
                # Create AdapterConfig object
                adapter_config = AdapterConfig(
                    adapter_path=config_dict["adapter_config"]["adapter_path"],
                    model_name=config_dict["adapter_config"]["model_name"],
                    adapter_type=config_dict["adapter_config"]["adapter_type"],
                    training_config=config_dict["adapter_config"].get("base_config", {}),
                    **{k: v for k, v in config_dict["adapter_config"].items() 
                       if k not in ["adapter_path", "model_name", "adapter_type", "base_config"]}
                )
                
                # Create EvaluationCombo object
                eval_combo = EvaluationCombo(
                    combo_id=config_dict["combo_id"],
                    rag_config=config_dict["rag_config"],
                    adapter_config=adapter_config
                )
                
                top_configs.append(eval_combo)
        
        print(f"📊 Loaded {len(top_configs)} top configurations out of {len(all_configs)} total")
        
        # Print configuration details
        print("\n🏆 TOP 10 CONFIGURATIONS TO EVALUATE:")
        for i, config in enumerate(top_configs, 1):
            adapter_type = config.adapter_config.adapter_type
            model_name = config.adapter_config.model_name
            chunking = config.rag_config["chunking_method"]
            retrieval = config.rag_config["retrieval_type"]
            chunk_limit = config.rag_config["chunk_limit"]
            
            print(f"   {i}. {config.combo_id}")
            print(f"      Model: {model_name} ({adapter_type})")
            print(f"      RAG: {chunking} + {retrieval} ({chunk_limit} chunks)")
        
        return top_configs
    
    def setup_session_directory(self):
        """Create directory structure for this evaluation session"""
        session_dir = self.base_dir / "evaluation_framework" / f"top10_full_test_session_{self.session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (session_dir / "configurations").mkdir(exist_ok=True)
        (session_dir / "results").mkdir(exist_ok=True)
        (session_dir / "analysis").mkdir(exist_ok=True)
        
        return session_dir
    
    def run_full_test_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation on top 10 configurations with full test set"""
        print("🚀 TOP 10 CONFIGURATIONS - FULL TEST SET EVALUATION")
        print("Using Complete Test Dataset (1,975 cases) with LLM-as-Judge")
        print("="*80)
        
        # Setup session directory
        session_dir = self.setup_session_directory()
        print(f"📁 Session directory: {session_dir}")
        
        # Load top configurations
        top_configs = self.load_top_configurations()
        
        if len(top_configs) != 10:
            print(f"⚠️  Warning: Found {len(top_configs)} configurations instead of 10")
        
        # Save configurations to session (convert to dict format for JSON)
        config_file = session_dir / "configurations" / "top10_evaluation_matrix.json"
        config_dicts = []
        for config in top_configs:
            config_dict = {
                "combo_id": config.combo_id,
                "rag_config": config.rag_config,
                "adapter_config": {
                    "adapter_path": config.adapter_config.adapter_path,
                    "model_name": config.adapter_config.model_name,
                    "adapter_type": config.adapter_config.adapter_type,
                    "training_config": config.adapter_config.training_config
                }
            }
            config_dicts.append(config_dict)
        
        with open(config_file, 'w') as f:
            json.dump(config_dicts, f, indent=2)
        
        print(f"\n🔬 EVALUATION SETUP:")
        print(f"   Configurations: {len(top_configs)}")
        print(f"   Test cases: 1,975 (full test set)")
        print(f"   LLM Judge: Enabled (TinfoilAgent)")
        print(f"   Parallel Workers: {self.llm_judge_workers}")
        print(f"   Total evaluations: {len(top_configs) * 1975:,}")
        print(f"   Estimated time: {len(top_configs) * 0.5:.1f}-{len(top_configs) * 2:.1f} hours (with parallelization)")
        
        # Run enhanced evaluation
        start_time = time.time()
        
        print(f"\n📊 Starting enhanced evaluation at {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            # Run evaluation on top configurations with full test set
            results = self.evaluation_pipeline.run_enhanced_evaluation(
                top_configs,
                resume=True  # Allow resuming if interrupted
            )
            
            # Save results
            results_file = session_dir / "results" / "top10_full_test_results.json"
            
            # Convert results to JSON format
            results_data = []
            for result in results:
                if hasattr(result, '__dict__'):
                    results_data.append(asdict(result))
                else:
                    results_data.append(result)
            
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            # Calculate summary
            valid_results = [r for r in results if r.error_message is None]
            failed_results = [r for r in results if r.error_message is not None]
            
            end_time = time.time()
            duration_hours = (end_time - start_time) / 3600
            
            summary = {
                "session_id": self.session_id,
                "session_dir": str(session_dir),
                "total_configurations": len(top_configs),
                "total_results": len(results),
                "successful_evaluations": len(valid_results),
                "failed_evaluations": len(failed_results),
                "test_cases_per_config": 1975,
                "total_test_evaluations": len(valid_results) * 1975,
                "duration_hours": duration_hours,
                "start_time": datetime.fromtimestamp(start_time).isoformat(),
                "end_time": datetime.fromtimestamp(end_time).isoformat()
            }
            
            # Generate detailed analysis
            analysis = self._generate_detailed_analysis(valid_results)
            analysis_file = session_dir / "analysis" / "top10_analysis_report.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            # Print results summary
            self._print_evaluation_summary(summary, valid_results)
            
            # Save summary
            summary_file = session_dir / "evaluation_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n📁 RESULTS SAVED:")
            print(f"   Full results: {results_file}")
            print(f"   Analysis: {analysis_file}")
            print(f"   Summary: {summary_file}")
            
            return {
                "summary": summary,
                "results": results,
                "valid_results": valid_results,
                "failed_results": failed_results,
                "analysis": analysis,
                "session_dir": str(session_dir)
            }
            
        except Exception as e:
            print(f"\n❌ EVALUATION FAILED: {e}")
            
            # Save error info
            error_file = session_dir / "evaluation_error.json"
            with open(error_file, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "session_id": self.session_id
                }, f, indent=2)
            
            raise
    
    def _generate_detailed_analysis(self, results: List[Any]) -> Dict[str, Any]:
        """Generate detailed analysis of evaluation results"""
        if not results:
            return {"error": "No valid results to analyze"}
        
        # Calculate statistics
        accuracies = [r.triage_accuracy for r in results]
        f1_scores = [r.f1_score for r in results]
        f2_scores = [r.f2_score for r in results]
        clinical_scores = [r.next_step_quality for r in results]
        inference_speeds = [r.inference_speed_tps for r in results]
        memory_usage = [r.memory_usage_mb for r in results]
        
        # Rank configurations
        ranked_by_f2 = sorted(results, key=lambda x: x.f2_score, reverse=True)
        ranked_by_accuracy = sorted(results, key=lambda x: x.triage_accuracy, reverse=True)
        ranked_by_clinical = sorted(results, key=lambda x: x.next_step_quality, reverse=True)
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_type": "top10_full_test_with_llm_judge",
            "total_configurations": len(results),
            "test_cases_per_config": 1975,
            
            "overall_statistics": {
                "triage_accuracy": {
                    "mean": float(np.mean(accuracies)),
                    "std": float(np.std(accuracies)),
                    "min": float(np.min(accuracies)),
                    "max": float(np.max(accuracies))
                },
                "f1_score": {
                    "mean": float(np.mean(f1_scores)),
                    "std": float(np.std(f1_scores)),
                    "min": float(np.min(f1_scores)),
                    "max": float(np.max(f1_scores))
                },
                "f2_score": {
                    "mean": float(np.mean(f2_scores)),
                    "std": float(np.std(f2_scores)),
                    "min": float(np.min(f2_scores)),
                    "max": float(np.max(f2_scores))
                },
                "clinical_appropriateness": {
                    "mean": float(np.mean(clinical_scores)),
                    "std": float(np.std(clinical_scores)),
                    "min": float(np.min(clinical_scores)),
                    "max": float(np.max(clinical_scores))
                },
                "inference_speed": {
                    "mean": float(np.mean(inference_speeds)),
                    "std": float(np.std(inference_speeds)),
                    "min": float(np.min(inference_speeds)),
                    "max": float(np.max(inference_speeds))
                }
            },
            
            "rankings": {
                "by_f2_score": [
                    {
                        "combo_id": r.combo_id,
                        "f2_score": r.f2_score,
                        "triage_accuracy": r.triage_accuracy,
                        "clinical_score": r.next_step_quality
                    }
                    for r in ranked_by_f2[:5]  # Top 5
                ],
                "by_triage_accuracy": [
                    {
                        "combo_id": r.combo_id,
                        "triage_accuracy": r.triage_accuracy,
                        "f2_score": r.f2_score,
                        "clinical_score": r.next_step_quality
                    }
                    for r in ranked_by_accuracy[:5]  # Top 5
                ],
                "by_clinical_appropriateness": [
                    {
                        "combo_id": r.combo_id,
                        "clinical_score": r.next_step_quality,
                        "triage_accuracy": r.triage_accuracy,
                        "f2_score": r.f2_score
                    }
                    for r in ranked_by_clinical[:5]  # Top 5
                ]
            },
            
            "best_overall": {
                "combo_id": ranked_by_f2[0].combo_id,
                "triage_accuracy": ranked_by_f2[0].triage_accuracy,
                "f2_score": ranked_by_f2[0].f2_score,
                "f1_score": ranked_by_f2[0].f1_score,
                "clinical_score": ranked_by_f2[0].next_step_quality,
                "inference_speed": ranked_by_f2[0].inference_speed_tps,
                "memory_usage": ranked_by_f2[0].memory_usage_mb
            }
        }
        
        return analysis
    
    def _print_evaluation_summary(self, summary: Dict[str, Any], results: List[Any]):
        """Print comprehensive evaluation summary"""
        print("\n" + "="*80)
        print("TOP 10 CONFIGURATIONS - FULL TEST EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\n📊 EVALUATION OVERVIEW:")
        print(f"   Configurations Tested: {summary['total_configurations']}")
        print(f"   Test Cases per Config: {summary['test_cases_per_config']:,}")
        print(f"   Total Evaluations: {summary['total_test_evaluations']:,}")
        print(f"   Success Rate: {summary['successful_evaluations']}/{summary['total_results']}")
        print(f"   Duration: {summary['duration_hours']:.1f} hours")
        
        if results:
            # Overall performance
            accuracies = [r.triage_accuracy for r in results]
            f2_scores = [r.f2_score for r in results]
            clinical_scores = [r.next_step_quality for r in results]
            
            print(f"\n🎯 OVERALL PERFORMANCE (Full Test Set):")
            print(f"   Average Triage Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
            print(f"   Average F2 Score: {np.mean(f2_scores):.3f} ± {np.std(f2_scores):.3f}")
            print(f"   Average Clinical Score: {np.mean(clinical_scores):.3f} ± {np.std(clinical_scores):.3f}")
            
            # Best performers
            best_f2 = max(results, key=lambda x: x.f2_score)
            best_accuracy = max(results, key=lambda x: x.triage_accuracy)
            best_clinical = max(results, key=lambda x: x.next_step_quality)
            
            print(f"\n🏆 BEST PERFORMERS:")
            print(f"   Best F2 Score: {best_f2.combo_id} ({best_f2.f2_score:.3f})")
            print(f"   Best Accuracy: {best_accuracy.combo_id} ({best_accuracy.triage_accuracy:.3f})")
            print(f"   Best Clinical: {best_clinical.combo_id} ({best_clinical.next_step_quality:.3f})")
            
            # Performance ranges
            print(f"\n📈 PERFORMANCE RANGES:")
            print(f"   Triage Accuracy: {min(accuracies):.3f} - {max(accuracies):.3f}")
            print(f"   F2 Score: {min(f2_scores):.3f} - {max(f2_scores):.3f}")
            print(f"   Clinical Score: {min(clinical_scores):.3f} - {max(clinical_scores):.3f}")

def main():
    """Main execution for top 10 full test evaluation"""
    parser = argparse.ArgumentParser(description="Top 10 Configurations - Full Test Set Evaluation")
    
    parser.add_argument("--base-dir", default="/Users/choemanseung/789/hft",
                       help="Base directory for evaluation")
    parser.add_argument("--resume", action="store_true",
                       help="Resume evaluation if interrupted")
    parser.add_argument("--llm-judge-workers", type=int, default=20,
                       help="Number of parallel workers for LLM judge API calls (default: 20)")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = Top10FullTestEvaluationRunner(args.base_dir, args.llm_judge_workers)
    
    try:
        print("🔬 Starting Top 10 Configurations Full Test Evaluation...")
        print(f"   Base directory: {args.base_dir}")
        print(f"   Resume mode: {args.resume}")
        print(f"   LLM Judge Workers: {args.llm_judge_workers}")
        
        results = runner.run_full_test_evaluation()
        
        print("\n✅ Top 10 full test evaluation completed successfully!")
        print(f"📁 Results directory: {results['session_dir']}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Top 10 evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())