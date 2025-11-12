#!/usr/bin/env python3
"""
Fast Optimized Master Evaluation Runner for Medical Triage System
Optimized to avoid reloading RAG systems and models for every configuration.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from datetime import datetime
import argparse
from collections import defaultdict

# Import optimized components
from optimized_config_generator import OptimizedConfigMatrixGenerator
from enhanced_evaluation_pipeline import EnhancedEvaluationPipeline
from appropriateness_judge import ClinicalAppropriatenessJudge
from truly_fast_evaluation_pipeline import TrulyFastEvaluationPipeline, ModelGroup

class FastOptimizedMasterEvaluationRunner:
    """Fast optimized master orchestrator that batches similar configurations"""
    
    def __init__(self, base_dir: str = "/Users/choemanseung/789/hft",
                 config: Optional[Dict[str, Any]] = None):
        self.base_dir = Path(base_dir)
        self.config = config or self._load_default_config()
        
        # Initialize optimized components
        self.matrix_generator = OptimizedConfigMatrixGenerator(str(self.base_dir))
        
        # Initialize evaluation pipeline
        skip_models = self.config.get("evaluation", {}).get("skip_models", False)
        enable_clinical = self.config.get("phases", {}).get("clinical_judgement", True)
        use_validation = self.config.get("evaluation", {}).get("use_validation_set", False)
        validation_size = self.config.get("evaluation", {}).get("validation_subset_size", 200)
        
        # Set environment variables for the pipeline
        if use_validation:
            import os
            os.environ["USE_VALIDATION_SET"] = "true"
            os.environ["EVAL_SUBSET_SIZE"] = str(validation_size)
            os.environ["CLINICAL_EVAL_LIMIT"] = str(validation_size)
        
        # Use the truly fast evaluation pipeline
        self.evaluation_pipeline = TrulyFastEvaluationPipeline(
            str(self.base_dir),
            llm_judge_workers=30
        )
        
        # Execution state
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.execution_log = []
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default evaluation configuration optimized for fast execution"""
        return {
            "phases": {
                "generate_matrix": True,
                "run_evaluation": True,
                "clinical_judgement": True,
                "performance_profiling": True,
                "statistical_analysis": True
            },
            "evaluation": {
                "batch_size": 8,
                "max_workers": 2,
                "resume": True,
                "validation_subset_size": 30,
                "use_validation_set": True
            },
            "clinical_judge": {
                "model": "llama3-3-70b",
                "use_tinfoil": True,
                "rate_limit_delay": 0.5,  # Faster rate limit
                "batch_size": 10
            },
            "performance": {
                "profile_memory": True,
                "benchmark_scalability": True,
                "save_detailed_profiles": True
            },
            "analysis": {
                "top_configurations": 20,
                "generate_visualizations": True,
                "save_comprehensive_report": True
            },
            "output": {
                "session_dir": None,
                "save_intermediate": True,
                "verbose": True
            }
        }
    
    def setup_session_directory(self):
        """Create directory structure for this evaluation session"""
        if self.config.get("session_dir") is None:
            session_dir = self.base_dir / f"fast_optimized_evaluation_session_{self.session_id}"
            self.config["session_dir"] = str(session_dir)
        else:
            session_dir = Path(self.config["session_dir"])
        session_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (session_dir / "configurations").mkdir(exist_ok=True)
        (session_dir / "results").mkdir(exist_ok=True)
        (session_dir / "profiles").mkdir(exist_ok=True)
        (session_dir / "analysis").mkdir(exist_ok=True)
        (session_dir / "visualizations").mkdir(exist_ok=True)
        
        return session_dir
    
    def group_configurations_by_model(self, evaluation_combos: List[Any]) -> List[ModelGroup]:
        """Group configurations by adapter path to minimize reloading"""
        grouped = defaultdict(list)
        
        for combo in evaluation_combos:
            # Group by adapter path since each adapter is a different fine-tuned model
            model_key = combo.adapter_config.adapter_path
            grouped[model_key].append(combo)
        
        print(f"📊 Grouped {len(evaluation_combos)} configurations into {len(grouped)} model groups:")
        for model_key, configs in grouped.items():
            print(f"   {model_key}: {len(configs)} configurations")
        
        # Convert to ModelGroup objects
        model_groups = []
        for model_key, configs in grouped.items():
            # Extract model info from first config in group
            first_config = configs[0]
            model_group = ModelGroup(
                model_name=first_config.adapter_config.model_name,
                adapter_type=first_config.adapter_config.adapter_type,
                adapter_path=first_config.adapter_config.adapter_path,
                configurations=configs
            )
            model_groups.append(model_group)
        
        return model_groups
    
    def run_fast_optimized_evaluation(self) -> Dict[str, Any]:
        """Run the fast optimized evaluation pipeline"""
        print("🚀 MEDICAL TRIAGE SYSTEM - FAST OPTIMIZED COMPREHENSIVE EVALUATION")
        print("Using Pre-Validated RAG Configurations with Batched Processing")
        print(f"Session ID: {self.session_id}")
        print("="*80)
        
        # Setup session directory
        session_dir = self.setup_session_directory()
        print(f"📁 Session directory: {session_dir}")
        
        results = {"session_id": self.session_id, "session_dir": str(session_dir)}
        
        try:
            # Phase 1: Generate optimized configuration matrix
            if self.config["phases"]["generate_matrix"]:
                print("\n" + "="*80)
                print("PHASE 1: OPTIMIZED CONFIGURATION MATRIX GENERATION")
                print("="*80)
                
                start_time = time.time()
                
                print("📊 Loading validated RAG configurations...")
                print(f"   Found {len(self.matrix_generator.validated_rag_configs)} validated RAG configs")
                
                print("🔍 Scanning adapter directories...")
                adapters = self.matrix_generator.scan_adapter_directories()
                print(f"   Found {len(adapters)} trained adapters")
                
                print("🔗 Creating optimized evaluation matrix...")
                evaluation_combos = self.matrix_generator.create_optimized_evaluation_matrix()
                
                # Print optimization impact
                self.matrix_generator.print_optimized_summary(evaluation_combos)
                
                # Save to session directory
                session_matrix_file = session_dir / "configurations" / "optimized_evaluation_matrix.json"
                matrix_file = self.matrix_generator.save_matrix_to_file(evaluation_combos, str(session_matrix_file))
                
                duration = time.time() - start_time
                
                results["phase1"] = {
                    "evaluation_combos": evaluation_combos,
                    "matrix_file": session_matrix_file,
                    "total_combinations": len(evaluation_combos),
                    "duration": duration
                }
                
                print(f"\n🎯 OPTIMIZATION SUCCESS:")
                print(f"   Evaluating {len(evaluation_combos)} combinations")
                print(f"   Time reduction: ~95% compared to exhaustive search")
                
            # Phase 2: Run fast evaluation pipeline
            if self.config["phases"]["run_evaluation"]:
                print("\n" + "="*80)
                print("PHASE 2: FAST BATCHED EVALUATION PIPELINE")
                print("="*80)
                
                start_time = time.time()
                
                # Group configurations by model to minimize reloading
                model_groups = self.group_configurations_by_model(evaluation_combos)
                
                all_results = []
                total_configs = len(evaluation_combos)
                
                print(f"\n🚀 Starting truly batched evaluation:")
                print(f"   Total configurations: {total_configs}")
                print(f"   Model groups: {len(model_groups)}")
                print(f"   Clinical judge workers: 30")
                print(f"   🎯 RAG system will be loaded ONCE")
                print(f"   🎯 Each model will be loaded ONCE per group")
                
                # Process each model group with true reuse
                for group_idx, model_group in enumerate(model_groups, 1):
                    print(f"\n📊 Processing group {group_idx}/{len(model_groups)}: {model_group.model_name}_{model_group.adapter_type}")
                    print(f"   Configurations in group: {len(model_group.configurations)}")
                    
                    group_start_time = time.time()
                    
                    # Evaluate the entire model group at once (reusing loaded components)
                    try:
                        group_results = self.evaluation_pipeline.evaluate_model_group(model_group)
                        all_results.extend(group_results)
                        
                        group_duration = time.time() - group_start_time
                        avg_per_config = group_duration / len(model_group.configurations)
                        
                        print(f"   ✅ Group completed in {group_duration:.1f}s ({avg_per_config:.1f}s per config)")
                        print(f"   Group average performance:")
                        
                        valid_group_results = [r for r in group_results if r.error_message is None]
                        if valid_group_results:
                            avg_acc = np.mean([r.triage_accuracy for r in valid_group_results])
                            avg_f2 = np.mean([r.f2_score for r in valid_group_results])
                            print(f"     Accuracy: {avg_acc:.3f}, F2: {avg_f2:.3f}")
                        
                    except Exception as e:
                        print(f"   ❌ Group failed: {e}")
                        # Add error results for all configs in failed group
                        for config in model_group.configurations:
                            error_result = type('ErrorResult', (), {
                                'combo_id': config.combo_id,
                                'error_message': str(e),
                                'triage_accuracy': 0.0,
                                'f2_score': 0.0
                            })()
                            all_results.append(error_result)
                
                # Add numpy import for calculations
                import numpy as np
                
                # Save results
                results_file = session_dir / "results" / "fast_evaluation_results.json"
                results_data = []
                for result in all_results:
                    if hasattr(result, '__dict__'):
                        results_data.append(asdict(result))
                    else:
                        results_data.append(result)
                
                with open(results_file, 'w') as f:
                    json.dump(results_data, f, indent=2)
                
                duration = time.time() - start_time
                
                # Calculate summary
                valid_results = [r for r in all_results if r.error_message is None]
                summary = {
                    "total_results": len(all_results),
                    "completed": len(valid_results),
                    "failed": len(all_results) - len(valid_results),
                    "duration_hours": duration / 3600
                }
                
                results["phase2"] = {
                    "results": all_results,
                    "results_file": results_file,
                    "summary": summary
                }
                
                print(f"\n✅ FAST EVALUATION COMPLETED:")
                print(f"   Total time: {duration/3600:.1f} hours")
                print(f"   Average per config: {duration/len(all_results):.1f}s")
                print(f"   Success rate: {len(valid_results)}/{len(all_results)}")
                
            # Skip other phases for now to focus on speed
            results["phase3"] = {"skipped": "Focus on fast evaluation"}
            results["phase4"] = {"skipped": "Focus on fast evaluation"}  
            results["phase5"] = {"skipped": "Focus on fast evaluation"}
            
            print(f"\n🎉 FAST OPTIMIZED EVALUATION PIPELINE COMPLETED!")
            print(f"📁 Results saved to: {session_dir}")
            
            return results
            
        except Exception as e:
            print(f"\n❌ FAST EVALUATION PIPELINE FAILED: {e}")
            raise

def main():
    """Main execution with fast optimized pipeline"""
    parser = argparse.ArgumentParser(description="Medical Triage System - Fast Optimized Evaluation Runner")
    
    parser.add_argument("--base-dir", default="/Users/choemanseung/789/hft",
                       help="Base directory for evaluation")
    parser.add_argument("--output-dir", help="Output directory for session results")
    parser.add_argument("--validation-size", type=int, default=30,
                       help="Size of stratified validation sample (default: 30)")
    parser.add_argument("--skip-clinical", action="store_true",
                       help="Skip clinical appropriateness evaluation")
    parser.add_argument("--mock-clinical", action="store_true",
                       help="Use mock clinical evaluation for faster hyperparameter tuning")
    
    args = parser.parse_args()
    
    # Create fast optimized configuration
    config = {
        "phases": {
            "generate_matrix": True,
            "run_evaluation": True,
            "clinical_judgement": not args.skip_clinical,
            "performance_profiling": False,  # Skip for speed
            "statistical_analysis": False   # Skip for speed
        },
        "evaluation": {
            "batch_size": 8,
            "max_workers": 2,
            "resume": True,
            "use_validation_set": True,
            "validation_subset_size": args.validation_size
        },
        "output": {
            "session_dir": args.output_dir,
            "verbose": True
        }
    }
    
    # Initialize and run fast evaluation
    runner = FastOptimizedMasterEvaluationRunner(args.base_dir, config)
    
    try:
        results = runner.run_fast_optimized_evaluation()
        return 0
    except Exception as e:
        print(f"❌ Fast evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())