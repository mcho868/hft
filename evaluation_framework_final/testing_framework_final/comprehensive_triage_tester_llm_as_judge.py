#!/usr/bin/env python3
"""
Final Medical Triage Testing Pipeline - Top 5 Configurations with LLM-as-Judge

This pipeline tests the TOP 5 PERFORMING CONFIGURATIONS on the full test dataset WITH LLM QUALITY EVALUATION:
1. SmolLM2-135M_4bit_high_capacity_safe_NoRAG (68.0% accuracy)
2. SmolLM2-135M_4bit_balanced_safe_NoRAG (59.5% accuracy)
3. SmolLM2-135M_4bit_performance_safe_NoRAG (57.0% accuracy)
4. SmolLM2-135M_4bit_high_capacity_safe_RAG_top1_structured_contextual_diverse (55.0% accuracy)
5. SmolLM2-135M_4bit_high_capacity_safe_RAG_top2_structured_pure_diverse (54.0% accuracy)

Key Features:
- Tests on FULL TEST DATASET (1975 cases)
- Only evaluates top 5 proven configurations
- LLM-as-Judge evaluation for reasoning and next step quality (0-100 scores)
- Comprehensive performance metrics including quality assessments
- Production-ready evaluation pipeline
- F2 score optimization (β=2 emphasizes recall for medical safety)
- External LLM evaluation using Tinfoil API (llama3-3-70b)
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from sklearn.metrics import f1_score, fbeta_score, classification_report, confusion_matrix

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('top5_final_testing_llm_judge.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Reduce logging spam from external modules
logging.getLogger('optimized_hybrid_evaluator').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Add final_retrieval_testing to path for optimized retrieval
final_retrieval_path = Path(__file__).parent.parent.parent / "final_retrieval_testing"
sys.path.insert(0, str(final_retrieval_path))

# MLX imports
try:
    from mlx_lm import load, generate
    import mlx.core as mx
    MLX_AVAILABLE = True
    logger.info("✅ MLX loaded successfully")
except ImportError:
    MLX_AVAILABLE = False
    logger.warning("⚠️  MLX not available - using mock mode")

# Optimized retrieval imports
try:
    from optimized_hybrid_evaluator import OptimizedMultiSourceRetriever, BiasConfig
    RETRIEVAL_AVAILABLE = True
    logger.info("✅ Optimized retrieval system loaded successfully")
except ImportError as e:
    RETRIEVAL_AVAILABLE = False
    logger.warning(f"⚠️  Optimized retrieval not available: {e}")

@dataclass
class EvaluationConfig:
    """Configuration for a single evaluation run"""
    model_name: str
    model_path: str
    adapter_path: Optional[str] = None
    rag_config: Optional[Dict[str, Any]] = None
    test_name: str = ""
    
    def __post_init__(self):
        if not self.test_name:
            rag_suffix = f"_RAG_{self.rag_config['chunking_method']}" if self.rag_config else "_NoRAG"
            adapter_suffix = "_FineTuned" if self.adapter_path else "_Base"
            self.test_name = f"{self.model_name}{adapter_suffix}{rag_suffix}"

@dataclass 
class EvaluationResult:
    """Results from a single evaluation"""
    config: EvaluationConfig
    timestamp: str
    
    # Performance metrics
    triage_accuracy: float
    f1_score: float
    f2_score: float
    
    # Detailed metrics
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]
    
    # Timing and efficiency
    total_inference_time: float
    avg_inference_time_per_case: float
    cases_evaluated: int
    
    # Error tracking
    success_count: int
    error_count: int
    error_details: List[str]
    
    # Triage extraction tracking
    unknown_triage_count: int = 0
    total_failures: int = 0
    
    # RAG-specific metrics (if applicable)
    rag_retrieval_time: Optional[float] = None
    rag_context_length_avg: Optional[float] = None
    
    # LLM-as-Judge metrics (if applicable)
    llm_judge_metrics: Optional[Dict[str, Any]] = None

class Top5MedicalTriageTester:
    """Final testing pipeline for TOP 5 medical triage configurations on full test dataset"""
    
    def __init__(self, base_dir: str = "/Users/choemanseung/789/hft"):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "mlx_models"
        self.adapters_dir = self.base_dir / "safety_triage_adapters"
        self.data_dir = self.base_dir / "Final_dataset"
        
        # Load FULL TEST dataset
        self.test_data = self._load_test_data()
        logger.info(f"📊 Loaded {len(self.test_data)} TEST cases for final evaluation")
        
        # Initialize RAG system if available
        self.retriever = None
        if RETRIEVAL_AVAILABLE:
            self._initialize_retrieval_system()
            # Suppress excessive logging from retrieval system
            import logging
            retrieval_logger = logging.getLogger('optimized_hybrid_evaluator')
            retrieval_logger.setLevel(logging.WARNING)
        
        # Define top RAG configurations from retrieval testing
        self.top_rag_configs = self._define_top_rag_configs()
        
        # Define TOP 5 model configurations only
        self.model_configs = self._define_top_5_model_configs()
        
        logger.info(f"🔧 Initialized evaluator with {len(self.model_configs)} model configs")
        logger.info(f"🔧 Top RAG configs: {len(self.top_rag_configs)}")
    
    def _load_test_data(self) -> List[Dict[str, Any]]:
        """Load FULL TEST DATASET for final evaluation"""
        
        test_file = self.data_dir / "simplified_triage_dialogues_test.json"
        if not test_file.exists():
            logger.error(f"❌ Test file not found: {test_file}")
            return []
        
        try:
            with open(test_file, 'r') as f:
                data = json.load(f)
            logger.info(f"✅ Loaded FULL TEST DATASET with {len(data)} cases")
        except Exception as e:
            logger.error(f"❌ Error loading test data: {e}")
            return []
        
        # Convert to standardized format
        converted_data = []
        for item in data:
            case = {
                "case_id": str(item["id"]),
                "query": item["query"],
                "triage_decision": item["final_triage_decision"],
                "next_steps": item["next_step"],
                "reasoning": item.get("reasoning", ""),
                "symptom": item.get("symptom", "unknown")
            }
            converted_data.append(case)
        
        logger.info(f"✅ Loaded {len(converted_data)} TEST cases")
        
        # Log data distribution
        triage_counts = {}
        for case in converted_data:
            decision = case["triage_decision"]
            triage_counts[decision] = triage_counts.get(decision, 0) + 1
        
        logger.info(f"📊 TEST DATASET Triage distribution: {triage_counts}")
        for decision, count in triage_counts.items():
            percentage = (count / len(converted_data)) * 100
            logger.info(f"    {decision}: {count} cases ({percentage:.1f}%)")
        
        return converted_data
    
    def _initialize_retrieval_system(self):
        """Initialize the optimized retrieval system"""
        try:
            embeddings_path = self.base_dir / "RAGdatav4" / "indiv_embeddings"
            self.retriever = OptimizedMultiSourceRetriever(
                model_name='all-MiniLM-L6-v2',
                embeddings_path=str(embeddings_path)
            )
            logger.info("✅ Optimized retrieval system initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize retrieval system: {e}")
            self.retriever = None
    
    def _define_top_rag_configs(self) -> List[Dict[str, Any]]:
        """Define top-performing RAG configurations from retrieval testing"""
        return [
            {
                "name": "top1_structured_contextual_diverse",
                "chunking_method": "structured_agent_tinfoil_medical",
                "retrieval_type": "contextual_rag", 
                "bias_config": "diverse",
                "pass_at_5": 0.595,  # From retrieval testing results
                "description": "Best overall performer - structured chunking + contextual RAG"
            },
            {
                "name": "top2_structured_pure_diverse", 
                "chunking_method": "structured_agent_tinfoil_medical",
                "retrieval_type": "pure_rag",
                "bias_config": "diverse", 
                "pass_at_5": 0.59,
                "description": "Close second - structured chunking + pure RAG"
            },
            {
                "name": "top3_contextual_sentence_diverse",
                "chunking_method": "contextual_sentence_c1024_o2_tinfoil",
                "retrieval_type": "contextual_rag",
                "bias_config": "diverse",
                "pass_at_5": 0.525,
                "description": "Third best - contextual sentence chunking"
            }
        ]
    
    def _define_top_5_model_configs(self) -> List[Dict[str, Any]]:
        """Define ONLY the TOP 5 performing configurations from evaluation results"""
        
        # Only include the exact model configuration needed for top 5
        model = {
            "name": "SmolLM2-135M_4bit",
            "path": self.models_dir / "SmolLM2-135M-Instruct-MLX_4bit",
            "family": "SmolLM2",
            "size": "135M",
            "quantization": "4bit",
            "adapters": [
                str(self.adapters_dir / "adapter_safe_triage_SmolLM2-135M_4bit_high_capacity_safe"),
                str(self.adapters_dir / "adapter_safe_triage_SmolLM2-135M_4bit_balanced_safe"),
                str(self.adapters_dir / "adapter_safe_triage_SmolLM2-135M_4bit_performance_safe")
            ]
        }
        
        return [model]
    
    def create_top_5_evaluation_matrix(self) -> List[EvaluationConfig]:
        """Create evaluation matrix for TOP 5 CONFIGURATIONS ONLY"""
        evaluation_configs = []
        
        logger.info("🏆 Creating TOP 5 evaluation matrix...")
        
        model = self.model_configs[0]  # Only SmolLM2-135M_4bit
        model_name = model["name"]
        model_path = str(model["path"])
        
        # TOP 5 CONFIGURATIONS based on evaluation results:
        
        # 1. SmolLM2-135M_4bit_high_capacity_safe_NoRAG (68.0% accuracy)
        high_capacity_adapter = str(self.adapters_dir / "adapter_safe_triage_SmolLM2-135M_4bit_high_capacity_safe")
        config1 = EvaluationConfig(
            model_name=model_name,
            model_path=model_path,
            adapter_path=high_capacity_adapter,
            rag_config=None,
            test_name=f"{model_name}_FineTuned_adapter_safe_triage_SmolLM2-135M_4bit_high_capacity_safe_NoRAG"
        )
        evaluation_configs.append(config1)
        
        # 2. SmolLM2-135M_4bit_balanced_safe_NoRAG (59.5% accuracy)
        balanced_adapter = str(self.adapters_dir / "adapter_safe_triage_SmolLM2-135M_4bit_balanced_safe")
        config2 = EvaluationConfig(
            model_name=model_name,
            model_path=model_path,
            adapter_path=balanced_adapter,
            rag_config=None,
            test_name=f"{model_name}_FineTuned_adapter_safe_triage_SmolLM2-135M_4bit_balanced_safe_NoRAG"
        )
        evaluation_configs.append(config2)
        
        # 3. SmolLM2-135M_4bit_performance_safe_NoRAG (57.0% accuracy)
        performance_adapter = str(self.adapters_dir / "adapter_safe_triage_SmolLM2-135M_4bit_performance_safe")
        config3 = EvaluationConfig(
            model_name=model_name,
            model_path=model_path,
            adapter_path=performance_adapter,
            rag_config=None,
            test_name=f"{model_name}_FineTuned_adapter_safe_triage_SmolLM2-135M_4bit_performance_safe_NoRAG"
        )
        evaluation_configs.append(config3)
        
        # 4. SmolLM2-135M_4bit_high_capacity_safe_RAG_top1_structured_contextual_diverse (55.0% accuracy)
        config4 = EvaluationConfig(
            model_name=model_name,
            model_path=model_path,
            adapter_path=high_capacity_adapter,
            rag_config=self.top_rag_configs[0],  # top1_structured_contextual_diverse
            test_name=f"{model_name}_FineTuned_adapter_safe_triage_SmolLM2-135M_4bit_high_capacity_safe_RAG_top1_structured_contextual_diverse"
        )
        evaluation_configs.append(config4)
        
        # 5. SmolLM2-135M_4bit_high_capacity_safe_RAG_top2_structured_pure_diverse (54.0% accuracy)
        config5 = EvaluationConfig(
            model_name=model_name,
            model_path=model_path,
            adapter_path=high_capacity_adapter,
            rag_config=self.top_rag_configs[1],  # top2_structured_pure_diverse
            test_name=f"{model_name}_FineTuned_adapter_safe_triage_SmolLM2-135M_4bit_high_capacity_safe_RAG_top2_structured_pure_diverse"
        )
        evaluation_configs.append(config5)
        
        logger.info(f"🏆 Created {len(evaluation_configs)} TOP 5 configurations for testing")
        logger.info("📊 All configurations use SmolLM2-135M_4bit model")
        logger.info("🎯 Testing on FULL test dataset (1975 cases)")
        
        return evaluation_configs
    
    def run_comprehensive_evaluation(self, 
                                   max_configs: Optional[int] = None,
                                   resume_from: Optional[str] = None) -> List[EvaluationResult]:
        """Run comprehensive evaluation on all configurations"""
        
        configs = self.create_top_5_evaluation_matrix()
        
        if max_configs:
            configs = configs[:max_configs]
            logger.info(f"🎯 Limited to first {max_configs} configurations")
        
        # Use the FULL TEST DATASET (1975 cases)
        test_sample = self.test_data
        
        results = []
        start_index = 0
        total_configs = len(configs)
        
        # Check for resume capability
        if resume_from:
            results, start_index = self._load_progress_and_resume(resume_from, configs)
            if start_index > 0:
                logger.info(f"🔄 Resuming from configuration {start_index+1}/{total_configs}")
        
        logger.info(f"🏆 Starting TOP 5 FINAL TESTING of {total_configs} configurations")
        logger.info(f"📊 Testing on {len(test_sample)} TEST cases (FULL DATASET)")
        
        # Create progress file for incremental saving
        progress_file = resume_from if resume_from else f"final_test_results_llm_judge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        for i, config in enumerate(configs[start_index:], start_index + 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"🔄 EVALUATION {i}/{total_configs}: {config.test_name}")
            logger.info(f"{'='*80}")
            
            try:
                result = self._evaluate_single_configuration(config, test_sample)
                results.append(result)
                
                logger.info(f"✅ COMPLETED {i}/{total_configs}: "
                          f"Acc={result.triage_accuracy:.3f}, "
                          f"F2={result.f2_score:.3f}, "
                          f"Time={result.avg_inference_time_per_case:.2f}s/case")
                
                # Save progress after each configuration
                self._save_incremental_progress(results, progress_file, i, total_configs)
                
            except Exception as e:
                logger.error(f"❌ FAILED {i}/{total_configs}: {config.test_name} - {e}")
                # Create error result
                error_result = EvaluationResult(
                    config=config,
                    timestamp=datetime.now().isoformat(),
                    triage_accuracy=0.0,
                    f1_score=0.0,
                    f2_score=0.0,
                    confusion_matrix=[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                    classification_report={},
                    total_inference_time=0.0,
                    avg_inference_time_per_case=0.0,
                    cases_evaluated=0,
                    success_count=0,
                    error_count=len(test_sample),
                    error_details=[str(e)],
                    llm_judge_metrics=None  # No LLM judge metrics for failed runs
                )
                results.append(error_result)
                
                # Save progress even after errors
                self._save_incremental_progress(results, progress_file, i, total_configs)
        
        logger.info(f"\n🎉 TOP 5 FINAL TESTING COMPLETED!")
        logger.info(f"📊 Total configurations: {total_configs}")
        logger.info(f"✅ Successful: {len([r for r in results if r.error_count == 0])}")
        logger.info(f"❌ Failed: {len([r for r in results if r.error_count > 0])}")
        
        return results
    
    def _evaluate_single_configuration(self, 
                                     config: EvaluationConfig, 
                                     test_data: List[Dict[str, Any]]) -> EvaluationResult:
        """Evaluate a single configuration on test data"""
        
        from testing_core_llm_as_judge import TriageInferenceEngine, PerformanceCalculator
        
        # Initialize inference engine with LLM judge
        engine = TriageInferenceEngine(
            retriever=self.retriever,
            enable_llm_judge=True,
            judge_model="llama3-3-70b",
            max_judge_workers=50  # 50 parallel LLM judge workers
        )
        
        # Load model
        model, tokenizer = engine.load_model_if_needed(
            model_path=config.model_path,
            adapter_path=config.adapter_path
        )
        
        # Run parallel evaluation on all test cases
        logger.info(f"🔄 Running parallel inference and LLM evaluation on {len(test_data)} cases...")
        
        case_results = engine.evaluate_cases_parallel(
            model=model,
            tokenizer=tokenizer,
            cases=test_data,
            rag_config=config.rag_config
        )
        
        # Calculate performance metrics
        metrics = PerformanceCalculator.calculate_metrics(case_results)
        
        # Print summary
        PerformanceCalculator.print_performance_summary(metrics, config.test_name)
        
        # Create comprehensive result
        result = EvaluationResult(
            config=config,
            timestamp=datetime.now().isoformat(),
            triage_accuracy=metrics["triage_accuracy"],
            f1_score=metrics["f1_score"],
            f2_score=metrics["f2_score"],
            confusion_matrix=metrics["confusion_matrix"],
            classification_report=metrics["classification_report"],
            total_inference_time=metrics["timing_stats"]["total_inference_time"],
            avg_inference_time_per_case=metrics["timing_stats"]["avg_inference_time"],
            cases_evaluated=metrics["cases_evaluated"],
            success_count=metrics["success_count"],
            error_count=metrics["error_count"],
            error_details=[r["error"] for r in case_results if not r["success"]],
            unknown_triage_count=metrics["unknown_triage_count"],
            total_failures=metrics["total_failures"],
            rag_retrieval_time=metrics["timing_stats"].get("avg_rag_time"),
            rag_context_length_avg=metrics["timing_stats"].get("avg_context_length"),
            llm_judge_metrics=metrics.get("llm_judge_metrics")  # Add LLM judge scores
        )
        
        return result
    
    def _save_incremental_progress(self, results: List[EvaluationResult], 
                                 progress_file: str, current: int, total: int):
        """Save evaluation progress incrementally"""
        try:
            progress_data = {
                "timestamp": datetime.now().isoformat(),
                "progress": f"{current}/{total}",
                "completed_configs": current,
                "total_configs": total,
                "results": [asdict(result) for result in results]
            }
            
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            
            logger.info(f"💾 Progress saved: {current}/{total} configs to {progress_file}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save progress: {e}")
    
    def _load_progress_and_resume(self, progress_file: str, configs: List[EvaluationConfig]) -> Tuple[List[EvaluationResult], int]:
        """Load previous progress and determine where to resume"""
        try:
            if not Path(progress_file).exists():
                logger.warning(f"⚠️ Progress file {progress_file} not found, starting fresh")
                return [], 0
            
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            
            # Recreate EvaluationResult objects from saved data
            results = []
            for result_data in progress_data.get("results", []):
                # Recreate EvaluationConfig
                config_data = result_data["config"]
                config = EvaluationConfig(
                    model_name=config_data["model_name"],
                    model_path=config_data["model_path"],
                    adapter_path=config_data.get("adapter_path"),
                    rag_config=config_data.get("rag_config"),
                    test_name=config_data["test_name"]
                )
                
                # Recreate EvaluationResult
                result = EvaluationResult(
                    config=config,
                    timestamp=result_data["timestamp"],
                    triage_accuracy=result_data["triage_accuracy"],
                    f1_score=result_data["f1_score"],
                    f2_score=result_data["f2_score"],
                    confusion_matrix=result_data["confusion_matrix"],
                    classification_report=result_data["classification_report"],
                    total_inference_time=result_data["total_inference_time"],
                    avg_inference_time_per_case=result_data["avg_inference_time_per_case"],
                    cases_evaluated=result_data["cases_evaluated"],
                    success_count=result_data["success_count"],
                    error_count=result_data["error_count"],
                    error_details=result_data["error_details"],
                    rag_retrieval_time=result_data.get("rag_retrieval_time"),
                    rag_context_length_avg=result_data.get("rag_context_length_avg"),
                    unknown_triage_count=result_data.get("unknown_triage_count", 0),
                    total_failures=result_data.get("total_failures", 0),
                    llm_judge_metrics=result_data.get("llm_judge_metrics")  # Add LLM judge metrics
                )
                results.append(result)
            
            completed_configs = progress_data.get("completed_configs", 0)
            logger.info(f"📂 Loaded progress: {completed_configs} configurations completed")
            
            return results, completed_configs
            
        except Exception as e:
            logger.error(f"❌ Failed to load progress: {e}")
            return [], 0
    
    @classmethod
    def find_latest_progress_file(cls) -> Optional[str]:
        """Find the most recent progress file in current directory"""
        progress_files = list(Path(".").glob("evaluation_progress_*.json"))
        if not progress_files:
            return None
        
        # Sort by modification time, newest first
        latest_file = max(progress_files, key=lambda p: p.stat().st_mtime)
        return str(latest_file)

if __name__ == "__main__":
    import sys
    
    tester = Top5MedicalTriageTester()
    
    # Check for resume argument
    if len(sys.argv) > 1 and sys.argv[1] == "--resume":
        # Try to find latest progress file
        latest_progress = tester.find_latest_progress_file()
        if latest_progress:
            logger.info(f"🔄 Found progress file: {latest_progress}")
            results = tester.run_comprehensive_evaluation(resume_from=latest_progress)
        else:
            logger.info("⚠️ No progress file found, starting fresh")
            results = tester.run_comprehensive_evaluation()
    elif len(sys.argv) > 1 and sys.argv[1].startswith("--resume="):
        # Resume from specific file
        resume_file = sys.argv[1].split("=", 1)[1]
        logger.info(f"🔄 Resuming from: {resume_file}")
        results = tester.run_comprehensive_evaluation(resume_from=resume_file)
    elif len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run small test first (single configuration on subset)
        logger.info("🧪 Running initial test with single configuration...")
        results = tester.run_comprehensive_evaluation(max_configs=1)
    else:
        # Full testing on all TOP 5 configurations
        logger.info("🏆 Starting TOP 5 FINAL TESTING on full test dataset...")
        results = tester.run_comprehensive_evaluation()
    
    # Save final results
    output_file = f"top5_final_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    results_data = []
    for result in results:
        results_data.append(asdict(result))
    
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    logger.info(f"💾 TOP 5 FINAL TEST results saved to {output_file}")