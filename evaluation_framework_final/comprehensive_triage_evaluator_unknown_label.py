#!/usr/bin/env python3
"""
Comprehensive Medical Triage Evaluation Pipeline - WITH UNKNOWN LABEL TRACKING

This version properly tracks UNKNOWN extractions as incorrect instead of mapping to GP.

This pipeline tests model performance on medical triage tasks using:
1. Top-performing RAG configurations from retrieval testing
2. Fine-tuned models with safety adapters (base models excluded)
3. Comprehensive logging and F2 score evaluation on validation data
4. PROPER HANDLING: Failed extractions marked as "UNKNOWN" → incorrect

Key Features:
- Proper integration with optimized hybrid retrieval
- Only fine-tuned model evaluation (base models can't do structured output)
- RAG vs no-RAG comparison
- Comprehensive logging for each evaluation step
- F2 score optimization (β=2 emphasizes recall for medical safety)
- Honest accuracy metrics (no false GP inflation)
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

import numpy as np
from sklearn.metrics import f1_score, fbeta_score, classification_report, confusion_matrix

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_evaluation_unknown_label.log'),
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
final_retrieval_path = Path(__file__).parent.parent / "final_retrieval_testing"
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
    
    # Triage extraction tracking - NEW FIELDS
    unknown_triage_count: int = 0
    total_failures: int = 0
    
    # RAG-specific metrics (if applicable)
    rag_retrieval_time: Optional[float] = None
    rag_context_length_avg: Optional[float] = None

class ComprehensiveMedicalTriageEvaluator:
    """Main evaluation pipeline for comprehensive medical triage testing"""
    
    def __init__(self, base_dir: str = "/Users/choemanseung/789/hft"):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "mlx_models"
        self.adapters_dir = self.base_dir / "safety_triage_adapters"
        self.data_dir = self.base_dir / "Final_dataset"
        
        # Load validation dataset
        self.validation_data = self._load_validation_data()
        logger.info(f"📊 Loaded {len(self.validation_data)} validation cases")
        
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
        
        # Define model configurations
        self.model_configs = self._define_model_configs()
        
        logger.info(f"🔧 Initialized evaluator with {len(self.model_configs)} model configs")
        logger.info(f"🔧 Top RAG configs: {len(self.top_rag_configs)}")
    
    def _load_validation_data(self) -> List[Dict[str, Any]]:
        """Load validation dataset for evaluation"""
        
        # First try to load stratified 200-case sample
        stratified_file = Path(__file__).parent / "stratified_sample_200.json"
        if stratified_file.exists():
            try:
                with open(stratified_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"✅ Using stratified 200-case sample")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load stratified sample: {e}")
                data = None
        else:
            data = None
        
        # Fallback to full validation data
        if data is None:
            val_file = self.data_dir / "simplified_triage_dialogues_val.json"
            if not val_file.exists():
                logger.error(f"❌ Validation file not found: {val_file}")
                return []
            
            try:
                with open(val_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"✅ Using full validation dataset")
            except Exception as e:
                logger.error(f"❌ Error loading validation data: {e}")
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
        
        logger.info(f"✅ Loaded {len(converted_data)} validation cases")
        
        # Log data distribution
        triage_counts = {}
        for case in converted_data:
            decision = case["triage_decision"]
            triage_counts[decision] = triage_counts.get(decision, 0) + 1
        
        logger.info(f"📊 Triage distribution: {triage_counts}")
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
    
    def _define_model_configs(self) -> List[Dict[str, Any]]:
        """Define available model configurations"""
        models = []
        
        # Base model configurations
        base_models = [
            {
                "name": "SmolLM2-135M_4bit",
                "path": self.models_dir / "SmolLM2-135M-Instruct-MLX_4bit",
                "family": "SmolLM2",
                "size": "135M",
                "quantization": "4bit"
            },
            {
                "name": "SmolLM2-135M_8bit", 
                "path": self.models_dir / "SmolLM2-135M-Instruct-MLX_8bit",
                "family": "SmolLM2",
                "size": "135M", 
                "quantization": "8bit"
            },
            {
                "name": "SmolLM2-360M_4bit",
                "path": self.models_dir / "SmolLM2-360M-Instruct-MLX_4bit",
                "family": "SmolLM2",
                "size": "360M",
                "quantization": "4bit"
            },
            {
                "name": "SmolLM2-360M_8bit",
                "path": self.models_dir / "SmolLM2-360M-Instruct-MLX_8bit", 
                "family": "SmolLM2",
                "size": "360M",
                "quantization": "8bit"
            },
            {
                "name": "Gemma-270M_4bit",
                "path": self.models_dir / "gemma-270m-mlx_4bit",
                "family": "Gemma",
                "size": "270M",
                "quantization": "4bit"
            },
            {
                "name": "Gemma-270M_8bit",
                "path": self.models_dir / "gemma-270m-mlx_8bit",
                "family": "Gemma", 
                "size": "270M",
                "quantization": "8bit"
            }
        ]
        
        # Find matching adapters for each base model
        for model in base_models:
            model_family = model["family"]
            model_size = model["size"]
            quantization = model["quantization"]
            
            # Look for adapters matching this model
            adapter_pattern = f"adapter_safe_triage_{model_family}-{model_size}_{quantization}_*_safe"
            matching_adapters = list(self.adapters_dir.glob(adapter_pattern))
            
            model["adapters"] = [str(adapter) for adapter in matching_adapters]
            models.append(model)
        
        return models
    
    def create_evaluation_matrix(self) -> List[EvaluationConfig]:
        """Create evaluation matrix - ONLY fine-tuned models with adapters"""
        evaluation_configs = []
        
        logger.info("🔧 Creating evaluation matrix (fine-tuned models only)...")
        
        for model in self.model_configs:
            model_name = model["name"]
            model_path = str(model["path"])
            
            # Skip base models - only evaluate fine-tuned models with adapters
            # Fine-tuned models without RAG
            for adapter_path in model["adapters"]:
                adapter_name = Path(adapter_path).name
                config = EvaluationConfig(
                    model_name=model_name,
                    model_path=model_path,
                    adapter_path=adapter_path,
                    rag_config=None,
                    test_name=f"{model_name}_FineTuned_{adapter_name}_NoRAG"
                )
                evaluation_configs.append(config)
                
                # Fine-tuned models with top RAG configs  
                for rag_config in self.top_rag_configs:
                    config = EvaluationConfig(
                        model_name=model_name,
                        model_path=model_path,
                        adapter_path=adapter_path,
                        rag_config=rag_config,
                        test_name=f"{model_name}_FineTuned_{adapter_name}_RAG_{rag_config['name']}"
                    )
                    evaluation_configs.append(config)
        
        logger.info(f"📊 Created {len(evaluation_configs)} fine-tuned evaluation configurations")
        logger.info("ℹ️  Skipping base models - only evaluating fine-tuned models with adapters")
        return evaluation_configs
    
    def run_comprehensive_evaluation(self, 
                                   max_configs: Optional[int] = None,
                                   resume_from: Optional[str] = None) -> List[EvaluationResult]:
        """Run comprehensive evaluation on all configurations"""
        
        configs = self.create_evaluation_matrix()
        
        if max_configs:
            configs = configs[:max_configs]
            logger.info(f"🎯 Limited to first {max_configs} configurations")
        
        # Use the loaded validation data (either stratified 200 or full dataset)
        validation_sample = self.validation_data
        
        results = []
        start_index = 0
        total_configs = len(configs)
        
        # Check for resume capability
        if resume_from:
            results, start_index = self._load_progress_and_resume(resume_from, configs)
            if start_index > 0:
                logger.info(f"🔄 Resuming from configuration {start_index+1}/{total_configs}")
        
        logger.info(f"🚀 Starting comprehensive evaluation of {total_configs} configurations")
        logger.info(f"📊 Evaluating on {len(validation_sample)} validation cases")
        
        # Create progress file for incremental saving
        progress_file = resume_from if resume_from else f"evaluation_progress_unknown_label_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        for i, config in enumerate(configs[start_index:], start_index + 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"🔄 EVALUATION {i}/{total_configs}: {config.test_name}")
            logger.info(f"{'='*80}")
            
            try:
                result = self._evaluate_single_configuration(config, validation_sample)
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
                    error_count=len(validation_sample),
                    error_details=[str(e)]
                )
                results.append(error_result)
                
                # Save progress even after errors
                self._save_incremental_progress(results, progress_file, i, total_configs)
        
        logger.info(f"\n🎉 COMPREHENSIVE EVALUATION COMPLETED!")
        logger.info(f"📊 Total configurations: {total_configs}")
        logger.info(f"✅ Successful: {len([r for r in results if r.error_count == 0])}")
        logger.info(f"❌ Failed: {len([r for r in results if r.error_count > 0])}")
        
        return results
    
    def _evaluate_single_configuration(self, 
                                     config: EvaluationConfig, 
                                     validation_data: List[Dict[str, Any]]) -> EvaluationResult:
        """Evaluate a single configuration on validation data"""
        
        from evaluation_core import TriageInferenceEngine, PerformanceCalculator
        
        # Initialize inference engine
        engine = TriageInferenceEngine(retriever=self.retriever)
        
        # Load model
        model, tokenizer = engine.load_model_if_needed(
            model_path=config.model_path,
            adapter_path=config.adapter_path
        )
        
        # Run evaluation on all validation cases
        case_results = []
        total_cases = len(validation_data)
        
        logger.info(f"🔄 Running inference on {total_cases} cases...")
        
        for i, case in enumerate(validation_data, 1):
            if i % 50 == 0 or i == total_cases:
                logger.info(f"   Progress: {i}/{total_cases} cases completed")
            
            case_result = engine.evaluate_single_case(
                model=model,
                tokenizer=tokenizer,
                case=case,
                rag_config=config.rag_config
            )
            case_results.append(case_result)
        
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
            rag_context_length_avg=metrics["timing_stats"].get("avg_context_length")
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
                "evaluation_type": "UNKNOWN_LABEL_TRACKING",
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
                
                # Recreate EvaluationResult with unknown tracking fields
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
                    unknown_triage_count=result_data.get("unknown_triage_count", 0),
                    total_failures=result_data.get("total_failures", 0),
                    rag_retrieval_time=result_data.get("rag_retrieval_time"),
                    rag_context_length_avg=result_data.get("rag_context_length_avg")
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
        progress_files = list(Path(".").glob("evaluation_progress_unknown_label_*.json"))
        if not progress_files:
            return None
        
        # Sort by modification time, newest first
        latest_file = max(progress_files, key=lambda p: p.stat().st_mtime)
        return str(latest_file)

if __name__ == "__main__":
    import sys
    
    evaluator = ComprehensiveMedicalTriageEvaluator()
    
    # Check for resume argument
    if len(sys.argv) > 1 and sys.argv[1] == "--resume":
        # Try to find latest progress file
        latest_progress = evaluator.find_latest_progress_file()
        if latest_progress:
            logger.info(f"🔄 Found progress file: {latest_progress}")
            results = evaluator.run_comprehensive_evaluation(resume_from=latest_progress)
        else:
            logger.info("⚠️ No progress file found, starting fresh")
            results = evaluator.run_comprehensive_evaluation()
    elif len(sys.argv) > 1 and sys.argv[1].startswith("--resume="):
        # Resume from specific file
        resume_file = sys.argv[1].split("=", 1)[1]
        logger.info(f"🔄 Resuming from: {resume_file}")
        results = evaluator.run_comprehensive_evaluation(resume_from=resume_file)
    elif len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run small test first (limit configurations)
        logger.info("🧪 Running initial test with limited scope...")
        results = evaluator.run_comprehensive_evaluation(max_configs=3)
    else:
        # Full evaluation
        logger.info("🚀 Starting full comprehensive evaluation with UNKNOWN label tracking...")
        results = evaluator.run_comprehensive_evaluation()
    
    # Save final results
    output_file = f"comprehensive_evaluation_results_unknown_label_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    results_data = []
    for result in results:
        results_data.append(asdict(result))
    
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    logger.info(f"💾 Final results saved to {output_file}")