#!/usr/bin/env python3
"""
Enhanced Evaluation Pipeline with Real Clinical Appropriateness Evaluation
Evaluates final results using TinfoilAgent LLM-as-judge after each combination.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import psutil
import os
from sklearn.metrics import f1_score, fbeta_score, classification_report
import concurrent.futures
import threading
from functools import partial
import random
from collections import defaultdict

# MLX imports with fallback
try:
    from mlx_lm import load, generate
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    print("⚠️  MLX not available. Using mock implementations for testing.")
    MLX_AVAILABLE = False
    
    # Mock implementations for testing without MLX
    def load(*args, **kwargs):
        class MockModel:
            pass
        class MockTokenizer:
            pass
        return MockModel(), MockTokenizer()
    
    def generate(model, tokenizer, prompt, max_tokens=200, verbose=False):
        # Simple mock response based on prompt keywords
        if "chest pain" in prompt.lower():
            return "ED - Emergency department evaluation recommended for chest pain"
        elif "headache" in prompt.lower():
            return "HOME - Rest and over-the-counter medication recommended"
        else:
            return "GP - General practitioner consultation recommended"

# Import classes from optimized config generator
from optimized_config_generator import EvaluationCombo, AdapterConfig
from integrated_rag_system import IntegratedRAGSystem, ValidatedRAGConfigProcessor
from appropriateness_judge import ClinicalAppropriatenessJudge, create_clinical_case_from_evaluation

@dataclass
class EvaluationResult:
    """Results from a single evaluation run"""
    combo_id: str
    timestamp: str
    triage_accuracy: float
    next_steps_accuracy: float
    clinical_appropriateness: float
    memory_usage_mb: float
    inference_speed_tps: float
    latency_ms: float
    error_message: Optional[str] = None
    detailed_metrics: Optional[Dict[str, Any]] = None

class IntegratedRAGSystemWrapper:
    """Wrapper for integrated RAG system to match evaluation interface"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rag_system = IntegratedRAGSystem()
        self.config_processor = ValidatedRAGConfigProcessor()
        
        # Process config to get detailed parameters
        self.processed_config = self.config_processor.process_config(config)
        
        chunking_method = config.get('chunking_method', 'unknown')
        retrieval_type = config.get('retrieval_type', 'unknown')
        print(f"Initializing integrated RAG system: {chunking_method}/{retrieval_type}")
    
    def retrieve_context(self, query: str) -> str:
        """Retrieve relevant context for query using integrated system"""
        return self.rag_system.retrieve_context(query, self.processed_config)

class EvaluationPipeline:
    """Base evaluation pipeline for medical triage system"""
    
    def __init__(self, base_dir: str = "/Users/choemanseung/789/hft", 
                 batch_size: int = 8, max_workers: int = 2, llm_judge_workers: int = 40):
        self.base_dir = Path(base_dir)
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.llm_judge_workers = llm_judge_workers  # Parallel workers for LLM judge
        
        self.test_dataset = self._load_test_dataset()
        
        # Current model cache to avoid reloading
        self.current_model = None
        self.current_tokenizer = None
        self.current_adapter_path = None
    
    def _stratified_sample(self, data: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
        """Create stratified sample maintaining ED/GP/HOME ratios"""
        # Group by triage decision
        grouped = defaultdict(list)
        for item in data:
            grouped[item["triage_decision"]].append(item)
        
        # Calculate original ratios
        total_cases = len(data)
        ratios = {decision: len(cases) / total_cases for decision, cases in grouped.items()}
        
        # Calculate target counts for sample
        target_counts = {}
        for decision, ratio in ratios.items():
            target_counts[decision] = max(1, int(sample_size * ratio))
        
        # Adjust if total doesn't match exactly
        total_target = sum(target_counts.values())
        if total_target != sample_size:
            # Adjust the largest group
            largest_group = max(target_counts.keys(), key=lambda k: target_counts[k])
            target_counts[largest_group] += sample_size - total_target
        
        # Sample from each group
        sampled_data = []
        for decision, target_count in target_counts.items():
            available = grouped[decision]
            if len(available) >= target_count:
                sampled = random.sample(available, target_count)
            else:
                sampled = available  # Use all if not enough
            sampled_data.extend(sampled)
        
        # Shuffle the final sample
        random.shuffle(sampled_data)
        
        print(f"📊 Stratified sampling summary:")
        for decision in ["ED", "GP", "HOME"]:
            original_count = len(grouped[decision])
            sampled_count = len([x for x in sampled_data if x["triage_decision"] == decision])
            original_pct = (original_count / total_cases) * 100
            sampled_pct = (sampled_count / len(sampled_data)) * 100
            print(f"   {decision}: {original_count} → {sampled_count} ({original_pct:.1f}% → {sampled_pct:.1f}%)")
        
        return sampled_data
    
    def _load_test_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset for evaluation (validation set for hyperparameter tuning)"""
        # Check if we should use validation set for hyperparameter tuning
        use_validation = os.getenv("USE_VALIDATION_SET", "false").lower() == "true"
        
        if use_validation:
            data_file = self.base_dir / "Final_dataset" / "simplified_triage_dialogues_val.json"
            dataset_type = "validation"
        else:
            data_file = self.base_dir / "Final_dataset" / "simplified_triage_dialogues_test.json"
            dataset_type = "test"
        
        if data_file.exists():
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                
                # Convert to evaluation format
                converted_data = []
                for item in data:
                    case = {
                        "input": item["query"],  # Use "query" as input as requested
                        "triage_decision": item["final_triage_decision"],
                        "next_steps": item["next_step"],
                        "case_id": str(item["id"]),
                        "symptom": item.get("symptom", "unknown"),
                        "reasoning": item.get("reasoning", "")
                    }
                    converted_data.append(case)
                
                # Take a configurable subset for faster evaluation
                subset_size = int(os.getenv("EVAL_SUBSET_SIZE", "30"))  # Default 30 for validation
                
                print(f"📊 Loaded {len(converted_data)} real triage {dataset_type} cases")
                
                if subset_size >= len(converted_data):
                    print(f"   Using all {len(converted_data)} cases")
                    return converted_data
                elif use_validation and subset_size < len(converted_data):
                    print(f"   Creating stratified sample of {subset_size} cases")
                    return self._stratified_sample(converted_data, subset_size)
                else:
                    print(f"   Using first {subset_size} cases")
                    return converted_data[:subset_size]
                
            except Exception as e:
                print(f"Error loading real {dataset_type} dataset: {e}")
        else:
            print(f"⚠️  Real {dataset_type} file not found: {data_file}")
        
        # Fallback to mock data if real data unavailable
        print("📝 Using mock test data as fallback")
        return self._generate_mock_test_data()
    
    def _generate_mock_test_data(self) -> List[Dict[str, Any]]:
        """Generate mock test data for evaluation"""
        mock_cases = []
        
        conditions = [
            ("chest pain, shortness of breath", "ED"),
            ("mild headache, no fever", "HOME"),
            ("persistent cough, fever", "GP"),
            ("severe abdominal pain", "ED"),
            ("minor cut on finger", "HOME")
        ]
        
        for i, (symptoms, triage) in enumerate(conditions * 20):  # 100 cases
            case = {
                "input": f"Patient {i+1}: {symptoms}",
                "triage_decision": triage,
                "next_steps": f"Recommended action for {triage} case",
                "case_id": f"mock_case_{i+1}"
            }
            mock_cases.append(case)
        
        return mock_cases
    
    def _load_model_with_adapter(self, adapter_config: AdapterConfig):
        """Load model with specific adapter"""
        # If skip_models is enabled, return mock objects
        if self.skip_models:
            print(f"🔄 Skipping model loading (mock mode): {adapter_config.model_name}")
            return load(), load()[1]  # Return mock objects
        
        # Check if we already have this adapter loaded
        if (self.current_adapter_path == adapter_config.adapter_path and 
            self.current_model is not None):
            return self.current_model, self.current_tokenizer
        
        try:
            print(f"Loading model: {adapter_config.model_name} with adapter: {adapter_config.adapter_path}")
            
            # Use local MLX models instead of downloading from HuggingFace
            model_mapping = {
                "SmolLM2-360M": "/Users/choemanseung/789/hft/mlx_models/SmolLM2-360M-Instruct-MLX_4bit",
                "SmolLM2-135M": "/Users/choemanseung/789/hft/mlx_models/SmolLM2-135M-Instruct-MLX_4bit", 
                "Gemma-270M": "/Users/choemanseung/789/hft/mlx_models/gemma-270m-mlx_4bit"
            }
            
            base_model_path = model_mapping.get(adapter_config.model_name, adapter_config.model_name)
            
            # Check if local model exists, otherwise fall back to mock
            if not os.path.exists(base_model_path):
                print(f"⚠️  Local model not found: {base_model_path}")
                if not MLX_AVAILABLE:
                    print("   Using mock model/tokenizer")
                    return load(), load()[1]  # Return mock objects
                else:
                    print("   Falling back to HuggingFace download")
            
            # Load model and tokenizer with adapter
            model, tokenizer = load(
                path_or_hf_repo=base_model_path,
                adapter_path=adapter_config.adapter_path
            )
            
            # Cache current model
            self.current_model = model
            self.current_tokenizer = tokenizer
            self.current_adapter_path = adapter_config.adapter_path
            
            return model, tokenizer
            
        except Exception as e:
            print(f"Error loading model/adapter: {e}")
            raise
    
    def _evaluate_single_case(self, model, tokenizer, rag_system, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single test case"""
        try:
            # Get RAG context
            context = rag_system.retrieve_context(test_case["input"])
            
            # Create prompt matching training format
            prompt = f"""Patient query: {test_case["input"]}

Context:
{context}

Provide triage decision, next steps, and reasoning:"""
            
            # Generate response
            start_time = time.time()
            response = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=200,
                verbose=False
            )
            end_time = time.time()
            
            # Extract structured components from response
            predicted_triage = self._extract_triage_decision(response)
            response_parts = self._extract_next_steps_and_reasoning(response)
            
            # Calculate metrics
            triage_correct = predicted_triage == test_case["triage_decision"]
            inference_time = end_time - start_time
            
            return {
                "case_id": test_case["case_id"],
                "predicted_triage": predicted_triage,
                "actual_triage": test_case["triage_decision"],
                "triage_correct": triage_correct,
                "predicted_next_steps": response_parts["next_steps"],
                "predicted_reasoning": response_parts["reasoning"],
                "actual_next_steps": test_case.get("next_steps", ""),
                "actual_reasoning": test_case.get("reasoning", ""),
                "full_response": response,
                "inference_time": inference_time
            }
            
        except Exception as e:
            return {
                "case_id": test_case.get("case_id", "unknown"),
                "error": str(e),
                "triage_correct": False,
                "inference_time": 0
            }
    
    def _extract_triage_decision(self, response: str) -> str:
        """Extract triage decision from structured model response"""
        response_upper = response.upper()
        
        # Look for structured "Triage Decision:" format first
        if "TRIAGE DECISION:" in response_upper:
            # Extract the line after "Triage Decision:"
            lines = response.split('\n')
            for i, line in enumerate(lines):
                if "TRIAGE DECISION:" in line.upper():
                    # Get the decision from the same line or next line
                    decision_text = line.upper()
                    if "ED" in decision_text:
                        return "ED"
                    elif "GP" in decision_text:
                        return "GP"
                    elif "HOME" in decision_text:
                        return "HOME"
                    # Check next line if decision not on same line
                    elif i + 1 < len(lines):
                        next_line = lines[i + 1].upper().strip()
                        if "ED" in next_line:
                            return "ED"
                        elif "GP" in next_line:
                            return "GP"
                        elif "HOME" in next_line:
                            return "HOME"
        
        # Fallback to keyword matching
        if "ED" in response_upper and "EMERGENCY" in response_upper:
            return "ED"
        elif "GP" in response_upper:
            return "GP"
        elif "HOME" in response_upper:
            return "HOME"
        elif "EMERGENCY" in response_upper:
            return "ED"
        elif "DOCTOR" in response_upper or "PHYSICIAN" in response_upper:
            return "GP"
        elif "SELF-CARE" in response_upper or "REST" in response_upper:
            return "HOME"
        
        # Default to GP if unclear
        return "GP"
    
    def _extract_next_steps_and_reasoning(self, response: str) -> Dict[str, str]:
        """Extract next steps and reasoning from structured model response"""
        lines = response.split('\n')
        next_steps = ""
        reasoning = ""
        
        current_section = None
        for line in lines:
            line_upper = line.upper().strip()
            if "NEXT STEP:" in line_upper:
                current_section = "next_steps"
                # Extract content from same line if present
                if ":" in line:
                    next_steps = line.split(":", 1)[1].strip()
            elif "REASONING:" in line_upper:
                current_section = "reasoning"
                # Extract content from same line if present  
                if ":" in line:
                    reasoning = line.split(":", 1)[1].strip()
            elif current_section == "next_steps" and line.strip() and not line_upper.startswith("REASONING"):
                next_steps += " " + line.strip()
            elif current_section == "reasoning" and line.strip():
                reasoning += " " + line.strip()
        
        return {
            "next_steps": next_steps.strip(),
            "reasoning": reasoning.strip()
        }
    
    def _measure_memory_usage(self) -> float:
        """Measure current memory usage in MB"""
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    
    def _calculate_performance_metrics(self, case_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate performance metrics from case results"""
        valid_results = [r for r in case_results if "error" not in r]
        
        if not valid_results:
            return {
                "triage_accuracy": 0.0,
                "f1_score": 0.0,
                "f2_score": 0.0,
                "avg_inference_time": 0.0,
                "total_inference_time": 0.0
            }
        
        # Basic accuracy
        triage_accuracy = np.mean([r["triage_correct"] for r in valid_results])
        avg_inference_time = np.mean([r["inference_time"] for r in valid_results])
        total_inference_time = sum([r["inference_time"] for r in valid_results])
        
        # Extract predictions and ground truth for F1/F2 calculation
        y_true = [r["actual_triage"] for r in valid_results]
        y_pred = [r["predicted_triage"] for r in valid_results]
        
        # Define class order with medical importance: ED > GP > HOME
        classes = ["ED", "GP", "HOME"]
        
        # Calculate F1 and F2 scores (weighted by class importance)
        f1 = f1_score(y_true, y_pred, labels=classes, average='weighted', zero_division=0)
        f2 = fbeta_score(y_true, y_pred, labels=classes, average='weighted', beta=2, zero_division=0)
        
        return {
            "triage_accuracy": float(triage_accuracy),
            "f1_score": float(f1),
            "f2_score": float(f2),
            "avg_inference_time": float(avg_inference_time),
            "total_inference_time": float(total_inference_time)
        }

@dataclass
class EnhancedEvaluationResult:
    """Enhanced results with core metrics"""
    combo_id: str
    timestamp: str
    triage_accuracy: float
    f1_score: float
    f2_score: float
    next_step_quality: float
    next_step_rationale: str
    memory_usage_mb: float
    inference_speed_tps: float
    latency_ms: float
    error_message: Optional[str] = None
    detailed_metrics: Optional[Dict[str, Any]] = None
    llm_judge_used: bool = False

class EnhancedEvaluationPipeline(EvaluationPipeline):
    """Enhanced evaluation pipeline with real LLM-as-judge clinical evaluation"""
    
    def __init__(self, base_dir: str = "/Users/choemanseung/789/hft", 
                 batch_size: int = 8, max_workers: int = 2,
                 enable_clinical_judge: bool = True, skip_models: bool = False,
                 llm_judge_workers: int = 30):
        super().__init__(base_dir, batch_size, max_workers, llm_judge_workers)
        
        self.enable_clinical_judge = enable_clinical_judge
        self.skip_models = skip_models
        
        # Initialize clinical appropriateness judge with TinfoilAgent
        if enable_clinical_judge:
            try:
                self.clinical_judge = ClinicalAppropriatenessJudge(
                    judge_model="llama3-3-70b",
                    use_tinfoil=True,
                    rate_limit_delay=1.0
                )
                print("✅ Clinical appropriateness judge initialized with TinfoilAgent")
            except Exception as e:
                print(f"⚠️  Clinical judge initialization failed: {e}")
                self.clinical_judge = None
                self.enable_clinical_judge = False
        else:
            self.clinical_judge = None
            print("🔄 Clinical appropriateness evaluation disabled")
    
    def _evaluate_single_clinical_case(self, clinical_case) -> Dict[str, Any]:
        """Evaluate a single clinical case with LLM judge (for parallel processing)"""
        try:
            case_id = getattr(clinical_case, 'case_id', 'unknown')
            
            if self.clinical_judge:
                scores = self.clinical_judge.evaluate_case(clinical_case)
                # Convert AppropriatenessScores to expected format
                result = {
                    "next_step_quality": scores.overall_score,
                    "rationale": scores.rationale,
                    "llm_used": True
                }
                return {
                    "case_id": case_id,
                    "success": True,
                    "result": result
                }
            else:
                return {
                    "case_id": case_id,
                    "success": False,
                    "result": {"next_step_quality": 5.0, "rationale": "No clinical judge available", "llm_used": False}
                }
        except Exception as e:
            case_id = getattr(clinical_case, 'case_id', 'unknown')
            return {
                "case_id": case_id,
                "success": False,
                "error": str(e),
                "result": {"next_step_quality": 0.0, "rationale": f"Error: {e}", "llm_used": False}
            }
    
    def _evaluate_clinical_appropriateness_parallel(self, clinical_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate clinical appropriateness using parallel workers"""
        if not clinical_cases:
            return {"next_step_quality": 0.0, "rationale": "No cases to evaluate", "llm_used": False}
        
        print(f"🤖 Running clinical appropriateness evaluation on {len(clinical_cases)} cases with {self.llm_judge_workers} parallel workers...")
        
        # Process cases in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.llm_judge_workers) as executor:
            # Submit all tasks
            future_to_case = {
                executor.submit(self._evaluate_single_clinical_case, case): case 
                for case in clinical_cases
            }
            
            # Collect results with progress tracking
            completed = 0
            for future in concurrent.futures.as_completed(future_to_case):
                result = future.result()
                results.append(result)
                completed += 1
                
                # Print progress every 100 cases
                if completed % 100 == 0 or completed == len(clinical_cases):
                    print(f"   Progress: {completed}/{len(clinical_cases)} clinical evaluations completed")
        
        # Aggregate results
        successful_results = [r["result"] for r in results if r["success"]]
        failed_count = len(results) - len(successful_results)
        
        if successful_results:
            avg_quality = np.mean([r["next_step_quality"] for r in successful_results])
            rationales = [r["rationale"] for r in successful_results]
            sample_rationale = rationales[0] if rationales else "No rationale available"
            llm_used = any(r.get("llm_used", False) for r in successful_results)
        else:
            avg_quality = 0.0
            sample_rationale = "All evaluations failed"
            llm_used = False
        
        print(f"✅ Clinical evaluation completed: {len(successful_results)} successful, {failed_count} failed")
        
        return {
            "next_step_quality": avg_quality,
            "rationale": sample_rationale,
            "llm_used": llm_used,
            "total_cases": len(clinical_cases),
            "successful_cases": len(successful_results),
            "failed_cases": failed_count
        }
    
    def evaluate_combination_with_clinical_assessment(self, combo) -> EnhancedEvaluationResult:
        """Evaluate combination with real clinical appropriateness assessment"""
        start_memory = self._measure_memory_usage()
        timestamp = datetime.now().isoformat()
        
        try:
            chunk_limit = combo.rag_config.get('chunk_limit', 10)
            print(f"🔄 Evaluating with clinical assessment: {combo.combo_id}")
            print(f"   Using {chunk_limit} chunks from RAG retrieval")
            
            # Initialize integrated RAG system with validated config
            rag_system = IntegratedRAGSystemWrapper(combo.rag_config)
            
            # Load model with adapter
            model, tokenizer = self._load_model_with_adapter(combo.adapter_config)
            
            # Evaluate on test dataset and collect case-level results
            case_results = []
            clinical_cases = []
            
            # Use full test dataset if EVAL_SUBSET_SIZE is set to full size, otherwise limit for clinical evaluation
            clinical_eval_limit = int(os.getenv("CLINICAL_EVAL_LIMIT", "10"))
            test_cases_to_use = self.test_dataset if clinical_eval_limit >= len(self.test_dataset) else self.test_dataset[:clinical_eval_limit]
            
            for test_case in test_cases_to_use:
                # Get model prediction
                case_result = self._evaluate_single_case(model, tokenizer, rag_system, test_case)
                case_results.append(case_result)
                
                # Create clinical case for LLM judge evaluation
                if self.enable_clinical_judge and self.clinical_judge:
                    clinical_case = create_clinical_case_from_evaluation(
                        case_id=f"{combo.combo_id}_{test_case.get('case_id', len(clinical_cases))}",
                        patient_input=test_case["input"],
                        model_triage=case_result.get("predicted_triage", "GP"),
                        model_steps=case_result.get("response", "Standard care recommended"),
                        true_triage=test_case.get("triage_decision", "GP"),
                        true_steps=test_case.get("next_steps", "Follow up as needed")
                    )
                    clinical_cases.append(clinical_case)
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(case_results)
            end_memory = self._measure_memory_usage()
            
            # Calculate tokens per second
            total_tokens = len(case_results) * 200
            inference_speed = total_tokens / max(metrics["total_inference_time"], 0.001)
            
            # Perform clinical appropriateness evaluation with parallel processing
            print(f"📊 Model inference completed for {len(case_results)} cases. Starting parallel clinical evaluation...")
            clinical_scores = self._evaluate_clinical_appropriateness_parallel(clinical_cases)
            
            # Create enhanced result
            result = EnhancedEvaluationResult(
                combo_id=combo.combo_id,
                timestamp=timestamp,
                triage_accuracy=metrics["triage_accuracy"],
                f1_score=metrics["f1_score"],
                f2_score=metrics["f2_score"],
                next_step_quality=clinical_scores["next_step_quality"],
                next_step_rationale=clinical_scores["rationale"],
                memory_usage_mb=end_memory - start_memory,
                inference_speed_tps=inference_speed,
                latency_ms=metrics["avg_inference_time"] * 1000,
                detailed_metrics=metrics,
                llm_judge_used=clinical_scores["llm_used"]
            )
            
            print(f"✅ Completed evaluation: {combo.combo_id}")
            if clinical_scores["llm_used"]:
                print(f"   Accuracy: {metrics['triage_accuracy']:.3f}, F2: {metrics['f2_score']:.3f}, Next Step Quality: {clinical_scores['next_step_quality']:.1f}/10")
            else:
                print(f"   Accuracy: {metrics['triage_accuracy']:.3f}, F2: {metrics['f2_score']:.3f} (Next step evaluation disabled)")
            print(f"   Chunks used: {chunk_limit} per query")
            
            return result
            
        except Exception as e:
            print(f"❌ Error evaluating {combo.combo_id}: {e}")
            return EnhancedEvaluationResult(
                combo_id=combo.combo_id,
                timestamp=timestamp,
                triage_accuracy=0.0,
                f1_score=0.0,
                f2_score=0.0,
                next_step_quality=0.0,
                next_step_rationale="Evaluation failed",
                memory_usage_mb=0.0,
                inference_speed_tps=0.0,
                latency_ms=0.0,
                error_message=str(e),
                llm_judge_used=False
            )
    
    def _evaluate_clinical_appropriateness(self, clinical_cases: List) -> Dict[str, Any]:
        """Evaluate next step quality using LLM judge"""
        
        if not self.enable_clinical_judge or not self.clinical_judge or not clinical_cases:
            return {
                "next_step_quality": 0.0,  # Set to 0 to indicate not evaluated
                "rationale": "Next step evaluation disabled",
                "llm_used": False
            }
        
        try:
            print(f"🤖 Running clinical appropriateness evaluation on {len(clinical_cases)} cases...")
            
            # Evaluate each case with TinfoilAgent
            all_scores = []
            for case in clinical_cases:
                try:
                    scores = self.clinical_judge.evaluate_case(case)
                    all_scores.append(scores)
                    print(f"   Case {case.case_id}: {scores.overall_score:.1f}/10")
                    
                except Exception as e:
                    print(f"   ⚠️  Case {case.case_id} evaluation failed: {e}")
                    continue
            
            if all_scores:
                # Calculate average next step quality score
                avg_next_step_quality = np.mean([s.overall_score for s in all_scores])  # Use overall_score as next step quality
                
                # Combine rationales
                rationales = [s.rationale for s in all_scores if s.rationale]
                combined_rationale = f"Average next step quality across {len(all_scores)} cases. " + \
                                   (rationales[0][:100] + "..." if rationales else "No detailed rationale available.")
                
                return {
                    "next_step_quality": float(avg_next_step_quality),
                    "rationale": combined_rationale,
                    "llm_used": True
                }
            else:
                print("⚠️  No successful next step evaluations")
                return {
                    "next_step_quality": 5.0,
                    "rationale": "Next step evaluation failed for all cases",
                    "llm_used": False
                }
                
        except Exception as e:
            print(f"❌ Next step evaluation error: {e}")
            return {
                "next_step_quality": 5.0,
                "rationale": f"Next step evaluation error: {str(e)}",
                "llm_used": False
            }
    
    def run_enhanced_evaluation(self, evaluation_combos: List, 
                              resume: bool = True) -> List[EnhancedEvaluationResult]:
        """Run evaluation with enhanced clinical assessment"""
        print(f"🚀 Starting enhanced evaluation with clinical assessment")
        print(f"   Evaluating {len(evaluation_combos)} combinations (includes 5 and 10 chunk variants)")
        print(f"   Clinical judge: {'Enabled (TinfoilAgent)' if self.enable_clinical_judge else 'Disabled'}")
        
        # Filter out already completed combinations if resuming
        if resume:
            # This would need to be implemented with proper resume logic
            pass
        
        results = []
        for i, combo in enumerate(evaluation_combos):
            print(f"\n📊 Progress: {i+1}/{len(evaluation_combos)}")
            
            result = self.evaluate_combination_with_clinical_assessment(combo)
            results.append(result)
            
            # Save intermediate results
            self._save_intermediate_result(result)
            
            # Print progress summary
            if (i + 1) % 5 == 0:
                self._print_progress_summary(results)
        
        print("\n🎉 Enhanced evaluation pipeline completed!")
        
        # Save comprehensive final results
        self._save_final_results(results)
        
        return results
    
    def _save_intermediate_result(self, result: EnhancedEvaluationResult):
        """Save intermediate result to prevent data loss"""
        try:
            results_file = self.base_dir / "enhanced_evaluation_results.jsonl"
            
            # Ensure directory exists
            results_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Append result as JSON line
            with open(results_file, 'a') as f:
                json.dump(asdict(result), f)
                f.write('\n')
        except Exception as e:
            print(f"⚠️ Failed to save intermediate result: {e}")
    
    def _save_final_results(self, results: List[EnhancedEvaluationResult]):
        """Save comprehensive final evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Create results directory
            results_dir = self.base_dir / f"evaluation_results_{timestamp}"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Save detailed results as JSON
            detailed_file = results_dir / "detailed_results.json"
            detailed_data = [asdict(result) for result in results]
            with open(detailed_file, 'w') as f:
                json.dump(detailed_data, f, indent=2)
            
            # 2. Save summary CSV for easy analysis
            summary_file = results_dir / "results_summary.csv"
            self._save_results_csv(results, summary_file)
            
            # 3. Save top performers
            top_performers_file = results_dir / "top_performers.json"
            self._save_top_performers(results, top_performers_file)
            
            # 4. Save configuration analysis
            config_analysis_file = results_dir / "configuration_analysis.json"
            self._save_configuration_analysis(results, config_analysis_file)
            
            # 5. Save evaluation metadata
            metadata_file = results_dir / "evaluation_metadata.json"
            self._save_evaluation_metadata(results, metadata_file)
            
            print(f"💾 Comprehensive results saved to: {results_dir}")
            print(f"   📊 Detailed results: {detailed_file}")
            print(f"   📋 Summary CSV: {summary_file}")
            print(f"   🏆 Top performers: {top_performers_file}")
            print(f"   🔧 Config analysis: {config_analysis_file}")
            
        except Exception as e:
            print(f"❌ Failed to save final results: {e}")
            # Fallback: save basic results to current directory
            fallback_file = self.base_dir / f"evaluation_results_fallback_{timestamp}.json"
            with open(fallback_file, 'w') as f:
                json.dump([asdict(result) for result in results], f, indent=2)
            print(f"💾 Fallback results saved to: {fallback_file}")
    
    def _save_results_csv(self, results: List[EnhancedEvaluationResult], csv_file: Path):
        """Save results summary as CSV"""
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'combo_id', 'timestamp', 'triage_accuracy', 'f1_score', 'f2_score',
                'next_step_quality', 'memory_usage_mb', 'inference_speed_tps',
                'latency_ms', 'llm_judge_used', 'error_message'
            ])
            
            # Data rows
            for result in results:
                writer.writerow([
                    result.combo_id, result.timestamp, result.triage_accuracy,
                    result.f1_score, result.f2_score, result.next_step_quality,
                    result.memory_usage_mb, result.inference_speed_tps,
                    result.latency_ms, result.llm_judge_used, result.error_message
                ])
    
    def _save_top_performers(self, results: List[EnhancedEvaluationResult], json_file: Path):
        """Save top performing configurations"""
        valid_results = [r for r in results if r.error_message is None]
        
        if not valid_results:
            return
        
        # Top performers by different metrics
        top_performers = {
            "top_f2_score": sorted(valid_results, key=lambda x: x.f2_score, reverse=True)[:10],
            "top_accuracy": sorted(valid_results, key=lambda x: x.triage_accuracy, reverse=True)[:10],
            "top_next_step_quality": sorted(valid_results, key=lambda x: x.next_step_quality, reverse=True)[:10],
            "fastest_inference": sorted(valid_results, key=lambda x: x.latency_ms)[:10],
            "lowest_memory": sorted(valid_results, key=lambda x: x.memory_usage_mb)[:10]
        }
        
        # Convert to serializable format
        serializable_data = {}
        for category, performers in top_performers.items():
            serializable_data[category] = [asdict(p) for p in performers]
        
        with open(json_file, 'w') as f:
            json.dump(serializable_data, f, indent=2)
    
    def _save_configuration_analysis(self, results: List[EnhancedEvaluationResult], json_file: Path):
        """Save configuration performance analysis"""
        valid_results = [r for r in results if r.error_message is None]
        
        if not valid_results:
            return
        
        # Analyze by configuration components
        chunk_performance = {}
        model_performance = {}
        
        for result in valid_results:
            combo_id = result.combo_id
            
            # Extract chunk info from combo_id (e.g., "S360_1_C5_hash")
            if "_C5_" in combo_id:
                chunk_key = "5_chunks"
            elif "_C10_" in combo_id:
                chunk_key = "10_chunks"
            else:
                chunk_key = "unknown_chunks"
            
            # Extract model info from combo_id
            if combo_id.startswith("S360"):
                model_key = "SmolLM2-360M"
            elif combo_id.startswith("G270"):
                model_key = "Gemma-270M"
            elif combo_id.startswith("S135"):
                model_key = "SmolLM2-135M"
            else:
                model_key = "unknown_model"
            
            # Aggregate performance
            if chunk_key not in chunk_performance:
                chunk_performance[chunk_key] = []
            if model_key not in model_performance:
                model_performance[model_key] = []
            
            chunk_performance[chunk_key].append(result.f2_score)
            model_performance[model_key].append(result.f2_score)
        
        # Calculate averages
        chunk_averages = {k: np.mean(v) for k, v in chunk_performance.items()}
        model_averages = {k: np.mean(v) for k, v in model_performance.items()}
        
        analysis = {
            "chunk_performance": chunk_averages,
            "model_performance": model_averages,
            "total_configurations": len(results),
            "successful_configurations": len(valid_results),
            "failure_rate": (len(results) - len(valid_results)) / len(results) if results else 0
        }
        
        with open(json_file, 'w') as f:
            json.dump(analysis, f, indent=2)
    
    def _save_evaluation_metadata(self, results: List[EnhancedEvaluationResult], json_file: Path):
        """Save evaluation session metadata"""
        metadata = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_configurations": len(results),
            "successful_evaluations": len([r for r in results if r.error_message is None]),
            "clinical_judge_enabled": self.enable_clinical_judge,
            "framework_version": "enhanced_v2",
            "evaluation_dataset": "simplified_triage_dialogues_test.json",
            "prompt_format": "training_compatible",
            "chunk_variants": ["5_chunks", "10_chunks"],
            "metrics_calculated": ["triage_accuracy", "f1_score", "f2_score", "next_step_quality"],
            "base_directory": str(self.base_dir)
        }
        
        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _print_progress_summary(self, results: List[EnhancedEvaluationResult]):
        """Print progress summary"""
        if not results:
            return
        
        valid_results = [r for r in results if r.error_message is None]
        if not valid_results:
            return
        
        avg_accuracy = np.mean([r.triage_accuracy for r in valid_results])
        avg_f1 = np.mean([r.f1_score for r in valid_results])
        avg_f2 = np.mean([r.f2_score for r in valid_results])
        avg_next_step = np.mean([r.next_step_quality for r in valid_results])
        llm_used_count = sum(1 for r in valid_results if r.llm_judge_used)
        
        print(f"📈 Progress Summary:")
        print(f"   Completed: {len(valid_results)}/{len(results)}")
        print(f"   Avg Accuracy: {avg_accuracy:.3f}, F1: {avg_f1:.3f}, F2: {avg_f2:.3f}")
        print(f"   Avg Next Step Quality: {avg_next_step:.1f}/10")
        print(f"   LLM Judge Used: {llm_used_count}/{len(valid_results)}")

def main():
    """Test enhanced evaluation pipeline"""
    print("🧪 Testing Enhanced Evaluation Pipeline with TinfoilAgent")
    
    # Create mock evaluation combo for testing
    class MockCombo:
        def __init__(self):
            self.combo_id = "test_S360_1_mock"
            self.rag_config = {
                'chunking_method': 'structured_agent_tinfoil_medical',
                'retrieval_type': 'contextual_rag',
                'bias_config': 'diverse',
                'pass_at_5': 0.595
            }
            self.adapter_config = type('AdapterConfig', (), {
                'adapter_path': '/mock/adapter/path',
                'model_name': 'SmolLM2-135M',
                'adapter_type': 'safety'
            })()
    
    # Initialize enhanced pipeline
    pipeline = EnhancedEvaluationPipeline(
        base_dir="/Users/choemanseung/789/hft",
        enable_clinical_judge=True
    )
    
    # Test single combination evaluation
    test_combo = MockCombo()
    
    try:
        result = pipeline.evaluate_combination_with_clinical_assessment(test_combo)
        
        print(f"\n✅ Test Result:")
        print(f"   Combo ID: {result.combo_id}")
        print(f"   Triage Accuracy: {result.triage_accuracy:.3f}")
        print(f"   Clinical Appropriateness: {result.clinical_appropriateness:.1f}/10")
        print(f"   Clinical Safety: {result.clinical_safety:.1f}/10")
        print(f"   LLM Judge Used: {result.llm_judge_used}")
        print(f"   Rationale: {result.clinical_rationale[:100]}...")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()