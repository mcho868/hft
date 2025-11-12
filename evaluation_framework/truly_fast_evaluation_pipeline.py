#!/usr/bin/env python3
"""
Truly Fast Evaluation Pipeline that actually reuses loaded components
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import psutil
from sklearn.metrics import f1_score, fbeta_score, classification_report
import concurrent.futures
import threading
from functools import partial
import random
from collections import defaultdict

# Set random seed for reproducible sampling
random.seed(42)

# MLX imports with fallback
try:
    from mlx_lm import load, generate
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    print("⚠️  MLX not available. Using mock implementations for testing.")
    MLX_AVAILABLE = False
    
    def load(*args, **kwargs):
        class MockModel:
            pass
        class MockTokenizer:
            pass
        return MockModel(), MockTokenizer()
    
    def generate(model, tokenizer, prompt, max_tokens=200, verbose=False):
        if "chest pain" in prompt.lower():
            return "ED - Emergency department evaluation recommended for chest pain"
        elif "headache" in prompt.lower():
            return "HOME - Rest and over-the-counter medication recommended"
        else:
            return "GP - General practitioner consultation recommended"

# Import classes from other modules
from optimized_config_generator import EvaluationCombo, AdapterConfig
from integrated_rag_system import IntegratedRAGSystem, ValidatedRAGConfigProcessor
from appropriateness_judge import ClinicalAppropriatenessJudge, create_clinical_case_from_evaluation
from enhanced_evaluation_pipeline import EnhancedEvaluationResult

@dataclass
class ModelGroup:
    """Group of configurations that share the same base model and adapter type"""
    model_name: str
    adapter_type: str
    adapter_path: str
    configurations: List[EvaluationCombo]

class TrulyFastEvaluationPipeline:
    """Evaluation pipeline that truly reuses loaded components"""
    
    def __init__(self, base_dir: str = "/Users/choemanseung/789/hft", 
                 llm_judge_workers: int = 30):
        self.base_dir = Path(base_dir)
        self.llm_judge_workers = llm_judge_workers
        
        # Load test dataset once
        self.test_dataset = self._load_test_dataset()
        
        # Initialize clinical judge once
        self.clinical_judge = ClinicalAppropriatenessJudge(
            judge_model="llama3-3-70b",
            use_tinfoil=True,
            rate_limit_delay=0.5
        )
        print("✅ Clinical appropriateness judge initialized")
        
        # Initialize RAG system once (reused for all configs)
        self.rag_system = None
        self.rag_config_processor = None
        
        # Current loaded model cache
        self.current_model = None
        self.current_tokenizer = None
        self.current_adapter_path = None
        
    def _load_test_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset for evaluation (validation set for hyperparameter tuning)"""
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
                        "input": item["query"],
                        "triage_decision": item["final_triage_decision"],
                        "next_steps": item["next_step"],
                        "case_id": str(item["id"]),
                        "symptom": item.get("symptom", "unknown"),
                        "reasoning": item.get("reasoning", "")
                    }
                    converted_data.append(case)
                
                subset_size = int(os.getenv("EVAL_SUBSET_SIZE", "200"))
                
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
        
        # Fallback to mock data
        print("📝 Using mock test data as fallback")
        return self._generate_mock_test_data()
    
    def _stratified_sample(self, data: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
        """Create stratified sample maintaining ED/GP/HOME ratios"""
        grouped = defaultdict(list)
        for item in data:
            grouped[item["triage_decision"]].append(item)
        
        total_cases = len(data)
        ratios = {decision: len(cases) / total_cases for decision, cases in grouped.items()}
        
        target_counts = {}
        for decision, ratio in ratios.items():
            target_counts[decision] = max(1, int(sample_size * ratio))
        
        total_target = sum(target_counts.values())
        if total_target != sample_size:
            largest_group = max(target_counts.keys(), key=lambda k: target_counts[k])
            target_counts[largest_group] += sample_size - total_target
        
        sampled_data = []
        for decision, target_count in target_counts.items():
            available = grouped[decision]
            if len(available) >= target_count:
                sampled = random.sample(available, target_count)
            else:
                sampled = available
            sampled_data.extend(sampled)
        
        random.shuffle(sampled_data)
        
        print(f"📊 Stratified sampling summary:")
        for decision in ["ED", "GP", "HOME"]:
            original_count = len(grouped[decision])
            sampled_count = len([x for x in sampled_data if x["triage_decision"] == decision])
            original_pct = (original_count / total_cases) * 100
            sampled_pct = (sampled_count / len(sampled_data)) * 100
            print(f"   {decision}: {original_count} → {sampled_count} ({original_pct:.1f}% → {sampled_pct:.1f}%)")
        
        # DEBUG: Log first few case IDs to see if sampling is deterministic
        case_ids = [x.get("case_id", str(i)) for i, x in enumerate(sampled_data[:5])]
        print(f"🔍 DEBUG: First 5 sampled case IDs: {case_ids}")
        
        return sampled_data
    
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
        
        for i, (symptoms, triage) in enumerate(conditions * 6):  # 30 cases
            case = {
                "input": f"Patient {i+1}: {symptoms}",
                "triage_decision": triage,
                "next_steps": f"Recommended action for {triage} case",
                "case_id": f"mock_case_{i+1}"
            }
            mock_cases.append(case)
        
        return mock_cases
    
    def _initialize_rag_system_once(self):
        """Initialize RAG system once and reuse it"""
        if self.rag_system is None:
            print("🔄 Initializing RAG system (one time setup)...")
            self.rag_system = IntegratedRAGSystem()
            self.rag_config_processor = ValidatedRAGConfigProcessor()
            print("✅ RAG system initialized and cached")
    
    def _load_model_if_different(self, adapter_config: AdapterConfig) -> Tuple[Any, Any]:
        """Load model only if it's different from currently loaded model"""
        if (self.current_adapter_path == adapter_config.adapter_path and 
            self.current_model is not None):
            print(f"♻️  Reusing already loaded model: {adapter_config.model_name}")
            return self.current_model, self.current_tokenizer
        
        print(f"🔄 Loading new model: {adapter_config.model_name} with adapter: {adapter_config.adapter_path}")
        
        if not MLX_AVAILABLE:
            return load(), load()[1]
        
        try:
            # Model mapping (simplified)
            model_mapping = {
                "SmolLM2-360M": "/Users/choemanseung/789/hft/mlx_models/SmolLM2-360M-Instruct-MLX_4bit",
                "SmolLM2-135M": "/Users/choemanseung/789/hft/mlx_models/SmolLM2-135M-Instruct-MLX_4bit",
                "Gemma-270M": "/Users/choemanseung/789/hft/mlx_models/gemma-270m-mlx_4bit"
            }
            
            base_model_path = model_mapping.get(adapter_config.model_name)
            if not base_model_path:
                raise ValueError(f"Unknown model: {adapter_config.model_name}")
            
            model, tokenizer = load(base_model_path, adapter_path=adapter_config.adapter_path)
            
            # Cache the loaded model
            self.current_model = model
            self.current_tokenizer = tokenizer
            self.current_adapter_path = adapter_config.adapter_path
            
            print(f"✅ Model loaded and cached")
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return load(), load()[1]
    
    def _extract_triage_decision(self, response: str) -> str:
        """Extract triage decision from model response"""
        # First, try to extract from the structured format: "Triage Decision: ED"
        import re
        triage_match = re.search(r'Triage Decision:\s*(ED|GP|HOME)', response, re.IGNORECASE)
        if triage_match:
            return triage_match.group(1).upper()
        
        # Fallback to keyword-based extraction
        response_upper = response.upper()
        
        if "ED" in response_upper or "EMERGENCY" in response_upper:
            return "ED"
        elif "HOME" in response_upper or "SELF-CARE" in response_upper:
            return "HOME"
        elif "GP" in response_upper or "GENERAL PRACTITIONER" in response_upper:
            return "GP"
        
        if any(word in response_upper for word in ["URGENT", "SEVERE", "CHEST PAIN"]):
            return "ED"
        elif any(word in response_upper for word in ["MILD", "REST", "MONITOR"]):
            return "HOME"
        else:
            return "GP"
    
    def _evaluate_single_case(self, model, tokenizer, rag_config, test_case):
        """Evaluate a single test case"""
        try:
            # Get RAG context using cached system
            processed_config = self.rag_config_processor.process_config(rag_config)
            context = self.rag_system.retrieve_context(test_case["input"], processed_config)
            
            # DEBUG: Log RAG configuration and context details
            case_id = test_case.get("case_id", "unknown")
            print(f"🔍 DEBUG [{case_id}]: RAG Config - chunking: {rag_config.get('chunking_method', 'N/A')}, "
                  f"retrieval: {rag_config.get('retrieval_type', 'N/A')}, "
                  f"chunks: {rag_config.get('chunk_limit', 'N/A')}")
            
            # Log context length and first 100 chars
            context_preview = context[:100].replace('\n', ' ') if context else "EMPTY"
            print(f"🔍 DEBUG [{case_id}]: Context length: {len(context)}, preview: {context_preview}...")
            
            # Create prompt
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
            
            # DEBUG: Log model response
            response_preview = response[:150].replace('\n', ' ') if response else "EMPTY"
            print(f"🔍 DEBUG [{case_id}]: Response preview: {response_preview}...")
            
            # Extract prediction
            predicted_triage = self._extract_triage_decision(response)
            
            # DEBUG: Log extraction result
            print(f"🔍 DEBUG [{case_id}]: Extracted triage: {predicted_triage}, "
                  f"actual: {test_case.get('triage_decision', 'N/A')}, "
                  f"correct: {predicted_triage == test_case.get('triage_decision', 'N/A')}")
            
            return {
                "case_id": test_case["case_id"],
                "predicted_triage": predicted_triage,
                "actual_triage": test_case["triage_decision"],
                "triage_correct": predicted_triage == test_case["triage_decision"],
                "full_response": response,
                "inference_time": end_time - start_time
            }
            
        except Exception as e:
            return {
                "case_id": test_case.get("case_id", "unknown"),
                "error": str(e),
                "triage_correct": False,
                "inference_time": 0
            }
    
    def _calculate_performance_metrics(self, case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics from case results"""
        if not case_results:
            return {"triage_accuracy": 0.0, "f1_score": 0.0, "f2_score": 0.0}
        
        # Filter valid results
        valid_results = [r for r in case_results if "triage_correct" in r]
        
        if not valid_results:
            return {"triage_accuracy": 0.0, "f1_score": 0.0, "f2_score": 0.0}
        
        # Calculate accuracy
        correct_predictions = sum(1 for r in valid_results if r["triage_correct"])
        accuracy = correct_predictions / len(valid_results)
        
        # Calculate F1 and F2 scores
        try:
            predicted = [r.get("predicted_triage", "GP") for r in valid_results]
            actual = [r.get("actual_triage", "GP") for r in valid_results]
            
            # Map to numbers for sklearn
            label_map = {"ED": 0, "GP": 1, "HOME": 2}
            y_true = [label_map.get(a, 1) for a in actual]
            y_pred = [label_map.get(p, 1) for p in predicted]
            
            f1 = f1_score(y_true, y_pred, average='weighted')
            f2 = fbeta_score(y_true, y_pred, beta=2, average='weighted')
        except:
            f1 = accuracy  # Fallback
            f2 = accuracy
        
        # Calculate timing
        inference_times = [r.get("inference_time", 0) for r in valid_results]
        
        return {
            "triage_accuracy": accuracy,
            "f1_score": f1,
            "f2_score": f2,
            "avg_inference_time": np.mean(inference_times) if inference_times else 0,
            "total_inference_time": sum(inference_times)
        }
    
    def _evaluate_single_clinical_case(self, clinical_case) -> Dict[str, Any]:
        """Evaluate a single clinical case with LLM judge"""
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
    
    def _evaluate_clinical_appropriateness_parallel(self, clinical_cases: List[Any]) -> Dict[str, Any]:
        """Evaluate clinical appropriateness using parallel workers"""
        if not clinical_cases:
            return {"next_step_quality": 0.0, "rationale": "No cases to evaluate", "llm_used": False}
        
        print(f"🤖 Running clinical appropriateness evaluation on {len(clinical_cases)} cases with {self.llm_judge_workers} parallel workers...")
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.llm_judge_workers) as executor:
            future_to_case = {
                executor.submit(self._evaluate_single_clinical_case, case): case 
                for case in clinical_cases
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_case):
                result = future.result()
                results.append(result)
                completed += 1
                
                if completed % 50 == 0 or completed == len(clinical_cases):
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
    
    def evaluate_model_group(self, model_group: ModelGroup) -> List[EnhancedEvaluationResult]:
        """Evaluate all configurations in a model group (reusing loaded model)"""
        results = []
        
        print(f"🔄 Loading model for group: {model_group.model_name}_{model_group.adapter_type}")
        
        # Initialize RAG system once
        self._initialize_rag_system_once()
        
        # Load model once for this group
        dummy_adapter = AdapterConfig(
            adapter_path=model_group.adapter_path,
            model_name=model_group.model_name,
            adapter_type=model_group.adapter_type
        )
        model, tokenizer = self._load_model_if_different(dummy_adapter)
        
        print(f"✅ Model loaded. Processing {len(model_group.configurations)} configurations...")
        
        # Evaluate each configuration in the group
        for i, combo in enumerate(model_group.configurations, 1):
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            timestamp = datetime.now().isoformat()
            
            print(f"   Config {i}/{len(model_group.configurations)}: {combo.combo_id}")
            
            try:
                # Evaluate all test cases for this configuration
                case_results = []
                clinical_cases = []
                
                print(f"     Running model inference on {len(self.test_dataset)} cases...")
                for test_case in self.test_dataset:
                    case_result = self._evaluate_single_case(model, tokenizer, combo.rag_config, test_case)
                    case_results.append(case_result)
                    
                    # Create clinical case for LLM judge
                    clinical_case = create_clinical_case_from_evaluation(
                        case_id=f"{combo.combo_id}_{test_case.get('case_id', len(clinical_cases))}",
                        patient_input=test_case["input"],
                        model_triage=case_result.get("predicted_triage", "GP"),
                        model_steps=case_result.get("full_response", "Standard care recommended"),
                        true_triage=test_case.get("triage_decision", "GP"),
                        true_steps=test_case.get("next_steps", "Follow up as needed")
                    )
                    clinical_cases.append(clinical_case)
                
                # Calculate performance metrics
                metrics = self._calculate_performance_metrics(case_results)
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Calculate tokens per second
                total_tokens = len(case_results) * 200
                inference_speed = total_tokens / max(metrics["total_inference_time"], 0.001)
                
                print(f"     Model inference completed. Starting clinical evaluation...")
                # Perform parallel clinical evaluation
                clinical_scores = self._evaluate_clinical_appropriateness_parallel(clinical_cases)
                
                # Create result
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
                
                results.append(result)
                
                print(f"     ✅ Completed: Acc={metrics['triage_accuracy']:.3f}, F2={metrics['f2_score']:.3f}, Clinical={clinical_scores['next_step_quality']:.1f}")
                
            except Exception as e:
                print(f"     ❌ Error: {e}")
                error_result = EnhancedEvaluationResult(
                    combo_id=combo.combo_id,
                    timestamp=timestamp,
                    triage_accuracy=0.0,
                    f1_score=0.0,
                    f2_score=0.0,
                    next_step_quality=0.0,
                    next_step_rationale=f"Error: {e}",
                    memory_usage_mb=0.0,
                    inference_speed_tps=0.0,
                    latency_ms=0.0,
                    error_message=str(e),
                    llm_judge_used=False
                )
                results.append(error_result)
        
        print(f"✅ Model group completed: {len(results)} configurations evaluated")
        return results