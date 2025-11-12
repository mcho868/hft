#!/usr/bin/env python3
"""
Baseline Performance Evaluation Framework for Medical Triage System
Tests base model performance vs base model + RAG vs base model + LoRA adapters
Uses same prompt format as main evaluation but without LLM-as-judge scoring.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse

import numpy as np
import psutil
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, classification_report

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

@dataclass
class BaselineResult:
    """Results from a single baseline evaluation"""
    test_name: str
    model_name: str
    timestamp: str
    triage_accuracy: float
    f1_score: float
    f2_score: float
    avg_inference_time: float
    memory_usage_mb: float
    total_cases: int
    correct_predictions: int
    error_message: Optional[str] = None
    detailed_metrics: Optional[Dict[str, Any]] = None

class BaselineEvaluationFramework:
    """Framework for evaluating baseline model performance"""
    
    def __init__(self, base_dir: str = "/Users/choemanseung/789/hft"):
        self.base_dir = Path(base_dir)
        self.test_dataset = self._load_test_dataset()
        
        # Model mappings with quantization variants
        self.base_model_paths = {
            "SmolLM2-360M-4bit": "/Users/choemanseung/789/hft/mlx_models/SmolLM2-360M-Instruct-MLX_4bit",
            "SmolLM2-360M-8bit": "/Users/choemanseung/789/hft/mlx_models/SmolLM2-360M-Instruct-MLX_8bit",
            "SmolLM2-135M-4bit": "/Users/choemanseung/789/hft/mlx_models/SmolLM2-135M-Instruct-MLX_4bit",
            "SmolLM2-135M-8bit": "/Users/choemanseung/789/hft/mlx_models/SmolLM2-135M-Instruct-MLX_8bit",
            "Gemma-270M-4bit": "/Users/choemanseung/789/hft/mlx_models/gemma-270m-mlx_4bit",
            "Gemma-270M-8bit": "/Users/choemanseung/789/hft/mlx_models/gemma-270m-mlx_8bit"
        }
        
        # RAG system
        self.rag_system = None
        
        # Current loaded model cache
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        
    def _load_test_dataset(self) -> List[Dict[str, Any]]:
        """Load test dataset for evaluation"""
        test_file = self.base_dir / "Final_dataset" / "simplified_triage_dialogues_test.json"
        
        if test_file.exists():
            try:
                with open(test_file, 'r') as f:
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
                
                # Use full test set for baseline evaluation (default: all 1,975 cases)
                subset_size = int(os.getenv("BASELINE_EVAL_SIZE", str(len(converted_data))))
                print(f"📊 Loaded {len(converted_data)} real triage test cases")
                print(f"   Using {subset_size} cases for baseline evaluation")
                return converted_data[:subset_size]
                
            except Exception as e:
                print(f"Error loading real test dataset: {e}")
        else:
            print(f"⚠️  Real test file not found: {test_file}")
        
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
        
        for i, (symptoms, triage) in enumerate(conditions * 395):  # 1,975 cases (same as real dataset)
            case = {
                "input": f"Patient {i+1}: {symptoms}",
                "triage_decision": triage,
                "next_steps": f"Recommended action for {triage} case",
                "case_id": f"mock_case_{i+1}"
            }
            mock_cases.append(case)
        
        return mock_cases
    
    def _load_base_model(self, model_name: str) -> Tuple[Any, Any]:
        """Load base model without adapters"""
        if self.current_model_name == model_name and self.current_model is not None:
            return self.current_model, self.current_tokenizer
        
        if model_name not in self.base_model_paths:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_path = self.base_model_paths[model_name]
        print(f"Loading base model: {model_name} from {model_path}")
        
        if not MLX_AVAILABLE:
            return load(), load()[1]
        
        try:
            model, tokenizer = load(model_path)
            self.current_model = model
            self.current_tokenizer = tokenizer
            self.current_model_name = model_name
            return model, tokenizer
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return load(), load()[1]  # Return mock objects
    
    def _load_model_with_adapter(self, model_name: str, adapter_path: str) -> Tuple[Any, Any]:
        """Load model with LoRA adapter"""
        print(f"Loading model: {model_name} with adapter: {adapter_path}")
        
        if not MLX_AVAILABLE:
            return load(), load()[1]
        
        try:
            if model_name in self.base_model_paths:
                base_model_path = self.base_model_paths[model_name]
            else:
                # Try to match base model name with quantization
                base_name = model_name.split('-')[0] + "-" + model_name.split('-')[1]  # e.g., SmolLM2-360M
                possible_paths = [k for k in self.base_model_paths.keys() if base_name in k]
                if possible_paths:
                    base_model_path = self.base_model_paths[possible_paths[0]]  # Use first match
                else:
                    raise ValueError(f"No matching base model path for {model_name}")
            
            model, tokenizer = load(base_model_path, adapter_path=adapter_path)
            return model, tokenizer
        except Exception as e:
            print(f"Error loading model with adapter: {e}")
            return load(), load()[1]  # Return mock objects
    
    def _initialize_rag_system(self):
        """Initialize RAG system for base model + RAG evaluation"""
        if self.rag_system is None:
            try:
                from integrated_rag_system import IntegratedRAGSystem
                self.rag_system = IntegratedRAGSystem()
                print("✅ RAG system initialized")
            except ImportError as e:
                print(f"⚠️  Could not import RAG system: {e}")
                self.rag_system = None
    
    def _create_base_prompt(self, test_case: Dict[str, Any]) -> str:
        """Create base prompt without RAG context (same format as main evaluation)"""
        return f"""Patient query: {test_case["input"]}

Provide triage decision, next steps, and reasoning:"""
    
    def _create_rag_prompt(self, test_case: Dict[str, Any]) -> str:
        """Create prompt with RAG context"""
        if self.rag_system is None:
            return self._create_base_prompt(test_case)
        
        try:
            # Use a basic RAG configuration
            rag_config = {
                'chunking_method': 'semantic_split',
                'retrieval_type': 'bm25_semantic_hybrid',
                'chunk_count': 5
            }
            context = self.rag_system.retrieve_context(test_case["input"], rag_config)
            
            return f"""Patient query: {test_case["input"]}

Context:
{context}

Provide triage decision, next steps, and reasoning:"""
        except Exception as e:
            print(f"RAG retrieval error: {e}")
            return self._create_base_prompt(test_case)
    
    def _extract_triage_decision(self, response: str) -> str:
        """Extract triage decision from model response"""
        response_upper = response.upper()
        
        # Look for explicit triage decisions
        if "ED" in response_upper or "EMERGENCY" in response_upper:
            return "ED"
        elif "HOME" in response_upper or "SELF-CARE" in response_upper:
            return "HOME"
        elif "GP" in response_upper or "GENERAL PRACTITIONER" in response_upper:
            return "GP"
        
        # Fallback: check for keywords
        if any(word in response_upper for word in ["URGENT", "SEVERE", "CHEST PAIN"]):
            return "ED"
        elif any(word in response_upper for word in ["MILD", "REST", "MONITOR"]):
            return "HOME"
        else:
            return "GP"  # Default
    
    def _calculate_metrics(self, predictions: List[str], ground_truth: List[str]) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        # Map triage decisions to numbers for sklearn
        label_map = {"ED": 0, "GP": 1, "HOME": 2}
        
        y_true = [label_map.get(gt, 1) for gt in ground_truth]  # Default to GP
        y_pred = [label_map.get(pred, 1) for pred in predictions]
        
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        f2 = fbeta_score(y_true, y_pred, beta=2, average='weighted')  # Prioritizes recall
        
        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "f2_score": f2
        }
    
    def evaluate_base_model(self, model_name: str) -> BaselineResult:
        """Evaluate base model performance without any enhancements"""
        print(f"\n{'='*60}")
        print(f"EVALUATING BASE MODEL: {model_name}")
        print(f"{'='*60}")
        
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # Load base model
            model, tokenizer = self._load_base_model(model_name)
            
            predictions = []
            ground_truth = []
            inference_times = []
            
            print(f"Processing {len(self.test_dataset)} test cases...")
            
            for i, test_case in enumerate(self.test_dataset):
                if i % 10 == 0:
                    print(f"  Progress: {i}/{len(self.test_dataset)}")
                
                # Create base prompt
                prompt = self._create_base_prompt(test_case)
                
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
                
                # Extract prediction
                predicted_triage = self._extract_triage_decision(response)
                
                predictions.append(predicted_triage)
                ground_truth.append(test_case["triage_decision"])
                inference_times.append(end_time - start_time)
            
            # Calculate metrics
            metrics = self._calculate_metrics(predictions, ground_truth)
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Count correct predictions
            correct_predictions = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
            
            result = BaselineResult(
                test_name="base_model",
                model_name=model_name,
                timestamp=datetime.now().isoformat(),
                triage_accuracy=metrics["accuracy"],
                f1_score=metrics["f1_score"],
                f2_score=metrics["f2_score"],
                avg_inference_time=np.mean(inference_times),
                memory_usage_mb=end_memory - start_memory,
                total_cases=len(self.test_dataset),
                correct_predictions=correct_predictions,
                detailed_metrics={
                    "predictions": predictions,
                    "ground_truth": ground_truth,
                    "inference_times": inference_times
                }
            )
            
            print(f"✅ Base model evaluation completed:")
            print(f"   Accuracy: {metrics['accuracy']:.3f}")
            print(f"   F1 Score: {metrics['f1_score']:.3f}")
            print(f"   F2 Score: {metrics['f2_score']:.3f}")
            print(f"   Avg Inference Time: {np.mean(inference_times):.3f}s")
            
            return result
            
        except Exception as e:
            return BaselineResult(
                test_name="base_model",
                model_name=model_name,
                timestamp=datetime.now().isoformat(),
                triage_accuracy=0.0,
                f1_score=0.0,
                f2_score=0.0,
                avg_inference_time=0.0,
                memory_usage_mb=0.0,
                total_cases=0,
                correct_predictions=0,
                error_message=str(e)
            )
    
    def evaluate_base_model_with_rag(self, model_name: str) -> BaselineResult:
        """Evaluate base model + RAG performance"""
        print(f"\n{'='*60}")
        print(f"EVALUATING BASE MODEL + RAG: {model_name}")
        print(f"{'='*60}")
        
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # Initialize RAG system
            self._initialize_rag_system()
            
            # Load base model
            model, tokenizer = self._load_base_model(model_name)
            
            predictions = []
            ground_truth = []
            inference_times = []
            
            print(f"Processing {len(self.test_dataset)} test cases with RAG...")
            
            for i, test_case in enumerate(self.test_dataset):
                if i % 10 == 0:
                    print(f"  Progress: {i}/{len(self.test_dataset)}")
                
                # Create RAG-enhanced prompt
                prompt = self._create_rag_prompt(test_case)
                
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
                
                # Extract prediction
                predicted_triage = self._extract_triage_decision(response)
                
                predictions.append(predicted_triage)
                ground_truth.append(test_case["triage_decision"])
                inference_times.append(end_time - start_time)
            
            # Calculate metrics
            metrics = self._calculate_metrics(predictions, ground_truth)
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            correct_predictions = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
            
            result = BaselineResult(
                test_name="base_model_plus_rag",
                model_name=model_name,
                timestamp=datetime.now().isoformat(),
                triage_accuracy=metrics["accuracy"],
                f1_score=metrics["f1_score"],
                f2_score=metrics["f2_score"],
                avg_inference_time=np.mean(inference_times),
                memory_usage_mb=end_memory - start_memory,
                total_cases=len(self.test_dataset),
                correct_predictions=correct_predictions,
                detailed_metrics={
                    "predictions": predictions,
                    "ground_truth": ground_truth,
                    "inference_times": inference_times
                }
            )
            
            print(f"✅ Base model + RAG evaluation completed:")
            print(f"   Accuracy: {metrics['accuracy']:.3f}")
            print(f"   F1 Score: {metrics['f1_score']:.3f}")
            print(f"   F2 Score: {metrics['f2_score']:.3f}")
            print(f"   Avg Inference Time: {np.mean(inference_times):.3f}s")
            
            return result
            
        except Exception as e:
            return BaselineResult(
                test_name="base_model_plus_rag",
                model_name=model_name,
                timestamp=datetime.now().isoformat(),
                triage_accuracy=0.0,
                f1_score=0.0,
                f2_score=0.0,
                avg_inference_time=0.0,
                memory_usage_mb=0.0,
                total_cases=0,
                correct_predictions=0,
                error_message=str(e)
            )
    
    def evaluate_base_model_with_adapter(self, model_name: str, adapter_path: str) -> BaselineResult:
        """Evaluate base model + LoRA adapter performance"""
        adapter_name = Path(adapter_path).name
        print(f"\n{'='*60}")
        print(f"EVALUATING BASE MODEL + LORA: {model_name}")
        print(f"Adapter: {adapter_name}")
        print(f"{'='*60}")
        
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # Load model with adapter
            model, tokenizer = self._load_model_with_adapter(model_name, adapter_path)
            
            predictions = []
            ground_truth = []
            inference_times = []
            
            print(f"Processing {len(self.test_dataset)} test cases with LoRA adapter...")
            
            for i, test_case in enumerate(self.test_dataset):
                if i % 10 == 0:
                    print(f"  Progress: {i}/{len(self.test_dataset)}")
                
                # Create base prompt (LoRA should enhance the model's capabilities)
                prompt = self._create_base_prompt(test_case)
                
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
                
                # Extract prediction
                predicted_triage = self._extract_triage_decision(response)
                
                predictions.append(predicted_triage)
                ground_truth.append(test_case["triage_decision"])
                inference_times.append(end_time - start_time)
            
            # Calculate metrics
            metrics = self._calculate_metrics(predictions, ground_truth)
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            correct_predictions = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
            
            result = BaselineResult(
                test_name="base_model_plus_lora",
                model_name=f"{model_name}+{adapter_name}",
                timestamp=datetime.now().isoformat(),
                triage_accuracy=metrics["accuracy"],
                f1_score=metrics["f1_score"],
                f2_score=metrics["f2_score"],
                avg_inference_time=np.mean(inference_times),
                memory_usage_mb=end_memory - start_memory,
                total_cases=len(self.test_dataset),
                correct_predictions=correct_predictions,
                detailed_metrics={
                    "adapter_path": adapter_path,
                    "predictions": predictions,
                    "ground_truth": ground_truth,
                    "inference_times": inference_times
                }
            )
            
            print(f"✅ Base model + LoRA evaluation completed:")
            print(f"   Accuracy: {metrics['accuracy']:.3f}")
            print(f"   F1 Score: {metrics['f1_score']:.3f}")
            print(f"   F2 Score: {metrics['f2_score']:.3f}")
            print(f"   Avg Inference Time: {np.mean(inference_times):.3f}s")
            
            return result
            
        except Exception as e:
            return BaselineResult(
                test_name="base_model_plus_lora",
                model_name=f"{model_name}+{adapter_name}",
                timestamp=datetime.now().isoformat(),
                triage_accuracy=0.0,
                f1_score=0.0,
                f2_score=0.0,
                avg_inference_time=0.0,
                memory_usage_mb=0.0,
                total_cases=0,
                correct_predictions=0,
                error_message=str(e)
            )
    
    def run_comprehensive_baseline_evaluation(self, models: List[str] = None, 
                                            adapter_paths: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive baseline evaluation"""
        print("🚀 MEDICAL TRIAGE BASELINE PERFORMANCE EVALUATION")
        print("="*80)
        
        if models is None:
            models = list(self.base_model_paths.keys())
        
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.base_dir / "evaluation_framework" / f"baseline_evaluation_{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Session directory: {session_dir}")
        print(f"📊 Test cases: {len(self.test_dataset)}")
        print(f"🤖 Models to test: {', '.join(models)}")
        
        all_results = []
        
        # Test each model in three configurations
        for model_name in models:
            print(f"\n🔬 Testing model: {model_name}")
            
            # 1. Base model only
            result = self.evaluate_base_model(model_name)
            all_results.append(result)
            
            # 2. Base model + RAG
            result = self.evaluate_base_model_with_rag(model_name)
            all_results.append(result)
            
            # 3. Base model + LoRA adapters (if provided)
            if adapter_paths:
                for adapter_path in adapter_paths:
                    if model_name.lower() in adapter_path.lower():  # Match adapter to model
                        result = self.evaluate_base_model_with_adapter(model_name, adapter_path)
                        all_results.append(result)
        
        # Save results
        results_file = session_dir / "baseline_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(result) for result in all_results], f, indent=2)
        
        # Generate analysis
        analysis = self._generate_baseline_analysis(all_results)
        analysis_file = session_dir / "baseline_analysis_report.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Print summary
        self._print_baseline_summary(all_results)
        
        print(f"\n📁 Results saved to: {session_dir}")
        print(f"📊 Full results: {results_file}")
        print(f"📈 Analysis report: {analysis_file}")
        
        return {
            "session_id": session_id,
            "session_dir": str(session_dir),
            "results": all_results,
            "analysis": analysis,
            "results_file": str(results_file),
            "analysis_file": str(analysis_file)
        }
    
    def _generate_baseline_analysis(self, results: List[BaselineResult]) -> Dict[str, Any]:
        """Generate analysis report from baseline results"""
        if not results:
            return {"error": "No results to analyze"}
        
        # Group results by test type
        by_test_type = {}
        for result in results:
            test_type = result.test_name
            if test_type not in by_test_type:
                by_test_type[test_type] = []
            by_test_type[test_type].append(result)
        
        # Calculate averages for each test type
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "total_results": len(results),
            "test_types": list(by_test_type.keys()),
            "summary_by_test_type": {}
        }
        
        for test_type, test_results in by_test_type.items():
            valid_results = [r for r in test_results if r.error_message is None]
            
            if valid_results:
                analysis["summary_by_test_type"][test_type] = {
                    "count": len(valid_results),
                    "avg_accuracy": np.mean([r.triage_accuracy for r in valid_results]),
                    "avg_f1_score": np.mean([r.f1_score for r in valid_results]),
                    "avg_f2_score": np.mean([r.f2_score for r in valid_results]),
                    "avg_inference_time": np.mean([r.avg_inference_time for r in valid_results]),
                    "best_performer": max(valid_results, key=lambda x: x.f2_score).model_name
                }
        
        # Find overall best performers
        valid_results = [r for r in results if r.error_message is None]
        if valid_results:
            analysis["best_performers"] = {
                "by_accuracy": max(valid_results, key=lambda x: x.triage_accuracy).model_name,
                "by_f1_score": max(valid_results, key=lambda x: x.f1_score).model_name,
                "by_f2_score": max(valid_results, key=lambda x: x.f2_score).model_name,
                "by_speed": min(valid_results, key=lambda x: x.avg_inference_time).model_name
            }
        
        return analysis
    
    def _print_baseline_summary(self, results: List[BaselineResult]):
        """Print baseline evaluation summary"""
        print("\n" + "="*80)
        print("BASELINE EVALUATION SUMMARY")
        print("="*80)
        
        # Group by test type
        by_test_type = {}
        for result in results:
            test_type = result.test_name
            if test_type not in by_test_type:
                by_test_type[test_type] = []
            by_test_type[test_type].append(result)
        
        for test_type, test_results in by_test_type.items():
            print(f"\n📊 {test_type.replace('_', ' ').title()}:")
            
            valid_results = [r for r in test_results if r.error_message is None]
            if not valid_results:
                print("   No valid results")
                continue
            
            # Sort by F2 score (medical priority)
            sorted_results = sorted(valid_results, key=lambda x: x.f2_score, reverse=True)
            
            for i, result in enumerate(sorted_results, 1):
                print(f"   {i}. {result.model_name}")
                print(f"      Accuracy: {result.triage_accuracy:.3f}")
                print(f"      F1 Score: {result.f1_score:.3f}")
                print(f"      F2 Score: {result.f2_score:.3f} (Medical Priority)")
                print(f"      Inference: {result.avg_inference_time:.3f}s/case")
                print(f"      Memory: {result.memory_usage_mb:+.1f}MB")
        
        # Overall comparison
        valid_results = [r for r in results if r.error_message is None]
        if valid_results:
            best_f2 = max(valid_results, key=lambda x: x.f2_score)
            print(f"\n🏆 BEST OVERALL PERFORMER (F2 Score):")
            print(f"   {best_f2.model_name} ({best_f2.test_name})")
            print(f"   F2 Score: {best_f2.f2_score:.3f}")
            print(f"   Accuracy: {best_f2.triage_accuracy:.3f}")
            print(f"   Correct: {best_f2.correct_predictions}/{best_f2.total_cases}")

def main():
    """Main execution for baseline evaluation"""
    parser = argparse.ArgumentParser(description="Medical Triage System - Baseline Performance Evaluation")
    
    parser.add_argument("--base-dir", default="/Users/choemanseung/789/hft",
                       help="Base directory for evaluation")
    parser.add_argument("--models", nargs="+", 
                       choices=["SmolLM2-360M-4bit", "SmolLM2-360M-8bit", "SmolLM2-135M-4bit", 
                               "SmolLM2-135M-8bit", "Gemma-270M-4bit", "Gemma-270M-8bit"],
                       help="Models to evaluate (default: all quantization variants)")
    parser.add_argument("--adapter-paths", nargs="+",
                       help="Paths to LoRA adapter directories")
    parser.add_argument("--eval-size", type=int, default=1975,
                       help="Number of test cases to evaluate (default: full test set)")
    
    args = parser.parse_args()
    
    # Set evaluation size
    os.environ["BASELINE_EVAL_SIZE"] = str(args.eval_size)
    
    # Initialize framework
    framework = BaselineEvaluationFramework(args.base_dir)
    
    try:
        results = framework.run_comprehensive_baseline_evaluation(
            models=args.models,
            adapter_paths=args.adapter_paths
        )
        
        print("\n✅ Baseline evaluation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Baseline evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())