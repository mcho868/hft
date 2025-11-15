#!/usr/bin/env python3
"""
Core testing functions for the TOP 5 medical triage configurations with LLM-as-Judge evaluation.
Contains model loading, inference, RAG integration logic, and LLM quality evaluation for final testing.
"""

import re
import time
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, fbeta_score, classification_report, confusion_matrix

# Import LLM Quality Judge and Parallel Manager
from llm_quality_judge import MedicalTriageQualityJudge, ReasoningQualityScores
from parallel_llm_judge import ParallelLLMJudgeManager

logger = logging.getLogger(__name__)

class TriageInferenceEngine:
    """Handles model loading, RAG context retrieval, triage inference, and parallel LLM quality evaluation"""
    
    def __init__(self, retriever=None, enable_llm_judge: bool = True, judge_model: str = "llama3-3-70b", max_judge_workers: int = 50):
        self.retriever = retriever
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_path = None
        self.current_adapter_path = None
        
        # Initialize Parallel LLM Quality Judge
        self.enable_llm_judge = enable_llm_judge
        self.max_judge_workers = max_judge_workers
        if enable_llm_judge:
            logger.info(f"🧑‍⚕️ Initializing Parallel LLM Quality Judge with {judge_model} ({max_judge_workers} workers)")
            self.judge_manager = None  # Will be initialized when needed
            self.judge_model = judge_model
        else:
            self.judge_manager = None
            logger.info("🚫 LLM Quality Judge disabled")
    
    def load_model_if_needed(self, model_path: str, adapter_path: Optional[str] = None) -> Tuple[Any, Any]:
        """Load model and tokenizer, reusing if already loaded"""
        
        # Check if we need to reload
        if (self.current_model is not None and 
            self.current_model_path == model_path and 
            self.current_adapter_path == adapter_path):
            logger.info(f"♻️  Reusing loaded model: {Path(model_path).name}")
            return self.current_model, self.current_tokenizer
        
        logger.info(f"🔄 Loading model: {Path(model_path).name}")
        if adapter_path:
            logger.info(f"🔄 With adapter: {Path(adapter_path).name}")
        
        try:
            from mlx_lm import load
            
            if adapter_path:
                model, tokenizer = load(model_path, adapter_path=adapter_path)
            else:
                model, tokenizer = load(model_path)
            
            # Cache the loaded model
            self.current_model = model
            self.current_tokenizer = tokenizer
            self.current_model_path = model_path
            self.current_adapter_path = adapter_path
            
            logger.info(f"✅ Model loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def get_rag_context(self, query: str, rag_config: Dict[str, Any]) -> Tuple[str, float]:
        """Retrieve RAG context using optimized retrieval system"""
        
        if not self.retriever or not rag_config:
            return "", 0.0
        
        start_time = time.time()
        
        try:
            chunking_method = rag_config["chunking_method"]
            retrieval_type = rag_config["retrieval_type"] 
            bias_config_name = rag_config["bias_config"]
            
            # Create sources config for the retriever
            if bias_config_name == "diverse":
                sources_config = {"healthify": 3, "mayo": 3, "nhs": 4}
            elif bias_config_name == "healthify_focused":
                sources_config = {"healthify": 5, "mayo": 2, "nhs": 3}
            else:
                sources_config = {"healthify": 3, "mayo": 3, "nhs": 4}
            
            # Create bias config from sources_config
            from optimized_hybrid_evaluator import BiasConfig
            bias_config = BiasConfig(
                name=bias_config_name,
                healthify=sources_config.get("healthify", 3),
                mayo=sources_config.get("mayo", 3), 
                nhs=sources_config.get("nhs", 4),
                description=f"Dynamic bias config: {bias_config_name}"
            )
            
            # Get retrieval results (single query)
            results_batch = self.retriever.search_batch_with_bias(
                queries=[query],
                bias_config=bias_config,
                chunking_method=chunking_method,
                retrieval_type=retrieval_type
            )
            results = results_batch[0] if results_batch else []
            
            # Format context from results
            context_parts = []
            for i, result in enumerate(results[:10]):  # Top 10 results
                source = result.get('source', 'unknown')
                content = result.get('text', '')
                if content:
                    context_parts.append(f"[{i+1}] ({source}) {content}")
            
            context = "\n".join(context_parts)
            retrieval_time = time.time() - start_time
            
            # Only log RAG details occasionally to avoid spam
            if len(context) == 0:
                logger.warning(f"⚠️ No RAG context retrieved for query")
            elif hasattr(self, '_rag_log_counter'):
                self._rag_log_counter = getattr(self, '_rag_log_counter', 0) + 1
                if self._rag_log_counter % 50 == 0:  # Log every 50th retrieval
                    logger.info(f"🔍 RAG retrieval progress: {len(context)} chars in {retrieval_time:.3f}s")
            else:
                self._rag_log_counter = 1
                logger.debug(f"🔍 RAG retrieval: {len(context)} chars in {retrieval_time:.3f}s")
            return context, retrieval_time
            
        except Exception as e:
            logger.warning(f"⚠️  RAG retrieval failed: {e}")
            return "", time.time() - start_time
    
    def create_prompt(self, query: str, context: str = "") -> str:
        """Create formatted prompt for triage inference"""
        
        if context:
            prompt = f"""Patient query: {query}

Context:
{context}

Provide triage decision, next steps, and reasoning:"""
        else:
            prompt = f"""Patient query: {query}

Provide triage decision, next steps, and reasoning:"""
        
        return prompt
    
    def generate_response(self, model, tokenizer, prompt: str, max_tokens: int = 200) -> str:
        """Generate model response"""
        
        try:
            from mlx_lm import generate
            
            response = generate(
                model=model,
                tokenizer=tokenizer, 
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False
            )
            
            # Log the actual response for debugging
            logger.debug(f"🔍 Generated response: {repr(response[:200])}")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return ""
    
    def extract_triage_decision(self, response: str) -> str:
        """Extract triage decision from model response"""
        
        # First, try structured format from fine-tuned models: "Triage decision: GP"
        triage_match = re.search(r'Triage\s+decision:\s*(ED|GP|HOME)', response, re.IGNORECASE)
        if triage_match:
            return triage_match.group(1).upper()
        
        # Also try alternative formats: "Triage Decision: ED" 
        triage_match = re.search(r'Triage\s+Decision:\s*(ED|GP|HOME)', response, re.IGNORECASE)
        if triage_match:
            return triage_match.group(1).upper()
        
        # Fallback to keyword search
        response_upper = response.upper()
        
        if "ED" in response_upper or "EMERGENCY" in response_upper:
            return "ED"
        elif "HOME" in response_upper or "SELF-CARE" in response_upper:
            return "HOME"
        elif "GP" in response_upper or "GENERAL PRACTITIONER" in response_upper:
            return "GP"
        
        # Content-based heuristics
        if any(word in response_upper for word in ["URGENT", "SEVERE", "IMMEDIATE", "CHEST PAIN"]):
            return "ED"
        elif any(word in response_upper for word in ["MILD", "REST", "MONITOR", "OTC"]):
            return "HOME"
        else:
            # Log when we can't extract triage decision - mark as incorrect
            logger.warning(f"⚠️ No triage decision found in response: {response[:100]}...")
            return "UNKNOWN"  # Mark as incorrect - no valid triage found
    
    def evaluate_single_case(self, 
                           model, 
                           tokenizer,
                           case: Dict[str, Any],
                           rag_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate a single case with comprehensive logging and LLM quality evaluation"""
        
        case_id = case["case_id"]
        query = case["query"]
        expected_triage = case["triage_decision"]
        expected_next_steps = case.get("next_steps", "")
        expected_reasoning = case.get("reasoning", "")
        
        logger.debug(f"🔍 Evaluating case {case_id}: {query[:100]}...")
        
        try:
            # Get RAG context if configured
            rag_time = 0.0
            context_length = 0
            if rag_config:
                context, rag_time = self.get_rag_context(query, rag_config)
                context_length = len(context)
                logger.debug(f"🔍 RAG context: {context_length} chars, {rag_time:.3f}s")
            else:
                context = ""
            
            # Create prompt
            prompt = self.create_prompt(query, context)
            
            # Generate response
            start_time = time.time()
            response = self.generate_response(model, tokenizer, prompt)
            inference_time = time.time() - start_time
            
            # Extract triage decision
            predicted_triage = self.extract_triage_decision(response)
            is_correct = predicted_triage == expected_triage
            
            logger.debug(f"🔍 Case {case_id}: {predicted_triage} vs {expected_triage} ({'✓' if is_correct else '✗'})")
            
            # Note: LLM judge evaluation is now handled separately in parallel
            # This method only does the main inference
            llm_judge_time = 0.0  # Will be filled in by parallel processing
            
            result = {
                "case_id": case_id,
                "query": query,
                "expected_triage": expected_triage,
                "predicted_triage": predicted_triage,
                "is_correct": is_correct,
                "response": response,
                "inference_time": inference_time,
                "rag_time": rag_time,
                "context_length": context_length,
                "success": True,
                "error": None,
                "expected_next_steps": expected_next_steps,
                "expected_reasoning": expected_reasoning,
                "llm_judge_time": llm_judge_time
            }
            
            # LLM judge scores will be added later by parallel processing
            # Initialize with default values
            result.update({
                "llm_next_step_quality_score": 0.0,
                "llm_reasoning_quality_score": 0.0,
                "llm_overall_quality_score": 0.0,
                "llm_next_step_rationale": "Parallel evaluation pending",
                "llm_reasoning_rationale": "Parallel evaluation pending",
                "llm_overall_rationale": "Parallel evaluation pending"
            })
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Case {case_id} failed: {e}")
            return {
                "case_id": case_id,
                "query": query,
                "expected_triage": expected_triage,
                "predicted_triage": "ERROR",
                "is_correct": False,
                "response": "",
                "inference_time": 0.0,
                "rag_time": 0.0,
                "context_length": 0,
                "success": False,
                "error": str(e),
                "expected_next_steps": expected_next_steps,
                "expected_reasoning": expected_reasoning,
                "llm_judge_time": 0.0,
                "llm_next_step_quality_score": 0.0,
                "llm_reasoning_quality_score": 0.0,
                "llm_overall_quality_score": 0.0,
                "llm_next_step_rationale": f"Evaluation failed: {str(e)}",
                "llm_reasoning_rationale": f"Evaluation failed: {str(e)}",
                "llm_overall_rationale": f"Evaluation failed: {str(e)}"
            }
    
    def evaluate_cases_parallel(self, model, tokenizer, cases: List[Dict[str, Any]], rag_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Evaluate multiple cases with parallel LLM judge processing"""
        
        logger.info(f"🚀 Starting parallel evaluation of {len(cases)} cases...")
        
        if not self.enable_llm_judge:
            # If LLM judge disabled, run sequential evaluation
            return [self.evaluate_single_case(model, tokenizer, case, rag_config) for case in cases]
        
        # Initialize parallel judge manager
        with ParallelLLMJudgeManager(judge_model=self.judge_model, max_judge_workers=self.max_judge_workers) as judge_manager:
            
            inference_results = []
            
            # Phase 1: Run inference and submit for parallel LLM evaluation
            logger.info("📊 Phase 1: Running inference and submitting for LLM evaluation...")
            for i, case in enumerate(cases, 1):
                case_id = case["case_id"]
                
                # Run inference (sequential for now to avoid model loading issues)
                inference_result = self.evaluate_single_case(model, tokenizer, case, rag_config)
                inference_results.append(inference_result)
                
                # Submit for parallel LLM evaluation
                if inference_result["success"]:
                    judge_manager.submit_for_evaluation(case_id, inference_result)
                
                # Progress logging
                if i % 50 == 0 or i == len(cases):
                    stats = judge_manager.get_statistics()
                    logger.info(f"   Progress: {i}/{len(cases)} inference, {stats['completed_judge']} LLM evaluations completed")
            
            # Phase 2: Wait for all LLM evaluations to complete
            logger.info("🔄 Phase 2: Waiting for LLM evaluations to complete...")
            llm_results = judge_manager.wait_for_all_pending(timeout=600.0)  # 10 minute timeout
            
            # Phase 3: Combine results
            logger.info("🔍 Phase 3: Combining results...")
            final_results = []
            
            for inference_result in inference_results:
                case_id = inference_result["case_id"]
                llm_scores = llm_results.get(case_id)
                
                # Update with LLM judge scores if available
                if llm_scores:
                    inference_result.update({
                        "llm_next_step_quality_score": llm_scores.next_step_quality_score,
                        "llm_reasoning_quality_score": llm_scores.reasoning_quality_score,
                        "llm_overall_quality_score": llm_scores.overall_score,
                        "llm_next_step_rationale": llm_scores.next_step_rationale,
                        "llm_reasoning_rationale": llm_scores.reasoning_rationale,
                        "llm_overall_rationale": llm_scores.overall_rationale
                    })
                else:
                    # Keep default values but mark as failed
                    inference_result.update({
                        "llm_next_step_rationale": "LLM evaluation failed or timed out",
                        "llm_reasoning_rationale": "LLM evaluation failed or timed out",
                        "llm_overall_rationale": "LLM evaluation failed or timed out"
                    })
                
                final_results.append(inference_result)
            
            # Final statistics
            final_stats = judge_manager.get_statistics()
            logger.info(f"✅ Parallel evaluation completed:")
            logger.info(f"   Inference: {len(final_results)} cases")
            logger.info(f"   LLM Judge: {final_stats['completed_judge']} cases")
            logger.info(f"   Avg LLM Judge Time: {final_stats['avg_judge_time']:.2f}s")
            logger.info(f"   Total LLM Judge Time: {final_stats['total_judge_time']:.1f}s")
            
            return final_results

class PerformanceCalculator:
    """Calculates performance metrics from evaluation results"""
    
    @staticmethod
    def calculate_metrics(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics including LLM judge scores"""
        
        if not case_results:
            return {
                "triage_accuracy": 0.0,
                "f1_score": 0.0,
                "f2_score": 0.0,
                "confusion_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                "classification_report": {},
                "timing_stats": {},
                "llm_judge_metrics": {}
            }
        
        # Filter successful results
        successful_results = [r for r in case_results if r["success"]]
        
        if not successful_results:
            logger.warning("⚠️  No successful results to calculate metrics")
            return PerformanceCalculator.calculate_metrics([])
        
        # Extract predictions and ground truth
        y_true = [r["expected_triage"] for r in successful_results]
        y_pred = [r["predicted_triage"] for r in successful_results]
        
        # Calculate accuracy
        correct_predictions = sum(1 for i, r in enumerate(successful_results) if r["is_correct"])
        accuracy = correct_predictions / len(successful_results)
        
        # Calculate F1 and F2 scores
        try:
            # Map labels to numbers for sklearn - include UNKNOWN as separate class
            label_map = {"ED": 0, "GP": 1, "HOME": 2, "UNKNOWN": 3}
            labels = ["ED", "GP", "HOME", "UNKNOWN"]
            
            y_true_encoded = [label_map.get(label, 1) for label in y_true]  # Default to GP for true labels
            y_pred_encoded = [label_map.get(label, 3) for label in y_pred]  # Default to UNKNOWN for predictions
            
            f1 = f1_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
            f2 = fbeta_score(y_true_encoded, y_pred_encoded, beta=2, average='weighted', zero_division=0)
            
            # Confusion matrix - 4x4 with UNKNOWN class
            cm = confusion_matrix(y_true_encoded, y_pred_encoded, labels=[0, 1, 2, 3])
            
            # Classification report
            class_report = classification_report(
                y_true_encoded, y_pred_encoded, 
                target_names=labels,
                labels=[0, 1, 2, 3],
                output_dict=True,
                zero_division=0
            )
            
        except Exception as e:
            logger.warning(f"⚠️  Error calculating sklearn metrics: {e}")
            f1 = accuracy
            f2 = accuracy
            cm = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            class_report = {}
        
        # Timing statistics
        inference_times = [r["inference_time"] for r in successful_results]
        rag_times = [r["rag_time"] for r in successful_results if r["rag_time"] > 0]
        context_lengths = [r["context_length"] for r in successful_results if r["context_length"] > 0]
        llm_judge_times = [r["llm_judge_time"] for r in successful_results if r.get("llm_judge_time", 0) > 0]
        
        timing_stats = {
            "total_inference_time": sum(inference_times),
            "avg_inference_time": np.mean(inference_times) if inference_times else 0.0,
            "avg_rag_time": np.mean(rag_times) if rag_times else 0.0,
            "avg_context_length": np.mean(context_lengths) if context_lengths else 0.0,
            "avg_llm_judge_time": np.mean(llm_judge_times) if llm_judge_times else 0.0,
            "total_llm_judge_time": sum(llm_judge_times) if llm_judge_times else 0.0
        }
        
        # LLM Judge statistics
        llm_judge_metrics = PerformanceCalculator._calculate_llm_judge_metrics(successful_results)
        
        # Count different types of failures
        unknown_triage_count = sum(1 for r in successful_results if r["predicted_triage"] == "UNKNOWN")
        error_count = len(case_results) - len(successful_results)
        
        return {
            "triage_accuracy": accuracy,
            "f1_score": f1,
            "f2_score": f2,
            "confusion_matrix": cm.tolist(),
            "classification_report": class_report,
            "timing_stats": timing_stats,
            "llm_judge_metrics": llm_judge_metrics,
            "cases_evaluated": len(successful_results),
            "success_count": len(successful_results),
            "error_count": error_count,
            "unknown_triage_count": unknown_triage_count,
            "total_failures": error_count + unknown_triage_count
        }
    
    @staticmethod
    def _calculate_llm_judge_metrics(successful_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate LLM judge quality metrics"""
        
        # Extract LLM judge scores
        next_step_scores = [r.get("llm_next_step_quality_score", 0) for r in successful_results 
                           if r.get("llm_next_step_quality_score") is not None]
        reasoning_scores = [r.get("llm_reasoning_quality_score", 0) for r in successful_results 
                           if r.get("llm_reasoning_quality_score") is not None]
        overall_scores = [r.get("llm_overall_quality_score", 0) for r in successful_results 
                         if r.get("llm_overall_quality_score") is not None]
        
        if not overall_scores:
            return {
                "avg_next_step_quality": 0.0,
                "avg_reasoning_quality": 0.0,
                "avg_overall_quality": 0.0,
                "std_overall_quality": 0.0,
                "min_overall_quality": 0.0,
                "max_overall_quality": 0.0,
                "cases_with_llm_evaluation": 0
            }
        
        return {
            "avg_next_step_quality": np.mean(next_step_scores) if next_step_scores else 0.0,
            "avg_reasoning_quality": np.mean(reasoning_scores) if reasoning_scores else 0.0,
            "avg_overall_quality": np.mean(overall_scores),
            "std_overall_quality": np.std(overall_scores),
            "min_overall_quality": np.min(overall_scores),
            "max_overall_quality": np.max(overall_scores),
            "cases_with_llm_evaluation": len(overall_scores)
        }
    
    @staticmethod
    def print_performance_summary(metrics: Dict[str, Any], config_name: str):
        """Print formatted performance summary"""
        
        logger.info(f"📊 PERFORMANCE SUMMARY: {config_name}")
        logger.info(f"   Triage Accuracy: {metrics['triage_accuracy']:.3f}")
        logger.info(f"   F1 Score: {metrics['f1_score']:.3f}")
        logger.info(f"   F2 Score: {metrics['f2_score']:.3f}")
        logger.info(f"   Cases Evaluated: {metrics['cases_evaluated']}")
        logger.info(f"   Success Rate: {metrics['success_count']}/{metrics['success_count'] + metrics['error_count']}")
        logger.info(f"   Unknown Triage Count: {metrics['unknown_triage_count']}")
        logger.info(f"   Total Failures: {metrics['total_failures']} (errors: {metrics['error_count']}, unknown: {metrics['unknown_triage_count']})")
        
        # Summary line with LLM judge score
        llm_metrics = metrics.get("llm_judge_metrics", {})
        if llm_metrics.get("avg_overall_quality", 0) > 0:
            logger.info(f"\n🏆 SUMMARY: Accuracy={metrics['triage_accuracy']:.3f}, F2={metrics['f2_score']:.3f}, LLM Quality={llm_metrics['avg_overall_quality']:.1f}/100")
        
        timing = metrics["timing_stats"]
        logger.info(f"   Avg Inference Time: {timing['avg_inference_time']:.3f}s")
        if timing['avg_rag_time'] > 0:
            logger.info(f"   Avg RAG Time: {timing['avg_rag_time']:.3f}s")
            logger.info(f"   Avg Context Length: {timing['avg_context_length']:.0f} chars")
        if timing.get('avg_llm_judge_time', 0) > 0:
            logger.info(f"   Avg LLM Judge Time: {timing['avg_llm_judge_time']:.3f}s")
        
        # Print LLM Judge metrics
        llm_metrics = metrics.get("llm_judge_metrics", {})
        if llm_metrics.get("cases_with_llm_evaluation", 0) > 0:
            logger.info(f"\n🧑‍⚕️ LLM JUDGE QUALITY SCORES:")
            logger.info(f"   Next Step Quality: {llm_metrics['avg_next_step_quality']:.1f}/100")
            logger.info(f"   Reasoning Quality: {llm_metrics['avg_reasoning_quality']:.1f}/100")
            logger.info(f"   Overall Quality: {llm_metrics['avg_overall_quality']:.1f}/100 (±{llm_metrics['std_overall_quality']:.1f})")
            logger.info(f"   Quality Range: {llm_metrics['min_overall_quality']:.1f}-{llm_metrics['max_overall_quality']:.1f}/100")
            logger.info(f"   Cases Evaluated by LLM: {llm_metrics['cases_with_llm_evaluation']}")