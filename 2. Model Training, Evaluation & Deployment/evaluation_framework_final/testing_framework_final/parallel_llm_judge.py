#!/usr/bin/env python3
"""
Parallel LLM-as-Judge Implementation for Medical Triage Testing

This module implements parallel processing where:
1. Main worker generates model responses
2. Separate LLM judge workers evaluate responses as they become available
3. Results are combined efficiently without blocking the main inference pipeline

Key Features:
- Queue-based parallel processing
- Non-blocking LLM judge evaluation
- Automatic result aggregation
- Progress tracking for both streams
"""

import queue
import threading
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime

from llm_quality_judge import MedicalTriageQualityJudge, ReasoningQualityScores

logger = logging.getLogger(__name__)

@dataclass
class EvaluationTask:
    """Task for LLM judge evaluation"""
    case_id: str
    case_data: Dict[str, Any]
    timestamp: float

@dataclass 
class EvaluationResult:
    """Combined result with inference and LLM judge scores"""
    case_id: str
    inference_result: Dict[str, Any]
    llm_judge_scores: Optional[ReasoningQualityScores]
    total_time: float

class ParallelLLMJudgeManager:
    """Manages parallel LLM judge evaluation alongside main inference"""
    
    def __init__(self, judge_model: str = "llama3-3-70b", max_judge_workers: int = 200, 
                 queue_size: int = 5000):
        self.judge_model = judge_model
        self.max_judge_workers = max_judge_workers
        self.queue_size = queue_size
        
        # Initialize components
        self.quality_judge = MedicalTriageQualityJudge(judge_model=judge_model, rate_limit_delay=0.05)
        
        # Queues for parallel processing
        self.evaluation_queue = queue.Queue(maxsize=queue_size)
        self.results_queue = queue.Queue()
        
        # Tracking
        self.completed_cases = {}
        self.pending_cases = set()
        self.judge_workers = []
        self.running = False
        
        # Statistics
        self.stats = {
            "total_cases": 0,
            "completed_inference": 0,
            "completed_judge": 0,
            "total_judge_time": 0.0,
            "avg_judge_time": 0.0
        }
        
        logger.info(f"🧑‍⚕️ Initialized Parallel LLM Judge Manager with {max_judge_workers} workers")
    
    def start_judge_workers(self):
        """Start background LLM judge worker threads"""
        self.running = True
        
        for i in range(self.max_judge_workers):
            worker = threading.Thread(
                target=self._judge_worker,
                name=f"LLMJudge-{i+1}",
                daemon=True
            )
            worker.start()
            self.judge_workers.append(worker)
        
        logger.info(f"✅ Started {len(self.judge_workers)} LLM judge workers")
    
    def stop_judge_workers(self):
        """Stop all judge worker threads"""
        self.running = False
        
        # Signal workers to stop
        for _ in range(len(self.judge_workers)):
            try:
                self.evaluation_queue.put(None, timeout=1.0)  # Poison pill
            except queue.Full:
                pass
        
        # Wait for workers to finish
        for worker in self.judge_workers:
            worker.join(timeout=5.0)
        
        self.judge_workers.clear()
        logger.info("🛑 Stopped all LLM judge workers")
    
    def submit_for_evaluation(self, case_id: str, inference_result: Dict[str, Any]):
        """Submit a case for LLM judge evaluation (non-blocking)"""
        
        if not self.running:
            logger.warning("⚠️ Judge workers not running, cannot submit case for evaluation")
            return
        
        # Prepare case data for judge
        case_data = {
            "case_id": case_id,
            "query": inference_result.get("query", ""),
            "predicted_triage": inference_result.get("predicted_triage", ""),
            "response": inference_result.get("response", ""),
            "expected_triage": inference_result.get("expected_triage", ""),
            "expected_next_steps": inference_result.get("expected_next_steps", ""),
            "expected_reasoning": inference_result.get("expected_reasoning", "")
        }
        
        task = EvaluationTask(
            case_id=case_id,
            case_data=case_data,
            timestamp=time.time()
        )
        
        try:
            self.evaluation_queue.put(task, timeout=1.0)
            self.pending_cases.add(case_id)
            self.stats["total_cases"] += 1
            logger.debug(f"📋 Submitted case {case_id} for LLM evaluation (queue size: {self.evaluation_queue.qsize()})")
        except queue.Full:
            logger.warning(f"⚠️ LLM judge queue full, skipping evaluation for case {case_id}")
    
    def get_completed_result(self, timeout: float = 1.0) -> Optional[EvaluationResult]:
        """Get a completed evaluation result (non-blocking with timeout)"""
        try:
            return self.results_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def wait_for_case(self, case_id: str, timeout: float = 30.0) -> Optional[ReasoningQualityScores]:
        """Wait for a specific case to complete LLM evaluation"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if case_id in self.completed_cases:
                result = self.completed_cases.pop(case_id)
                self.pending_cases.discard(case_id)
                return result
            
            # Check for any completed results
            completed_result = self.get_completed_result(timeout=0.5)
            if completed_result:
                if completed_result.case_id == case_id:
                    self.pending_cases.discard(case_id)
                    return completed_result.llm_judge_scores
                else:
                    # Store for later retrieval
                    self.completed_cases[completed_result.case_id] = completed_result.llm_judge_scores
        
        logger.warning(f"⏰ Timeout waiting for LLM evaluation of case {case_id}")
        return None
    
    def wait_for_all_pending(self, timeout: float = 120.0) -> Dict[str, ReasoningQualityScores]:
        """Wait for all pending cases to complete evaluation"""
        start_time = time.time()
        results = {}
        
        logger.info(f"⏳ Waiting for {len(self.pending_cases)} pending LLM evaluations...")
        
        while self.pending_cases and (time.time() - start_time < timeout):
            completed_result = self.get_completed_result(timeout=1.0)
            if completed_result:
                results[completed_result.case_id] = completed_result.llm_judge_scores
                self.pending_cases.discard(completed_result.case_id)
                
                if len(self.pending_cases) % 10 == 0:
                    logger.info(f"⏳ {len(self.pending_cases)} LLM evaluations remaining...")
        
        # Add any remaining completed cases
        results.update(self.completed_cases)
        self.completed_cases.clear()
        
        if self.pending_cases:
            logger.warning(f"⚠️ {len(self.pending_cases)} LLM evaluations did not complete within timeout")
        else:
            logger.info("✅ All LLM evaluations completed")
        
        return results
    
    def _judge_worker(self):
        """Background worker thread for LLM judge evaluation"""
        worker_name = threading.current_thread().name
        logger.info(f"🧑‍⚕️ {worker_name} started")
        
        while self.running:
            try:
                # Get task from queue
                task = self.evaluation_queue.get(timeout=1.0)
                
                if task is None:  # Poison pill
                    break
                
                logger.debug(f"🧑‍⚕️ {worker_name} evaluating case {task.case_id}")
                
                # Perform LLM evaluation
                start_time = time.time()
                llm_scores = self.quality_judge.evaluate_case(task.case_data)
                evaluation_time = time.time() - start_time
                
                # Update statistics
                self.stats["completed_judge"] += 1
                self.stats["total_judge_time"] += evaluation_time
                self.stats["avg_judge_time"] = self.stats["total_judge_time"] / self.stats["completed_judge"]
                
                # Create result
                result = EvaluationResult(
                    case_id=task.case_id,
                    inference_result=task.case_data,
                    llm_judge_scores=llm_scores,
                    total_time=evaluation_time
                )
                
                # Put result in results queue
                self.results_queue.put(result)
                
                logger.debug(f"✅ {worker_name} completed case {task.case_id} in {evaluation_time:.2f}s (Overall: {llm_scores.overall_score:.1f}/100)")
                
                # Mark task as done
                self.evaluation_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ {worker_name} error evaluating case: {e}")
                self.evaluation_queue.task_done()
        
        logger.info(f"🛑 {worker_name} stopped")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return {
            **self.stats,
            "queue_size": self.evaluation_queue.qsize(),
            "pending_cases": len(self.pending_cases),
            "completed_cases_cached": len(self.completed_cases),
            "workers_active": len([w for w in self.judge_workers if w.is_alive()])
        }
    
    def __enter__(self):
        """Context manager entry"""
        self.start_judge_workers()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_judge_workers()

# Convenience function for parallel evaluation
def run_parallel_evaluation(cases: List[Dict[str, Any]], 
                          inference_function: Callable,
                          max_judge_workers: int = 3,
                          progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
    """
    Run parallel evaluation with inference and LLM judge
    
    Args:
        cases: List of test cases
        inference_function: Function to run model inference
        max_judge_workers: Number of LLM judge workers
        progress_callback: Optional callback for progress updates
    
    Returns:
        List of combined results with LLM judge scores
    """
    
    results = []
    
    with ParallelLLMJudgeManager(max_judge_workers=max_judge_workers) as judge_manager:
        
        logger.info(f"🚀 Starting parallel evaluation of {len(cases)} cases")
        
        # Process cases
        for i, case in enumerate(cases, 1):
            case_id = case.get("case_id", f"case_{i}")
            
            # Run inference (main thread)
            inference_result = inference_function(case)
            judge_manager.stats["completed_inference"] += 1
            
            # Submit for LLM evaluation (parallel)
            judge_manager.submit_for_evaluation(case_id, inference_result)
            
            # Progress reporting
            if i % 200 == 0 or i == len(cases):
                stats = judge_manager.get_statistics()
                logger.info(f"📊 Progress: {i}/{len(cases)} inference, {stats['completed_judge']} LLM judge completed")
                
                if progress_callback:
                    progress_callback(i, len(cases), stats)
        
        logger.info("🔄 All inference completed, waiting for LLM evaluations...")
        
        # Wait for all LLM evaluations to complete
        llm_results = judge_manager.wait_for_all_pending(timeout=300.0)  # 5 minute timeout
        
        # Combine results
        for i, case in enumerate(cases):
            case_id = case.get("case_id", f"case_{i+1}")
            
            # Get inference result (we need to store these during the loop above)
            inference_result = inference_function(case)  # Re-run for now (could be optimized)
            
            # Get LLM judge result
            llm_scores = llm_results.get(case_id)
            
            # Combine results
            combined_result = inference_result.copy()
            if llm_scores:
                combined_result.update({
                    "llm_next_step_quality_score": llm_scores.next_step_quality_score,
                    "llm_reasoning_quality_score": llm_scores.reasoning_quality_score,
                    "llm_overall_quality_score": llm_scores.overall_score,
                    "llm_next_step_rationale": llm_scores.next_step_rationale,
                    "llm_reasoning_rationale": llm_scores.reasoning_rationale,
                    "llm_overall_rationale": llm_scores.overall_rationale
                })
            
            results.append(combined_result)
        
        # Final statistics
        final_stats = judge_manager.get_statistics()
        logger.info(f"✅ Parallel evaluation completed:")
        logger.info(f"   Inference: {final_stats['completed_inference']} cases")
        logger.info(f"   LLM Judge: {final_stats['completed_judge']} cases")
        logger.info(f"   Avg LLM Judge Time: {final_stats['avg_judge_time']:.2f}s")
    
    return results