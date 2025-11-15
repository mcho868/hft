#!/usr/bin/env python3
"""
Tinfoil Agent vs. Rule-Based RAG Evaluation Script

This script tests two distinct medical triage approaches:
1. Tinfoil Agent as Generator: Uses RAG to find context, then calls the Tinfoil API (e.g., llama3-3-70b)
   to generate a full triage response (decision, next steps, reasoning). The quality of this
   response is then evaluated by an LLM-as-Judge.

2. Rule-Based RAG: Uses the same RAG pipeline to retrieve the top 10 relevant chunks, but
   determines the triage decision directly via a hard-coded rule based on the retrieved
   documents' metadata (ED > GP > Home priority). This approach involves no generative LLM.

Both approaches are tested against two RAG configurations:
- RAG_top10_structured_contextual_diverse
- RAG_top10_structured_pure_diverse

Usage:
  python test_tinfoil_vs_rule_based_rag.py          # Full test on all configurations
  python test_tinfoil_vs_rule_based_rag.py --test   # Test run on 1 configuration
  python test_tinfoil_vs_rule_based_rag.py --resume # Resume from latest progress
"""

import sys
import logging
import requests
import os
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict
from collections import Counter

# Import base classes from the existing framework
from comprehensive_triage_tester_llm_as_judge import Top5MedicalTriageTester, EvaluationConfig, EvaluationResult
from testing_core_llm_as_judge import TriageInferenceEngine, PerformanceCalculator
from llm_quality_judge import TinfoilLLMClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tinfoil_vs_rule_based_rag_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Main Tester Class ---

class TinfoilVsRuleBasedTester(Top5MedicalTriageTester):
    """
    A testing framework to compare a Tinfoil-powered generative agent against a
    non-generative, rule-based RAG triage system.
    """

    def __init__(self, base_dir: str = "/Users/choemanseung/789/hft"):
        super().__init__(base_dir)
        self.tinfoil_client = TinfoilLLMClient(model_name="llama3-3-70b")
        self.prompt_logged = False
        logger.info("✅ Initialized Tinfoil vs. Rule-Based Tester")

    def _define_top_rag_configs(self) -> List[Dict[str, Any]]:
        """
        Override to define the specific RAG configurations for this test,
        ensuring top_k is set to 10.
        """
        return [
            {
                "name": "top10_structured_contextual_diverse",
                "chunking_method": "structured_agent_tinfoil_medical",
                "retrieval_type": "contextual_rag",
                "bias_config": "diverse",
                "top_k": 10,
                "description": "Top 10 structured chunking + contextual RAG with diverse sources."
            },
            {
                "name": "top10_structured_pure_diverse",
                "chunking_method": "structured_agent_tinfoil_medical",
                "retrieval_type": "pure_rag",
                "bias_config": "diverse",
                "top_k": 10,
                "description": "Top 10 structured chunking + pure RAG with diverse sources."
            },
        ]

    def create_evaluation_matrix(self) -> List[EvaluationConfig]:
        """
        Creates the evaluation matrix for the Tinfoil vs. Rule-Based test.
        """
        evaluation_configs = []
        rag_configs = self._define_top_rag_configs()

        # First, add the new "No RAG" Tinfoil agent case
        no_rag_config = EvaluationConfig(
            model_name="tinfoil_agent_no_rag",
            model_path="N/A",
            adapter_path=None,
            rag_config=None,
            test_name="Tinfoil_Agent_NoRAG"
        )
        evaluation_configs.append(no_rag_config)

        for rag_config in rag_configs:
            # 1. Tinfoil Agent as Generator
            tinfoil_config = EvaluationConfig(
                model_name="tinfoil_agent",
                model_path="N/A",
                adapter_path=None,
                rag_config=rag_config,
                test_name=f"Tinfoil_Agent_{rag_config['name']}"
            )
            evaluation_configs.append(tinfoil_config)

            # 2. Rule-Based RAG
            rule_based_config = EvaluationConfig(
                model_name="rule_based_rag",
                model_path="N/A",
                adapter_path=None,
                rag_config=rag_config,
                test_name=f"Rule_Based_RAG_{rag_config['name']}"
            )
            evaluation_configs.append(rule_based_config)

        logger.info(f"✅ Created {len(evaluation_configs)} configurations for Tinfoil vs. Rule-Based test.")
        return evaluation_configs

    def _evaluate_single_configuration(self,
                                     config: EvaluationConfig,
                                     test_data: List[Dict[str, Any]]) -> EvaluationResult:
        """
        Router to select the correct evaluation method based on the config.
        """
        if config.model_name == "tinfoil_agent" or config.model_name == "tinfoil_agent_no_rag":
            logger.info(f"🚀 Running evaluation with Tinfoil Agent as generator (RAG: {'Yes' if config.rag_config else 'No'})...")
            return self._evaluate_with_tinfoil_agent(config, test_data)
        elif config.model_name == "rule_based_rag":
            logger.info("⚖️ Running evaluation with Rule-Based RAG...")
            return self._evaluate_with_rule_based_rag(config, test_data)
        else:
            raise ValueError(f"Unknown model_name in config: {config.model_name}")

    def _generate_with_tinfoil(self, query: str, context: str) -> str:
        """Generates a response using the Tinfoil API."""
        if context:
            prompt = f"""Patient query: {query}

Context:
{context}

Provide Triage decision, next steps and reasoing. answer in this exact format.

Triage Decison: (ED, GP, HOME)
Next Steps:
Reasoning:"""
        else:
            prompt = f"""Patient query: {query}

Provide Triage decision, next steps and reasoing. answer in this exact format.

Triage Decison: (ED, GP, HOME)
Next Steps:
Reasoning:"""

        if not self.prompt_logged:
            logger.info(f"Prompt being used for Tinfoil Agent:\n{prompt}")
            self.prompt_logged = True
        return self.tinfoil_client.complete(prompt)

    def _process_single_case_parallel(self, case: Dict[str, Any], config: EvaluationConfig, engine: Any) -> Dict[str, Any]:
        """
        Worker function to process a single case in parallel, including RAG and Tinfoil inference.
        """
        try:
            start_time = time.time()
            context, rag_time = engine.get_rag_context(case["query"], config.rag_config)
            response = self._generate_with_tinfoil(case["query"], context)
            inference_time = time.time() - start_time - rag_time
            predicted_triage = engine.extract_triage_decision(response)

            return {
                "case_id": case["case_id"],
                "query": case["query"],
                "expected_triage": case["triage_decision"],
                "predicted_triage": predicted_triage,
                "is_correct": predicted_triage == case["triage_decision"],
                "response": response,
                "inference_time": inference_time,
                "rag_time": rag_time,
                "context_length": len(context),
                "success": True,
                "error": None,
                "expected_next_steps": case.get("next_steps", ""),
                "expected_reasoning": case.get("reasoning", ""),
            }
        except Exception as e:
            logger.error(f"❌ Case {case.get('case_id')} failed during parallel processing: {e}")
            return {
                "case_id": case.get("case_id"),
                "query": case.get("query"),
                "expected_triage": case.get("triage_decision"),
                "predicted_triage": "ERROR",
                "is_correct": False,
                "response": "",
                "inference_time": 0.0,
                "rag_time": 0.0,
                "context_length": 0,
                "success": False,
                "error": str(e),
                "expected_next_steps": case.get("next_steps", ""),
                "expected_reasoning": case.get("reasoning", ""),
            }

    def _evaluate_with_tinfoil_agent(self, config: EvaluationConfig, test_data: List[Dict[str, Any]]) -> EvaluationResult:
        """
        Evaluates the Tinfoil Agent in parallel. It uses RAG for context, Tinfoil for generation,
        and an LLM-as-Judge for quality scoring.
        """
        from testing_core_llm_as_judge import TriageInferenceEngine, PerformanceCalculator
        from parallel_llm_judge import ParallelLLMJudgeManager
        from concurrent.futures import ThreadPoolExecutor, as_completed

        engine = TriageInferenceEngine(retriever=self.retriever, enable_llm_judge=True)
        self.prompt_logged = False # Reset for each new configuration
        total_cases = len(test_data)
        
        inference_results = []
        futures = []

        with ParallelLLMJudgeManager(judge_model="llama3-3-70b", max_judge_workers=200, queue_size=2000) as judge_manager:
            with ThreadPoolExecutor(max_workers=200) as executor:
                # Phase 1a: Submit all inference tasks to the thread pool
                logger.info(f"📊 Submitting {total_cases} cases to {executor._max_workers} inference workers...")
                for case in test_data:
                    futures.append(executor.submit(self._process_single_case_parallel, case, config, engine))

                # Phase 1b: Process completed inferences and submit to judge
                logger.info("🔄 Processing completed inferences and submitting for LLM evaluation...")
                for i, future in enumerate(as_completed(futures), 1):
                    if i % 200 == 0 or i == total_cases:
                        cases_left = total_cases - i
                        logger.info(f"Inference Progress: {i}/{total_cases} cases processed. {cases_left} cases left.")
                    
                    inference_result = future.result()
                    inference_results.append(inference_result)
                    if inference_result["success"]:
                        judge_manager.submit_for_evaluation(inference_result["case_id"], inference_result)

            # Phase 2: Wait for all LLM evaluations to complete
            logger.info("🔄 Phase 2: Waiting for all LLM evaluations to complete...")
            llm_results = judge_manager.wait_for_all_pending(timeout=600.0)

            # Phase 3: Combine results
            logger.info("🔍 Phase 3: Combining inference results with LLM Judge scores...")
            final_results = []
            for res in inference_results:
                case_id = res["case_id"]
                llm_scores = llm_results.get(case_id)
                if llm_scores:
                    res.update({
                        "llm_next_step_quality_score": llm_scores.next_step_quality_score,
                        "llm_reasoning_quality_score": llm_scores.reasoning_quality_score,
                        "llm_overall_quality_score": llm_scores.overall_score,
                    })
                final_results.append(res)

        metrics = PerformanceCalculator.calculate_metrics(final_results)

        return EvaluationResult(
            config=config, timestamp=datetime.now().isoformat(),
            triage_accuracy=metrics["triage_accuracy"], f1_score=metrics["f1_score"], f2_score=metrics["f2_score"],
            confusion_matrix=metrics["confusion_matrix"], classification_report=metrics["classification_report"],
            total_inference_time=metrics["timing_stats"]["total_inference_time"],
            avg_inference_time_per_case=metrics["timing_stats"]["avg_inference_time"],
            cases_evaluated=metrics["cases_evaluated"], success_count=metrics["success_count"], error_count=metrics["error_count"], error_details=[],
            llm_judge_metrics=metrics.get("llm_judge_metrics", {})
        )

    def _evaluate_with_rule_based_rag(self, config: EvaluationConfig, test_data: List[Dict[str, Any]]) -> EvaluationResult:
        """
        Evaluates the rule-based RAG approach. No LLM generation is involved.
        """
        from testing_core_llm_as_judge import PerformanceCalculator
        from optimized_hybrid_evaluator import BiasConfig

        case_results = []
        total_rag_time = 0
        total_cases = len(test_data)

        for i, case in enumerate(test_data, 1):
            if i % 200 == 0 or i == 1:
                cases_left = total_cases - i
                logger.info(f"Progress: {i}/{total_cases} cases evaluated. {cases_left} cases left.")

            start_time = time.time()

            # Correctly construct the BiasConfig object
            rag_config = config.rag_config
            bias_config_name = rag_config["bias_config"]
            if bias_config_name == "diverse":
                sources_config = {"healthify": 3, "mayo": 3, "nhs": 4}
            elif bias_config_name == "healthify_focused":
                sources_config = {"healthify": 5, "mayo": 2, "nhs": 3}
            else:
                sources_config = {"healthify": 3, "mayo": 3, "nhs": 4}
            
            bias_config = BiasConfig(
                name=bias_config_name,
                healthify=sources_config.get("healthify", 3),
                mayo=sources_config.get("mayo", 3), 
                nhs=sources_config.get("nhs", 4),
                description=f"Dynamic bias config: {bias_config_name}"
            )

            # We need the raw chunks from the retriever
            results_batch = self.retriever.search_batch_with_bias(
                queries=[case["query"]],
                bias_config=bias_config,
                chunking_method=rag_config["chunking_method"],
                retrieval_type=rag_config["retrieval_type"]
            )
            rag_time = time.time() - start_time
            total_rag_time += rag_time
            
            retrieved_chunks = results_batch[0] if results_batch else []

            # --- CORRECTED LOGIC: Parse triage level from chunk text ---
            triage_levels = []
            for chunk in retrieved_chunks:
                chunk_text = chunk.get('text', '')
                match = re.search(r"Triage level: (ED|GP|HOME)", chunk_text, re.IGNORECASE)
                if match:
                    triage_levels.append(match.group(1).upper())
                else:
                    triage_levels.append('UNKNOWN')
            
            counts = Counter(triage_levels)

            if counts['ED'] > 0:
                predicted_triage = 'ED'
            elif counts['GP'] > 0:
                predicted_triage = 'GP'
            elif counts['HOME'] > 0:
                predicted_triage = 'HOME'
            else:
                predicted_triage = 'UNKNOWN'

            result = {
                "case_id": case["case_id"],
                "query": case["query"],
                "expected_triage": case["triage_decision"],
                "predicted_triage": predicted_triage,
                "is_correct": predicted_triage == case["triage_decision"],
                "response": f"Rule-based decision from {len(retrieved_chunks)} chunks. Votes: {counts}",
                "inference_time": 0.001, # Placeholder for rule application time
                "rag_time": rag_time,
                "context_length": sum(len(c.get('text', '')) for c in retrieved_chunks),
                "success": True, "error": None,
            }
            case_results.append(result)

        metrics = PerformanceCalculator.calculate_metrics(case_results)
        
        return EvaluationResult(
            config=config, timestamp=datetime.now().isoformat(),
            triage_accuracy=metrics["triage_accuracy"], f1_score=metrics["f1_score"], f2_score=metrics["f2_score"],
            confusion_matrix=metrics["confusion_matrix"], classification_report=metrics["classification_report"],
            total_inference_time=total_rag_time, # For this test, inference time is RAG time
            avg_inference_time_per_case=metrics["timing_stats"]["avg_rag_time"],
            cases_evaluated=len(test_data), success_count=len(test_data), error_count=0, error_details=[],
            llm_judge_metrics=None # No LLM judge for rule-based
        )

    def run_comprehensive_evaluation(self,
                                   max_configs: Optional[int] = None,
                                   resume_from: Optional[str] = None) -> List[EvaluationResult]:
        """
        Run evaluation on the defined matrix. Overridden to use the correct matrix.
        """
        configs = self.create_evaluation_matrix() # Use our new matrix

        if max_configs:
            configs = configs[:max_configs]
        
        test_sample = self.test_data
        results = []
        start_index = 0
        total_configs = len(configs)

        if resume_from:
            results, start_index = self._load_progress_and_resume(resume_from, configs)
            if start_index > 0:
                logger.info(f"🔄 Resuming from configuration {start_index+1}/{total_configs}")

        progress_file = resume_from if resume_from else f"evaluation_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        for i, config in enumerate(configs[start_index:], start_index + 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"🔄 EVALUATION {i}/{total_configs}: {config.test_name}")
            logger.info(f"{'='*80}")

            try:
                result = self._evaluate_single_configuration(config, test_sample)
                results.append(result)
                logger.info(f"✅ COMPLETED {i}/{total_configs}: Acc={{result.triage_accuracy:.3f}}, F2={{result.f2_score:.3f}}")
                self._save_incremental_progress(results, progress_file, i, total_configs)
            except Exception as e:
                logger.error(f"❌ FAILED {i}/{total_configs}: {config.test_name} - {e}", exc_info=True)
                # Create and save error result
        
        logger.info("\n🎉 Tinfoil vs. Rule-Based RAG Testing Completed!")
        return results


def main():
    """Main execution function"""
    logger.info("🏆 Tinfoil Agent vs. Rule-Based RAG Configuration Testing 🏆")
    
    try:
        tester = TinfoilVsRuleBasedTester()
    except Exception as e:
        logger.error(f"❌ Failed to initialize tester: {e}", exc_info=True)
        return 1

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "--test":
            logger.info("🧪 Running TEST MODE (1 configuration)")
            results = tester.run_comprehensive_evaluation(max_configs=1)
        elif mode == "--resume":
            logger.info("🔄 Running RESUME MODE from hardcoded path...")
            resume_file = "/Users/choemanseung/789/hft/tinfoil_vs_rules_results_20251006_164242.json"
            if Path(resume_file).exists():
                logger.info(f"📂 Using progress file: {resume_file}")
                results = tester.run_comprehensive_evaluation(resume_from=resume_file)
            else:
                logger.error(f"❌ Hardcoded resume file not found: {resume_file}")
                results = tester.run_comprehensive_evaluation()
        else:
            logger.warning(f"⚠️ Unknown argument: {mode}. Running FULL MODE.")
            results = tester.run_comprehensive_evaluation()
    else:
        logger.info("🚀 Running FULL MODE")
        results = tester.run_comprehensive_evaluation()

    logger.info("🎉 Testing complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
