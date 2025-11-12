#!/usr/bin/env python3
"""
Runner script for TOP 5 Medical Triage Configuration Testing with LLM-as-Judge

This script tests the top 5 performing configurations on the full test dataset WITH LLM QUALITY EVALUATION.
Based on evaluation results, these are:

1. SmolLM2-135M_4bit_high_capacity_safe_NoRAG (68.0% accuracy)
2. SmolLM2-135M_4bit_balanced_safe_NoRAG (59.5% accuracy) 
3. SmolLM2-135M_4bit_performance_safe_NoRAG (57.0% accuracy)
4. SmolLM2-135M_4bit_high_capacity_safe_RAG_top1_structured_contextual_diverse (55.0% accuracy)
5. SmolLM2-135M_4bit_high_capacity_safe_RAG_top2_structured_pure_diverse (54.0% accuracy)

Additional Features:
- LLM-as-Judge evaluation for reasoning and next step quality (0-100 scores)
- External evaluation using Tinfoil API (llama3-3-70b)
- Comprehensive quality metrics beyond just accuracy

Usage:
  python run_top5_test_llm_as_judge.py              # Full test on all 1975 cases with LLM judge
  python run_top5_test_llm_as_judge.py --test       # Test run on 1 configuration with LLM judge
  python run_top5_test_llm_as_judge.py --resume     # Resume from latest progress
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main execution function"""
    
    logger.info("🏆 TOP 5 MEDICAL TRIAGE CONFIGURATION TESTING WITH LLM-AS-JUDGE")
    logger.info("=" * 70)
    logger.info("Testing the best performing configurations on full test dataset")
    logger.info("Test dataset: 1975 cases")
    logger.info("Configurations: Top 5 from evaluation results")
    logger.info("🧑‍⚕️ LLM Judge: llama3-3-70b via Tinfoil API")
    logger.info("📊 Quality Metrics: Next Step Quality + Reasoning Quality (0-100 scores)")
    logger.info("=" * 70)
    
    # Import the LLM-as-judge tester
    try:
        from comprehensive_triage_tester_llm_as_judge import Top5MedicalTriageTester
    except ImportError as e:
        logger.error(f"❌ Failed to import LLM-as-judge tester: {e}")
        logger.error("Make sure you're in the correct directory and dependencies are installed")
        logger.error("Required: comprehensive_triage_tester_llm_as_judge.py, testing_core_llm_as_judge.py, llm_quality_judge.py")
        return 1
    
    # Initialize tester
    try:
        tester = Top5MedicalTriageTester()
        logger.info("✅ Tester initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize tester: {e}")
        return 1
    
    # Determine run mode
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "--test":
            logger.info("🧪 Running TEST MODE (1 configuration with LLM judge)")
            results = tester.run_comprehensive_evaluation(max_configs=1)
        elif mode == "--resume":
            logger.info("🔄 Running RESUME MODE (with LLM judge)")
            latest_progress = tester.find_latest_progress_file()
            if latest_progress:
                logger.info(f"📂 Found progress file: {latest_progress}")
                results = tester.run_comprehensive_evaluation(resume_from=latest_progress)
            else:
                logger.info("⚠️ No progress file found, starting fresh")
                results = tester.run_comprehensive_evaluation()
        else:
            logger.warning(f"⚠️ Unknown argument: {mode}")
            logger.info("🚀 Running FULL MODE (all 5 configurations with LLM judge)")
            results = tester.run_comprehensive_evaluation()
    else:
        logger.info("🚀 Running FULL MODE (all 5 configurations with LLM judge)")
        results = tester.run_comprehensive_evaluation()
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 FINAL TESTING SUMMARY")
    logger.info("=" * 60)
    
    if results:
        successful_results = [r for r in results if r.error_count == 0]
        failed_results = [r for r in results if r.error_count > 0]
        
        logger.info(f"✅ Successful configurations: {len(successful_results)}")
        logger.info(f"❌ Failed configurations: {len(failed_results)}")
        
        if successful_results:
            logger.info("\n🏆 TOP PERFORMERS WITH LLM QUALITY SCORES:")
            sorted_results = sorted(successful_results, key=lambda x: x.triage_accuracy, reverse=True)
            for i, result in enumerate(sorted_results, 1):
                logger.info(f"  {i}. {result.config.test_name}")
                logger.info(f"     Accuracy: {result.triage_accuracy:.3f}")
                logger.info(f"     F2 Score: {result.f2_score:.3f}")
                logger.info(f"     Unknown Cases: {result.unknown_triage_count}")
                logger.info(f"     Avg Time: {result.avg_inference_time_per_case:.3f}s")
                
                # Check if LLM judge metrics are available
                if hasattr(result, 'classification_report') and isinstance(result.classification_report, dict):
                    llm_metrics = result.classification_report.get('llm_judge_metrics', {})
                    if llm_metrics.get('avg_overall_quality', 0) > 0:
                        logger.info(f"     🧑‍⚕️ LLM Quality: {llm_metrics['avg_overall_quality']:.1f}/100")
                        logger.info(f"        Next Steps: {llm_metrics['avg_next_step_quality']:.1f}/100")
                        logger.info(f"        Reasoning: {llm_metrics['avg_reasoning_quality']:.1f}/100")
    else:
        logger.warning("⚠️ No results obtained")
    
    logger.info("=" * 70)
    logger.info("🎉 TOP 5 TESTING WITH LLM-AS-JUDGE COMPLETED!")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)